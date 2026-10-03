"""Serve the NeuroMechFly simulation: dish view, eye view, telemetry.

  python sim/nmf/serve.py [--port 5060]

Streams two MJPEG feeds - the dish from above and what the fly's left eye
actually sees, reconstructed from its 721 ommatidia - plus a JSON telemetry
endpoint.

Physics settings are the ones the stability work landed on: MuJoCo's
`implicitfast` integrator and a 1e-5 timestep. Explicit Euler at 1e-4 integrated
the actuator damping unstably (kv*dt/I about 89, where the limit is 2) and the
fly was flung across the dish at 130 mm/s with no command given.
"""
from __future__ import annotations

import argparse
import base64
import io
import math
import os
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flygym import Simulation
from flygym.utils.math import Rotation3D
from PIL import Image

from sim.nmf.gait import TripodGait
from flygym.vision.retina import Retina

from sim.nmf.world import (DISH_R, PADS, Dish, add_wall_contacts, build_fly,
                           color_fly, fix_clipping)
from sim.brain import DT_MS, Brain
from sim.nmf.olfaction import CUE_DRIVE, PadOdour
from sim.world import LIGHT_DRIVE, MAX_SPEED, WALL_DRIVE

# -- brain -> body ------------------------------------------------------------
# Leg motor pools rest at 12-22 Hz while the brain as a whole rests near 2 Hz, so
# every readout here is a CHANGE from that pool's own measured baseline.
LEG_RATE_REF = 10.0        # delta-Hz counting as full forward drive
# Baseline leg-motor firing means "standing", so a resting fly reads out zero
# forward drive and would never leave the spot it started on. Real flies walk
# spontaneously; a static connectome does not capture the neuromodulation that
# makes them. This bias is an explicit model term, not something measured.
FORAGE_BIAS = 0.35
# Turn command is -1..1 for the gait, where 1 is about 52 deg/s. The steering
# signal is the adapted left-minus-right of a 182-neuron lateralised descending
# ensemble, measured in sim/probe_steering.py - not DNa01/DNa02 alone, which is
# four cells and pure shot noise.
STEER_GAIN = 1.9
IMBALANCE_GAIN = 0.25      # leg-pool left/right asymmetry
WANDER_YAW = 0.10          # exploration, so a quiet network still moves
MOTOR_POOLS = [
    "mn_leg_front_L", "mn_leg_mid_L", "mn_leg_hind_L",
    "mn_leg_front_R", "mn_leg_mid_R", "mn_leg_hind_R",
    "mn_proboscis",
]

BRAIN_NPZ = os.path.join("sim", "brain_malecns.npz")
GEOM_NPZ = os.path.join("sim", "brain_geometry.npz")
# How many somata stream their firing rate to the page. All 139,662 are drawn;
# this is how many carry live activity, strided across the set.
ACTIVITY_POINTS = 45000

# 5e-5 measured: 0.28x realtime, zero passive drift, 4 ground contacts and a
# 7 mm/s walk. 1e-5 was 35x slower for no stability gain once the integrator was
# right, and made the view unwatchably choppy.
TIMESTEP = 5e-5
INTEGRATOR_IMPLICITFAST = 3
# kv=15 was the single thing stopping this fly from walking. A position servo has
# to supply kv*velocity just to move at all, and at the gait's joint speeds that
# demand exceeded what kp could deliver against the actuators' +/-30 force limit,
# so the legs lagged their targets by up to a radian and the feet never found the
# floor. Measured: 0.63 of 6 feet on the ground at kv=15, 3.23 at kv=1.
KP, KV = 300.0, 1.0
# Measured per rendered frame, after the gait work roughly doubled the physics
# rate to 0.48x realtime: a camera costs about 8 ms REGARDLESS of resolution
# (8.3 ms at 360x480, 8.9 ms at 480x640), because the cost is scene setup rather
# than pixels, and the JPEG encode is another 1-2 ms for 5 kB. So the frame rate
# is what to spend on, and resolution is nearly free. 0.07 s is ~14 fps for the
# main view and the eye view together, against the 6 fps that looked choppy.
RENDER_EVERY_S = 0.07          # wall-clock seconds between rendered frames
# The ommatidial readout is the one genuinely expensive view. The readout itself
# is 29 ms (the hex-to-image conversion is only 0.2 ms of that), but it also forces
# an extra GPU pass over both eyes, and showing it at 1.25 Hz halved the whole
# simulation rate: 0.194x realtime with the dish and eye views alone against 0.096x
# with this one added. It changes slowly, so it gets its own much slower cadence.
OMMA_EVERY_S = 1.2
# 2.6x the pixels of the 260x350 this started at. Resolution is not free - the
# same scene costs 0.167x realtime at 360x480, 0.136x here and 0.115x at 480x600 -
# but the frame rate still trebles against the old 6 fps, and the view is the
# difference between a legible fly and a smudge.
CAM_RES = (420, 560)
JPEG_QUALITY = 88
# Joint targets do not need recomputing at the physics rate. The gait is pure
# Python, and calling it 100,000 times per simulated second made it, not MuJoCo,
# the bottleneck. 2 kHz is far above the ~7 Hz stepping rhythm.
CONTROL_EVERY = 50             # physics steps between control updates (2.5 ms)
# How long a camera keeps being rendered after the page last asked for it.
WANT_TTL_S = 2.0
CAMERAS = {
    "hover": "fly/hover",          # above and behind: body, heading and sky
    "chase": "fly/chase",          # low and close: watch the legs
    "dish": "dish_angled",
    "top": "dish_top",
    "eye": "fly/l_eye_cam_camera",
}


class NMFSim:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.fly, self.skeleton = build_fly(with_vision=True)
        self.dish = Dish()
        self.dish.add_fly(
            self.fly,
            spawn_position=np.array([0.0, 0.0, 2.0]),
            spawn_rotation=Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)),
            add_ground_contact_sensors=False,
        )
        self.n_wall_pairs = add_wall_contacts(self.dish)
        self.sim = Simulation(self.dish)
        self.sim.reset()
        self.sim.mj_model.opt.integrator = INTEGRATOR_IMPLICITFAST
        self.sim.mj_model.opt.timestep = TIMESTEP
        self._tune()
        color_fly(self.sim)
        self.znear_mm, self.zfar_mm = fix_clipping(self.sim)
        print(f"  dish wall: {self.n_wall_pairs} fly/wall contact pairs", flush=True)
        for _ in range(int(0.6 / TIMESTEP)):      # let it settle onto its feet
            self.sim.step()
        self.gait = TripodGait(self.sim)
        # The renderer is created inside loop(), on the thread that will use it:
        # MuJoCo's GL context is thread-affine and binding it on the main thread
        # then rendering from the simulation thread raises
        # "WGL: Failed to make context current".
        self.renderer = None
        # `Simulation.retina` is not the Retina helper, so build one directly.
        # Without this the ommatidial view silently produced nothing.
        self.retina = Retina()

        self._load_brain()

        self.lock = threading.Lock()
        self.jpeg = {k: None for k in list(CAMERAS) + ["omma"]}
        # Only cameras the page is actually asking for get rendered, and they are
        # dropped again once it stops asking. Rendering all six every frame cost
        # more than the physics did.
        now = time.time()
        self.wanted = {"hover": now, "eye": now}
        self.state = {}
        self.sim_t = 0.0
        self.wall0 = time.time()
        self.drive, self.turn = 0.6, 0.0
        # Both default on. `control` is the brain driving the legs; `smell` is
        # whether the pads emit any odour. Turning smell off with control on is
        # the control arm of the chemotaxis assay.
        self.control = True
        self.smell = True
        self.running = True
        # Wall-clock seconds spent in each part of the loop, so the cost of the
        # views can be read off a running server instead of guessed at.
        self.prof = dict(gait=0.0, phys=0.0, render=0.0, jpeg=0.0, omma=0.0,
                         state=0.0, loop=0.0, n=0, rcall=0.0, rebuilds=0,
                         brain=0.0)
        self.prof_t0 = time.time()

    def _load_brain(self):
        """The MaleCNS connectome, and the soma positions used to draw it.

        The brain costs about 0.69 ms of wall clock per millisecond simulated
        (1.4x realtime on its own), so it is affordable next to the physics. It is
        stepped on simulated time, in lockstep with the body, rather than on a
        timer of its own.
        """
        self.brain = None
        self.geom = None
        self.odour = None
        self.motor = {}
        if not os.path.exists(BRAIN_NPZ):
            print(f"  no {BRAIN_NPZ}; brain view disabled "
                  f"(build it with sim/build_brain.py)", flush=True)
            return
        t0 = time.time()
        self.brain = Brain(BRAIN_NPZ, rng=np.random.default_rng(7))
        print(f"  brain: {self.brain.N:,} neurons loaded in {time.time() - t0:.1f} s",
              flush=True)
        print("  giving the pads their odours and measuring motor baselines ...",
              flush=True)
        self.odour = PadOdour(self.brain, self.dish.pad_xy,
                              [n for (n, _p, _c) in PADS],
                              rng=np.random.default_rng(2))
        self.brain.measure_baseline(MOTOR_POOLS)
        bl = self.brain.baseline
        print(f"    legs rest at ~"
              f"{np.mean([v for k, v in bl.items() if 'leg' in k]):.1f} Hz, "
              f"proboscis {bl['mn_proboscis']:.1f} Hz", flush=True)
        if not os.path.exists(GEOM_NPZ):
            print(f"  no {GEOM_NPZ}; brain drawn without activity", flush=True)
            return
        g = np.load(GEOM_NPZ, allow_pickle=True)
        sim_idx = g["sim_idx"]
        stride = max(1, len(sim_idx) // ACTIVITY_POINTS)
        sel = np.arange(0, len(sim_idx), stride)
        self.geom = {
            "xyz": np.asarray(g["xyz"], dtype=np.float32),
            "group": np.asarray(g["group"], dtype=np.uint8),
            "groups": [str(x) for x in g["groups"]],
            "act_stride": int(stride),
        }
        self._act_sim_idx = sim_idx[sel]
        self._brain_ms = 0.0
        print(f"  geometry: {len(self.geom['xyz']):,} somata, "
              f"{len(sel):,} streaming activity", flush=True)

    def _read_motor(self):
        """Forward drive and turn command, from the brain's own motor output."""
        b = self.brain
        left = float(np.mean([b.delta_rate("mn_leg_front_L"),
                              b.delta_rate("mn_leg_mid_L"),
                              b.delta_rate("mn_leg_hind_L")]))
        right = float(np.mean([b.delta_rate("mn_leg_front_R"),
                               b.delta_rate("mn_leg_mid_R"),
                               b.delta_rate("mn_leg_hind_R")]))
        leg = float(np.clip((left + right) / (2 * LEG_RATE_REF), 0.0, 1.0))
        imbalance = float(np.clip((right - left) / LEG_RATE_REF, -1.0, 1.0))
        drive = float(np.clip(leg + FORAGE_BIAS, 0.0, 1.0))
        steer = float(b.steer_signal())
        turn = float(np.clip(
            STEER_GAIN * steer
            + IMBALANCE_GAIN * imbalance
            + WANDER_YAW * float(self.rng.normal(0.0, 1.0)),
            -1.0, 1.0))
        self.motor = {"leg_delta_L": round(left, 2), "leg_delta_R": round(right, 2),
                      "steer": round(steer, 4),
                      "proboscis_delta": round(float(b.delta_rate("mn_proboscis")), 2)}
        return drive, turn

    def _step_brain(self, sim_ms, speed_mm_s, touching_wall):
        """Advance the brain by `sim_ms` of simulated time, in 1 ms steps.

        The sensory input is what this body can actually report: ambient light,
        proprioception scaled by how fast the fly is walking, and bristle
        mechanoreception when it is against the wall. No odour is injected here -
        the request pads are part of sim/world.py's arena, not this one - so the
        activity shown is the resting network plus locomotor drive.
        """
        if self.brain is None:
            return
        self._brain_ms += sim_ms
        n = int(self._brain_ms / DT_MS)
        self._brain_ms -= n * DT_MS
        d = self.sim.mj_data
        x, y, th = float(d.qpos[0]), float(d.qpos[1]), self._heading()
        for _ in range(min(n, 8)):          # cap: never spiral if we fall behind
            self.brain.clear_input()
            self.brain.inject("visual", LIGHT_DRIVE)
            self.brain.inject("mechano_proprio",
                              0.25 + 0.55 * min(1.0, speed_mm_s / MAX_SPEED))
            if touching_wall:
                self.brain.inject("mechano_tactile", WALL_DRIVE)
            if self.odour is not None and self.smell:
                self.odour.sense(x, y, th)
            self.brain.step(excitability=1.0)

    def _activity_b64(self):
        """Firing rates of the streamed somata as base64 uint8, 25 Hz saturating."""
        if self.brain is None or self.geom is None:
            return None
        a = self.brain.rate[self._act_sim_idx] * (1000.0 / DT_MS) * (255.0 / 25.0)
        np.clip(a, 0, 255, out=a)
        return base64.b64encode(a.astype(np.uint8).tobytes()).decode("ascii")

    def _tune(self):
        m = self.sim.mj_model
        for i in range(m.nu):
            if m.actuator(i).name.endswith("-adhesion"):
                m.actuator_ctrlrange[i] = [0.0, 1.0]
                m.actuator_ctrllimited[i] = 1
                continue
            m.actuator_gainprm[i][0] = KP
            m.actuator_biasprm[i][1] = -KP
            m.actuator_biasprm[i][2] = -KV
            m.actuator_ctrlrange[i] = [-np.pi, np.pi]
            m.actuator_ctrllimited[i] = 0

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _to_jpeg(arr, quality=JPEG_QUALITY):
        if arr is None:
            return None
        a = np.asarray(arr)
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def _ommatidia_image(self):
        """What the fly actually sees: 721 facets, not a camera image."""
        try:
            om = np.asarray(self.sim.get_ommatidia_readouts("fly"))
        except Exception:
            return None
        if self.retina is None:
            return None
        try:
            left = om[0]                            # (721, 2), two photoreceptor types
            img = np.asarray(
                self.retina.hex_pxls_to_human_readable(left, color_8bit=True)
            )
            if img.ndim == 3:                        # collapse the two channels
                img = img.max(axis=-1)
            return img
        except Exception as e:
            print("ommatidia render failed:", type(e).__name__, e, flush=True)
            return None

    def profile_report(self):
        """Share of wall-clock time in each part of the loop, as percentages."""
        wall = max(time.time() - self.prof_t0, 1e-6)
        out = {k: round(v / wall * 100, 1) for k, v in self.prof.items()
               if k not in ("n", "loop", "rebuilds", "rcall")}
        out["ticks_per_s"] = round(self.prof["n"] / wall, 1)
        out["unaccounted"] = round(
            100 - sum(v for k, v in out.items()
                      if k not in ("ticks_per_s", "unaccounted")), 1)
        out["_rcall"] = round(self.prof["rcall"] / wall * 100, 1)
        out["_rebuilds"] = self.prof["rebuilds"]
        return out

    def _make_renderer(self):
        """The one renderer, built on the thread that will use it."""
        self.renderer = mujoco.Renderer(self.sim.mj_model, CAM_RES[0], CAM_RES[1])
        self.cam_id = {
            k: mujoco.mj_name2id(self.sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
            for k, cam in CAMERAS.items()
        }
        missing = [k for k, v in self.cam_id.items() if v < 0]
        if missing:
            print("cameras not found in the model:", missing, flush=True)
        self.prof["rebuilds"] += 1

    def _render(self, key):
        """Render one camera. Returns an HxWx3 uint8 array, or None."""
        cid = self.cam_id.get(key, -1)
        if cid < 0:
            return None
        self.renderer.update_scene(self.sim.mj_data, camera=cid)
        return self.renderer.render()

    def _heading(self):
        w, x, y, z = self.sim.mj_data.qpos[3:7]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    # -- main loop ---------------------------------------------------------
    def loop(self):
        self._make_renderer()
        next_render = 0.0
        next_omma = 0.0
        turn_timer = 0.0
        k = 0
        ctrl_dt = TIMESTEP * CONTROL_EVERY
        while self.running:
            k += 1
            t_a = time.time()
            # Drive and turn now come from the brain itself, read after it has
            # been stepped (below). Until the first step they stay at their
            # initial values. If the brain failed to load, fall back to the
            # wander so the body still does something visible.
            if self.brain is None:
                turn_timer -= ctrl_dt
                if turn_timer <= 0:
                    self.turn = float(self.rng.normal(0, 0.45))
                    turn_timer = float(self.rng.uniform(0.4, 1.4))
            self.gait.step(ctrl_dt, self.drive, self.turn)
            self.gait.apply()
            t_b = time.time()
            # Run the whole control interval inside MuJoCo. Stepping one at a
            # time from Python made the interpreter, not the physics, the
            # bottleneck and held the GIL hard enough to starve the web server.
            mujoco.mj_step(self.sim.mj_model, self.sim.mj_data, nstep=CONTROL_EVERY)
            t_c = time.time()
            d0 = self.sim.mj_data
            speed = float(np.linalg.norm(d0.qvel[:2]))
            near_wall = float(np.hypot(d0.qpos[0], d0.qpos[1])) > DISH_R - 3.0
            self._step_brain(ctrl_dt * 1000.0, speed, near_wall)
            if self.brain is not None and self.control:
                self.drive, self.turn = self._read_motor()
            t_br = time.time()
            self.prof["brain"] += t_br - t_c
            self.prof["gait"] += t_b - t_a
            self.prof["phys"] += t_c - t_b
            self.prof["n"] += 1
            self.sim_t += ctrl_dt
            # No explicit yield here. mj_step releases the GIL for the whole batch,
            # which is already several milliseconds, so the web threads get their
            # turn without one - and on Windows a yield hands any runnable thread a
            # full ~15 ms scheduler timeslice, so yielding once per 2.5 ms of
            # simulation cost most of the throughput: 0.07x realtime against 0.34x
            # for the same loop with nobody polling it.

            now = time.time()
            if now >= next_render:
                next_render = now + RENDER_EVERY_S
                with self.lock:
                    # forget cameras the page has stopped polling for
                    self.wanted = {k: t for k, t in self.wanted.items()
                                   if now - t < WANT_TTL_S}
                    if not self.wanted:
                        self.wanted = {"hover": now}
                    want = set(self.wanted)
                # the ommatidial view is derived from the eye camera, so that one
                # has to be rendered whenever the retina panel is being shown
                do_omma = "omma" in want and now >= next_omma
                if do_omma:
                    next_omma = now + OMMA_EVERY_S
                    want.add("eye")
                want.discard("omma")
                frames = {}
                t_r = time.time()
                if want:
                    t_e = time.time()
                    for key in want:
                        arr = self._render(key)
                        if arr is not None:
                            frames[key] = arr
                    t_rc = time.time()
                    self.prof["rcall"] += t_rc - t_e
                t_r2 = time.time()
                self.prof["render"] += t_r2 - t_r
                omma = self._ommatidia_image() if do_omma else None
                t_o = time.time()
                self.prof["omma"] += t_o - t_r2
                d = self.sim.mj_data
                t_j = time.time()
                with self.lock:
                    for key, arr in frames.items():
                        self.jpeg[key] = self._to_jpeg(arr)
                    if omma is not None:
                        self.jpeg["omma"] = self._to_jpeg(omma)
                    self.state = {
                        "sim_t": round(self.sim_t, 3),
                        "realtime": round(self.sim_t / max(now - self.wall0, 1e-6), 4),
                        "pos": [round(float(d.qpos[0]), 2), round(float(d.qpos[1]), 2)],
                        "z": round(float(d.qpos[2]), 3),
                        "heading_deg": round(math.degrees(self._heading()), 1),
                        "speed": round(float(np.linalg.norm(d.qvel[:2])), 2),
                        "contacts": int(d.ncon),
                        "drive": round(self.drive, 2),
                        "turn": round(self.turn, 2),
                        "dish_r": DISH_R,
                        "pads": [
                            {"name": n, "x": round(x, 1), "y": round(y, 1)}
                            for (n, _p, _c), (x, y) in zip(PADS, self.dish.pad_xy)
                        ],
                        "prof": self.profile_report(),
                    }
                    if self.brain is not None:
                        self.state["brain"] = {
                            "n": int(self.brain.N),
                            "rate": round(float(self.brain.rate.mean()) * 1000.0, 2),
                            "alive": int(self.brain.alive.sum()),
                            "control": bool(self.control),
                            "smell": bool(self.smell),
                            "near": list(self.odour.near) if self.odour else [],
                            **self.motor,
                        }
                        act = self._activity_b64()
                        if act is not None:
                            self.state["act"] = act
                self.prof["state"] += time.time() - t_j
            self.prof["loop"] += time.time() - t_a



def make_app(sim: NMFSim):
    from flask import Flask, Response, jsonify, send_from_directory

    here = Path(__file__).resolve().parent
    app = Flask(__name__, static_folder=None)

    @app.route("/")
    def index():
        return send_from_directory(here / "static", "view.html")

    @app.route("/state")
    def state():
        with sim.lock:
            return jsonify(sim.state)

    # Single-shot JPEG endpoints rather than MJPEG streams. Three long-lived
    # multipart responses plus the telemetry poll exhausted the browser's
    # per-host connection limit and the page failed to load.
    def frame(key):
        with sim.lock:
            # remember what the page is watching so the loop renders only that
            if key in CAMERAS or key == "omma":
                sim.wanted[key] = time.time()
            buf = sim.jpeg.get(key)
        if buf is None:
            return Response(status=503)
        return Response(buf, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.route("/geometry")
    def geometry():
        """Soma positions and group tags for the brain view, sent once."""
        import json
        g = sim.geom
        if g is None:
            return Response(json.dumps({"n": 0}), mimetype="application/json")
        return Response(
            json.dumps({
                "n": int(len(g["xyz"])),
                "xyz": base64.b64encode(g["xyz"].tobytes()).decode("ascii"),
                "group": base64.b64encode(g["group"].tobytes()).decode("ascii"),
                "groups": g["groups"],
                "act_stride": g["act_stride"],
            }),
            mimetype="application/json",
        )

    @app.route("/frame/<key>")
    def f_any(key):
        return frame(key)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5060)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("building NeuroMechFly and the dish ...", flush=True)
    sim = NMFSim(seed=args.seed)
    print(f"  {sim.sim.mj_model.nu} actuators, {sim.sim.mj_model.ngeom} geoms, "
          f"dish radius {DISH_R} mm", flush=True)
    threading.Thread(target=sim.loop, daemon=True).start()
    app = make_app(sim)
    print(f"\n  open http://127.0.0.1:{args.port}\n", flush=True)
    app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
