"""Run the fly: connectome + metabolism + arena, with a 3D viewer.

  python sim/run.py                       # serve the viewer on :5050
  python sim/run.py --headless 600        # run 600 simulated seconds, print a report
  python sim/run.py --headless 600 --seed 3 --csv out.csv

The simulation clock advances in 1 ms brain steps. It runs as fast as the CPU
allows and reports its own realtime factor; nothing is faked to keep up.
"""
import argparse, base64, json, math, os, sys, threading, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.brain import Brain, DT_MS
from sim.body import Body
from sim.world import World

NPZ = os.path.join("sim", "brain_malecns.npz")
GEOM = os.path.join("sim", "brain_geometry.npz")

# How many somata to stream activity for. All 139,662 are drawn; activity is sent
# for an evenly-spaced subsample to keep each frame small (one byte per neuron).
ACTIVITY_POINTS = 45000
PHYS_EVERY = 5      # world physics every 5 ms
BODY_EVERY = 20     # metabolism every 20 ms
PUBLISH_EVERY = 25  # push state to the viewer every 25 ms of sim time

# Dopaminergic drive injected into the real DAN population on reward/punishment.
DAN_DRIVE_MV = 1.2


class Sim:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.brain = Brain(NPZ, rng=rng)
        # Every motor readout is relative to rest, so measure rest first.
        print("  measuring resting baselines (1.5 s of simulated time) ...", flush=True)
        self.brain.measure_baseline([
            "mn_proboscis", "mn_wing", "mn_abdomen", "mn_neck",
            "mn_leg_front_L", "mn_leg_front_R", "mn_leg_mid_L", "mn_leg_mid_R",
            "mn_leg_hind_L", "mn_leg_hind_R",
            "descending", "dan", "mbon", "kenyon",
        ])
        bl = self.brain.baseline
        print(f"    proboscis {bl['mn_proboscis']:.1f} Hz, "
              f"legs ~{np.mean([bl[k] for k in bl if 'leg' in k]):.1f} Hz, "
              f"MBON {bl['mbon']:.1f} Hz", flush=True)
        print("  calibrating the steering pathway (dead-ahead odour) ...", flush=True)
        from sim.world import CUE_DRIVE
        refL, refR = self.brain.measure_steer_reference(CUE_DRIVE)
        print(f"    steering DNs under a symmetric odour: L {refL:.2f} Hz, R {refR:.2f} Hz",
              flush=True)
        self.body = Body(self.brain.N, rng=np.random.default_rng(seed + 100))
        self.world = World(self.brain, rng=np.random.default_rng(seed + 200))
        self.geom = None
        if os.path.exists(GEOM):
            g = np.load(GEOM, allow_pickle=True)
            sim_idx = g["sim_idx"]
            stride = max(1, len(sim_idx) // ACTIVITY_POINTS)
            self.act_sel = np.arange(0, len(sim_idx), stride)
            self.geom = {
                "xyz": g["xyz"], "group": g["group"],
                "groups": [str(x) for x in g["groups"]],
                "sim_idx": sim_idx,
                "act_stride": int(stride),
            }
            self._act_sim_idx = sim_idx[self.act_sel]
        self.lock = threading.Lock()
        self.state = {}
        self.running = True
        self.step_i = 0
        self.wall_t0 = time.time()
        self.rt_factor = 0.0
        self.motor = {}
        self.press_log = []
        self._last_presses = {p.name: 0 for p in self.world.pads}
        self._spikes_window = 0
        self._spikes_body = 0
        self._dop = 0.0

    def _publish(self):
        b, body, w = self.brain, self.body, self.world
        st = {
            "sim_t": round(self.step_i * DT_MS / 1000.0, 2),
            "realtime_factor": round(self.rt_factor, 3),
            "brain": {
                "n_neurons": b.N,
                "steer_pair": [round(v, 2) for v in b.steer_pair()],
                "steer_n": [
                    len(b.steer_idxL) if b.steer_idxL is not None else 0,
                    len(b.steer_idxR) if b.steer_idxR is not None else 0,
                ],
                "n_dead": b.n_dead,
                "spike_rate_hz": round(
                    self._spikes_window / max(1, PUBLISH_EVERY) * 1000.0, 0
                ),
                "learned": round(b.learned_depression, 4),
                "excitability": round(body.excitability, 3),
                "pops": {
                    k: round(b.pop_rate(k), 2)
                    for k in (
                        "mn_proboscis", "descending", "kenyon", "mbon", "dan",
                        "olfactory_L", "olfactory_R",
                        "grn_labellar", "grn_leg", "grn_tastepeg", "hygro",
                        "olfactory", "visual", "mn_leg_front_L", "mn_leg_front_R",
                    )
                },
            },
            "body": body.snapshot(),
            "world": w.snapshot(),
            "motor": {k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in self.motor.items()},
            "press_log": self.press_log[-20:],
        }
        if self.geom is not None:
            # rolling rate -> 0..255. Scale so ~25 Hz saturates.
            a = b.rate[self._act_sim_idx] * (1000.0 / DT_MS) * (255.0 / 25.0)
            np.clip(a, 0, 255, out=a)
            st["act"] = base64.b64encode(a.astype(np.uint8).tobytes()).decode("ascii")
        
        with self.lock:
            self.state = st
        self._spikes_window = 0

    def loop(self, max_steps=None, on_tick=None):
        b, body, w = self.brain, self.body, self.world
        while self.running and (max_steps is None or self.step_i < max_steps):
            w.apply_senses(body)

            # dopaminergic reinforcement from the previous physics tick
            if self._dop > 0.0:
                b.inject("dan", DAN_DRIVE_MV * self._dop)

            spiking = b.step(excitability=body.excitability)
            n_spk = len(spiking)
            self._spikes_window += n_spk
            self._spikes_body += n_spk
            b.update_plasticity(spiking, self._dop)
            self._dop *= 0.92

            self.step_i += 1

            if self.step_i % PHYS_EVERY == 0:
                self.motor = w.step(DT_MS * PHYS_EVERY, body)
                self._dop = max(
                    self._dop,
                    min(1.0, self.motor["reward"] + self.motor["punish"]),
                )
                for p in w.pads:
                    if p.presses > self._last_presses[p.name]:
                        self._last_presses[p.name] = p.presses
                        self.press_log.append({
                            "t": round(self.step_i * DT_MS / 1000.0, 1),
                            "pad": p.name,
                            "payload": p.payload,
                            "hunger": round(body.hunger, 3),
                            "thirst": round(body.thirst, 3),
                            "salt": round(body.salt_need, 3),
                        })

            if self.step_i % BODY_EVERY == 0:
                body.tick(
                    DT_MS * BODY_EVERY / 1000.0,
                    self._spikes_body,            # actual spikes since last tick
                    int(b.alive.sum()),
                    self.motor.get("drive", 0.0),
                    self.motor.get("ingesting", False),
                    b,
                )
                self._spikes_body = 0
                if not body.alive:
                    self._publish()
                    if on_tick:
                        on_tick(self)
                    return

            if self.step_i % PUBLISH_EVERY == 0:
                el = time.time() - self.wall_t0
                self.rt_factor = (self.step_i * DT_MS / 1000.0) / max(el, 1e-6)
                self._publish()
                if on_tick:
                    on_tick(self)


# ---------------------------------------------------------------- reporting
def report(sim):
    b, body, w = sim.brain, sim.body, sim.world
    print("\n" + "=" * 68)
    print(f"  simulated {sim.step_i*DT_MS/1000:.0f} s   "
          f"wall {time.time()-sim.wall_t0:.0f} s   "
          f"realtime x{sim.rt_factor:.2f}")
    print("=" * 68)
    s = body.snapshot()
    print(f"  {'ALIVE' if s['alive'] else 'DEAD - ' + str(s['cause_of_death'])}"
          f"   age {s['age_s']:.0f}s")
    print(f"  energy {s['energy']:+.3f}  crop {s['crop']:.3f}  water {s['water']:.3f}"
          f"  Na {s['sodium']:.3f}  damage {s['damage']:.3f}  mass {s['mass']:.3f}")
    print(f"  neurons dead {b.n_dead:,}/{b.N:,}   MB depression {b.learned_depression:.3f}")
    print(f"  ingested: " + "  ".join(f"{k}={v:.2f}" for k, v in s["ingested"].items()))
    print("\n  pad                presses   seconds_consumed")
    for p in w.pads:
        print(f"    {p.name:8} {p.payload:9} {p.presses:>5}   {p.consumed:>8.1f}")
    if sim.press_log:
        print("\n  presses vs internal state at the moment of pressing:")
        for e in sim.press_log[-14:]:
            print(f"    t={e['t']:>7.1f}s  {e['pad']:8} "
                  f"hunger={e['hunger']:.2f} thirst={e['thirst']:.2f} salt={e['salt']:.2f}")
    print(f"\n  mean population rates (Hz):")
    for k in ("mn_proboscis", "mn_leg_front_L", "mn_leg_front_R", "descending",
              "kenyon", "mbon", "dan", "olfactory", "visual"):
        print(f"    {k:18} {b.pop_rate(k):7.2f}")


# ------------------------------------------------------------------- server
def serve(sim, port):
    from flask import Flask, Response, send_from_directory

    here = os.path.dirname(os.path.abspath(__file__))
    app = Flask(__name__, static_folder=None)

    @app.route("/")
    def index():
        return send_from_directory(os.path.join(here, "static"), "arena.html")

    @app.route("/geometry")
    def geometry():
        if sim.geom is None:
            return Response(json.dumps({"n": 0}), mimetype="application/json")
        g = sim.geom
        return Response(
            json.dumps({
                "n": int(len(g["xyz"])),
                "xyz": base64.b64encode(
                    g["xyz"].astype(np.float32).tobytes()).decode("ascii"),
                "group": base64.b64encode(g["group"].tobytes()).decode("ascii"),
                "groups": g["groups"],
                "act_stride": g["act_stride"],
            }),
            mimetype="application/json",
        )

    @app.route("/state")
    def state():
        with sim.lock:
            return Response(json.dumps(sim.state), mimetype="application/json")

    @app.route("/stream")
    def stream():
        def gen():
            last = None
            while True:
                with sim.lock:
                    st = sim.state
                if st and st is not last:
                    last = st
                    yield f"data: {json.dumps(st)}\n\n"
                time.sleep(0.04)
        return Response(gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})

    t = threading.Thread(target=sim.loop, daemon=True)
    t.start()
    print(f"\n  fly is alive - open http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", type=float, default=None,
                    help="run N simulated seconds with no viewer, then report")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None, help="write a per-second trace")
    args = ap.parse_args()

    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print(f"loading connectome from {NPZ} ...")
    sim = Sim(seed=args.seed)
    print(f"  {sim.brain.N:,} neurons, {len(sim.brain.edge_post):,} synapses")

    if args.headless is None:
        serve(sim, args.port)
        return

    rows = []
    last_logged = [-1]

    def on_tick(s):
        t = int(s.step_i * DT_MS / 1000.0)
        if t != last_logged[0]:
            last_logged[0] = t
            bs = s.body.snapshot()
            rows.append({
                "t": t, "energy": bs["energy"], "water": bs["water"],
                "sodium": bs["sodium"], "damage": bs["damage"], "mass": bs["mass"],
                "dead": s.brain.n_dead, "learned": s.brain.learned_depression,
                "spike_hz": s.state.get("brain", {}).get("spike_rate_hz", 0),
                "proboscis": round(s.brain.pop_rate("mn_proboscis"), 2),
                "drive": round(s.motor.get("drive", 0.0), 3),
                **{p.name: p.presses for p in s.world.pads},
            })
            if t % 60 == 0:
                print(f"  t={t:>5}s  energy={bs['energy']:+.3f} water={bs['water']:.3f} "
                      f"dead={s.brain.n_dead:>6,} x{s.rt_factor:.2f} realtime", flush=True)

    sim.loop(max_steps=int(args.headless * 1000 / DT_MS), on_tick=on_tick)
    report(sim)

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            wtr = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"\n  trace -> {args.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
