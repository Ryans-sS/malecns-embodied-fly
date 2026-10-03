"""Does the embodied fly actually steer towards the pads, or just wander into them?

The brain's steering output drives a physical body here, so the question has to be
asked of the whole loop rather than of the brain alone. Odour ON is compared
against odour OFF with everything else identical - same seed, same starting pose,
same wander noise - because an unpaired comparison buries this effect under its
own noise.

  python sim/nmf/test_chemotaxis.py [trials] [seconds]
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.stdout.reconfigure(encoding="utf-8")

import mujoco

from sim.nmf.serve import NMFSim, TIMESTEP, CONTROL_EVERY
from sim.nmf.world import PAD_R

TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0


def trial(sim, smell, seed, secs):
    """One run. Returns closest approach, time on a pad, and distance travelled."""
    sim.smell = smell
    sim.control = True
    sim.rng = np.random.default_rng(seed)
    sim.brain.rng = np.random.default_rng(seed + 1)
    # same start every time: centre of the dish, facing +x
    d = sim.sim.mj_data
    mujoco.mj_resetData(sim.sim.mj_model, d)
    d.qpos[:3] = [0.0, 0.0, 2.0]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(sim.sim.mj_model, d)
    sim.brain.v[:] = 0.0
    sim.brain.refrac[:] = 0.0
    sim.brain.rate[:] = 0.0
    sim.gait.phase = 0.0
    sim._brain_ms = 0.0
    sim.drive, sim.turn = 0.5, 0.0

    # mj_resetData zeroes d.ctrl, which with these gains drives every joint to
    # 0 rad and folds the fly into a heap - it then never walks and every trial
    # returns the same numbers. Hold the gait's standing pose while it settles.
    ctrl_dt = TIMESTEP * CONTROL_EVERY
    d.ctrl[:] = sim.gait.neutral
    for _ in range(int(0.8 / ctrl_dt)):
        sim.gait.step(ctrl_dt, 0.0, 0.0)
        sim.gait.apply()
        mujoco.mj_step(sim.sim.mj_model, d, nstep=CONTROL_EVERY)

    pads = np.array(sim.dish.pad_xy)
    closest = np.full(len(pads), 1e9)
    on_pad = 0.0
    path = 0.0
    prev = np.array(d.qpos[:2], dtype=float).copy()
    for _ in range(int(secs / ctrl_dt)):
        sim.gait.step(ctrl_dt, sim.drive, sim.turn)
        sim.gait.apply()
        mujoco.mj_step(sim.sim.mj_model, d, nstep=CONTROL_EVERY)
        speed = float(np.linalg.norm(d.qvel[:2]))
        near_wall = float(np.hypot(d.qpos[0], d.qpos[1])) > 47.0
        sim._step_brain(ctrl_dt * 1000.0, speed, near_wall)
        sim.drive, sim.turn = sim._read_motor()
        p = np.array(d.qpos[:2], dtype=float)
        path += float(np.linalg.norm(p - prev)); prev = p.copy()
        dist = np.linalg.norm(pads - p, axis=1)
        closest = np.minimum(closest, dist)
        if dist.min() <= PAD_R:
            on_pad += ctrl_dt
    return float(closest.min()), on_pad, path


print(f"building the fly and the brain ...", flush=True)
sim = NMFSim(seed=0)
print(f"\n{TRIALS} paired trials of {SECS:.0f} s, odour ON vs OFF\n", flush=True)
print(f"{'seed':>6}{'arm':>6}{'closest mm':>12}{'on pad s':>10}{'path mm':>9}")
res = {True: [], False: []}
t0 = time.time()
for k in range(TRIALS):
    seed = 400 + 17 * k
    for smell in (True, False):
        c, on, path = trial(sim, smell, seed, SECS)
        res[smell].append((c, on, path))
        print(f"{seed:>6}{'ON' if smell else 'OFF':>6}{c:12.2f}{on:10.2f}{path:9.1f}",
              flush=True)

print(f"\n(ran in {(time.time() - t0) / 60:.1f} min)")
print(f"\n{'arm':>6}{'closest mm':>14}{'on pad s':>12}{'path mm':>10}")
for smell in (True, False):
    a = np.array(res[smell])
    print(f"{'ON' if smell else 'OFF':>6}{a[:, 0].mean():9.2f} +/-{a[:, 0].std():4.2f}"
          f"{a[:, 1].mean():8.2f} +/-{a[:, 1].std():4.2f}"
          f"{a[:, 2].mean():7.1f} +/-{a[:, 2].std():4.1f}")
on = np.array(res[True])
off = np.array(res[False])
closer = off[:, 0].mean() - on[:, 0].mean()
longer = on[:, 1].mean() - off[:, 1].mean()
print(f"\n  odour gets the fly {closer:+.2f} mm closer and {longer:+.2f} s more "
      f"time on a pad")
print("  verdict:", "CHEMOTAXIS" if (closer > 1.0 and longer > 0.0) else
      "no measurable benefit from the odour")
