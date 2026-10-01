"""How big is the odour steering signal IN THE ARENA, against its own noise?

probe_steering.py drove a whole antennal lobe (931-1,699 ORNs) at 0.88 mV and
found huge lateralised responses. But a pad in the arena only drives its own
sparse cue subset - about 80-140 ORNs per side at <=0.55 mV. That is an order of
magnitude less drive, which would explain why the approach assay shows no benefit
from odour even though the probe looked emphatic.

This measures the thing that actually matters: the separation between the steering
imbalance with a pad on the LEFT versus on the RIGHT, divided by how much that
imbalance fluctuates on its own. That ratio (d-prime) is what decides whether the
fly can steer. Below ~1 it cannot, no matter how the gain is set.

  python sim/diag_steer_snr.py
"""
import math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from sim.brain import Brain
from sim.body import Body
from sim import world as W
from sim.world import World

SETTLE_MS = 700
MEASURE_MS = 1800

brain = Brain(os.path.join("sim", "brain_malecns.npz"))
body = Body(brain.N, rng=np.random.default_rng(5))
world = World(brain, rng=np.random.default_rng(2))
pad = world.pads[0]


def raw_imbalance():
    if brain.steer_idxL is not None:
        L = float(brain.rate[brain.steer_idxL] @ brain.steer_wL) * 1000.0
        R = float(brain.rate[brain.steer_idxR] @ brain.steer_wR) * 1000.0
    else:
        L, R = brain.pop_rate("dn_steer_L"), brain.pop_rate("dn_steer_R")
    return L - R


def run(bearing_deg, cue_on=True, dist=5.0, seed=21):
    if not cue_on:
        saved = [(p.cue_L, p.cue_R) for p in world.pads]
        empty = np.array([], dtype=np.int32)
        for p in world.pads:
            p.cue_L, p.cue_R = empty, empty
    bearing = math.radians(bearing_deg)
    world.theta = 0.0
    world.x = pad.x - dist * math.cos(bearing)
    world.y = pad.y - dist * math.sin(bearing)
    brain.v[:] = 0.0
    brain.refrac[:] = 0.0
    brain.rate[:] = 0.0
    brain.alive[:] = True
    brain.rng = np.random.default_rng(seed)
    samples, orn = [], []
    for t in range(SETTLE_MS + MEASURE_MS):
        world.apply_senses(body)
        brain.step(excitability=1.0)
        if t >= SETTLE_MS:
            samples.append(raw_imbalance())
            orn.append((brain.pop_rate("olfactory_L"), brain.pop_rate("olfactory_R")))
    if not cue_on:
        for p, (l, r) in zip(world.pads, saved):
            p.cue_L, p.cue_R = l, r
    a = np.array(samples)
    o = np.array(orn)
    return a, o.mean(axis=0)


print(f"cue subset per pad: {len(pad.cue_L)} left ORNs + {len(pad.cue_R)} right "
      f"(of {len(brain.pop('olfactory_L'))} / {len(brain.pop('olfactory_R'))} available)")
print(f"CUE_DRIVE {W.CUE_DRIVE} mV, sharpness {W.CUE_ANTENNA_SHARPNESS}\n")

left, orn_l = run(+90)
right, orn_r = run(-90)
none_, orn_n = run(0, cue_on=False)

print(f"{'condition':18}{'ORN L':>9}{'ORN R':>9}{'imbalance mean':>17}{'sd':>9}")
for name, a, o in (("pad on LEFT", left, orn_l), ("pad on RIGHT", right, orn_r),
                   ("no odour", none_, orn_n)):
    print(f"{name:18}{o[0]:9.2f}{o[1]:9.2f}{a.mean():17.2f}{a.std():9.2f}")

pooled = math.sqrt((left.var() + right.var()) / 2) + 1e-9
dprime = abs(left.mean() - right.mean()) / pooled
print(f"\nseparation  |left - right| = {abs(left.mean()-right.mean()):.2f} Hz")
print(f"noise       pooled sd       = {pooled:.2f} Hz")
print(f"d-prime                     = {dprime:.3f}")
print("verdict:", "usable" if dprime > 1.0 else
      ("marginal" if dprime > 0.4 else "TOO WEAK TO STEER ON"))
