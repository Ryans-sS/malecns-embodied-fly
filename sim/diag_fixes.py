"""Check two fixes directly: the turn bias, and whether the fly can learn at all.

A. With no odour anywhere, the turn command should hover around zero. It was
   settling near +0.55, which makes the fly circle.
B. Standing on a dispensing sugar pad should drive the reward dopaminergic
   neurons and depress KC->MBON synapses. Mushroom-body depression had been
   stuck at 0.0000 in every run because the teaching signal was gated behind
   actually swallowing.

  python sim/diag_fixes.py
"""
import math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from sim.brain import Brain, DT_MS
from sim.body import Body
from sim import world as W
from sim.world import World

brain = Brain(os.path.join("sim", "brain_malecns.npz"))
brain.measure_baseline(["mn_proboscis", "mn_leg_front_L", "mn_leg_front_R",
                        "mn_leg_mid_L", "mn_leg_mid_R", "mn_leg_hind_L",
                        "mn_leg_hind_R"])
brain.measure_steer_reference(W.CUE_DRIVE)

# ---------------------------------------------------------------- A
body = Body(brain.N, rng=np.random.default_rng(5))
world = World(brain, rng=np.random.default_rng(2))
world.x, world.y, world.theta = 0.0, 0.0, 0.0   # centre, far from every pad
brain.v[:] = 0.0; brain.refrac[:] = 0.0; brain.rate[:] = 0.0
brain.rng = np.random.default_rng(31)
samples = []
for t in range(14000):
    world.apply_senses(body)
    brain.step(excitability=1.0)
    # hold the fly still: we want the command, not the trajectory
    world.x, world.y = 0.0, 0.0
    if t >= 6000:
        samples.append(brain.steer_signal())
a = np.array(samples)
print("A. turn command with no odour nearby")
print(f"   mean {a.mean():+.3f}   sd {a.std():.3f}   "
      f"|mean| {abs(a.mean()):.3f}  (was +0.55)")
print(f"   verdict: {'OK - centred' if abs(a.mean()) < 0.2 else 'STILL BIASED'}\n")

# ---------------------------------------------------------------- B
body = Body(brain.N, rng=np.random.default_rng(6))
body.energy = 0.20          # hungry, so the reward is worth something
body.water = 0.40
world = World(brain, rng=np.random.default_rng(2))
pad = world.pads[0]         # SUGAR
world.x, world.y, world.theta = pad.x, pad.y, 0.0
pad.deliver_ms = 60000.0    # hold it open for the whole test
brain.v[:] = 0.0; brain.refrac[:] = 0.0; brain.rate[:] = 0.0
brain.rng = np.random.default_rng(33)
dop = 0.0
dan0 = None
start_learn = brain.learned_depression
for t in range(16000):
    world.apply_senses(body)
    if dop > 0:
        brain.inject("dan", 1.2 * dop)
    spk = brain.step(excitability=body.excitability)
    brain.update_plasticity(spk, dop)
    dop *= 0.92
    world.x, world.y = pad.x, pad.y      # keep it on the pad
    if t % 5 == 0:
        m = world.step(DT_MS * 5, body)
        dop = max(dop, min(1.0, m["reward"] + m["punish"]))
        world.x, world.y = pad.x, pad.y
    if t == 3000:
        dan0 = brain.pop_rate("dan")

print("B. standing on a dispensing SUGAR pad while hungry")
print(f"   taste-peg GRNs      {brain.pop_rate('grn_tastepeg'):7.2f} Hz")
print(f"   proboscis MNs       {brain.delta_rate('mn_proboscis'):+7.2f} Hz above rest")
print(f"   DAN rate            {brain.pop_rate('dan'):7.2f} Hz  (at 3 s: {dan0:.2f})")
print(f"   reward signal       {dop:7.3f}")
print(f"   sucrose ingested    {body.ingested['sucrose']:7.3f}")
print(f"   MB depression       {start_learn:.4f} -> {brain.learned_depression:.4f}")
print(f"   verdict: {'OK - the mushroom body learned' if brain.learned_depression > 0.001 else 'STILL NOT LEARNING'}")
