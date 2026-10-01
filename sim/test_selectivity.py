"""Did training depress the TRAINED odour's synapses more than another odour's?

This measures the learned change where it actually lives - in the KC->MBON
weights - instead of inferring it from mushroom-body output. The output readout
turned out to be uninterpretable: MBONs are 51% cholinergic and 48% inhibitory
(glutamate + GABA), so averaging them mixes approach-driving and avoidance-driving
channels and the sign of any change is meaningless.

Protocol: find which Kenyon cells each odour drives, train on one odour with
reward, then compare the mean synaptic weight multiplier on synapses leaving the
trained odour's cells against those leaving another odour's cells. Cells driven by
both are excluded, so the comparison is between the parts that differ.

A sham arm with the reward withheld gives the null.

  python sim/test_selectivity.py [n_seeds]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import sim.brain as BR
from sim.brain import Brain, V_THRESH
from sim import world as W
from sim.world import World

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
TRAIN_MS = 3000
PROBE_SETTLE, PROBE_MEASURE = 600, 800
DAN_MV = 1.2

brain = Brain(os.path.join("sim", "brain_malecns.npz"))
kc = brain.pop("kenyon")
world = World(brain, rng=np.random.default_rng(2))
trained, control = world.pads[0], world.pads[2]

# which Kenyon cell is presynaptic to each plastic synapse
kc_pre = brain._kc_pre_of_syn


def drive(pad, settle, measure, reward=False, seed=81, learn=False):
    brain.rng = np.random.default_rng(seed)
    brain.v[:] = 0.0
    brain.refrac[:] = 0.0
    brain.rate[:] = 0.0
    brain.alive[:] = True
    acc = np.zeros(len(kc))
    n = 0
    for t in range(settle + measure):
        brain.clear_input()
        brain.inject("visual", W.LIGHT_DRIVE)
        brain.inject("mechano_proprio", 0.4)
        if pad is not None:
            brain.ext[pad.cue_L] += np.float32(W.CUE_DRIVE)
            brain.ext[pad.cue_R] += np.float32(W.CUE_DRIVE)
        if reward:
            brain.inject("dan", DAN_MV)
        spk = brain.step(excitability=1.0)
        if learn:
            brain.update_plasticity(spk, 1.0 if reward else 0.0)
        if t >= settle:
            acc += brain.rate[kc]
            n += 1
    return acc / n * 1000.0


def reset():
    brain.w_mult[:] = 1.0
    brain.kc_tag[:] = 0.0
    brain._mb_dirty = False
    brain.edge_w[brain.kc_to_mbon] = brain._base_w_kc


# which cells each odour drives (measured once, no plasticity active)
reset()
base = drive(None, PROBE_SETTLE, PROBE_MEASURE, seed=11)
ev_t = drive(trained, PROBE_SETTLE, PROBE_MEASURE, seed=11) - base
ev_c = drive(control, PROBE_SETTLE, PROBE_MEASURE, seed=11) - base
set_t = ev_t > 0.5
set_c = ev_c > 0.5
only_t = kc[set_t & ~set_c]
only_c = kc[set_c & ~set_t]
print(f"Kenyon cells driven by the trained odour only : {len(only_t):,}")
print(f"Kenyon cells driven by the control odour only : {len(only_c):,}")
print(f"driven by both (excluded)                     : {int((set_t & set_c).sum()):,}\n")

mask_t = np.isin(kc_pre, only_t)
mask_c = np.isin(kc_pre, only_c)
print(f"plastic synapses from trained-only cells: {int(mask_t.sum()):,}")
print(f"plastic synapses from control-only cells: {int(mask_c.sum()):,}\n")

print(f"{'seed':>6}{'arm':>10}{'trained w':>12}{'control w':>12}{'difference':>13}")
res = {"reward": [], "sham": []}
for k in range(N_SEEDS):
    seed = 300 + k * 13
    for arm, rewarded in (("reward", True), ("sham", False)):
        reset()
        drive(trained, 0, TRAIN_MS, reward=rewarded, seed=seed, learn=True)
        wt = float(brain.w_mult[mask_t].mean())
        wc = float(brain.w_mult[mask_c].mean())
        res[arm].append(wt - wc)
        print(f"{seed:>6}{arm:>10}{wt:12.4f}{wc:12.4f}{wt - wc:+13.4f}", flush=True)

r = np.array(res["reward"])
s = np.array(res["sham"])
print(f"\n  rewarded difference: {r.mean():+.4f} +/- {r.std():.4f}")
print(f"  sham difference    : {s.mean():+.4f} +/- {s.std():.4f}")
gap = r.mean() - s.mean()
print(f"  reward minus sham  : {gap:+.4f}")
print("\nverdict:", "SELECTIVE LEARNING - the trained odour's synapses were "
      "depressed more than the control odour's"
      if gap < -0.01 else
      ("learning is present but NOT selective between odours"
       if abs(float(brain.learned_depression)) > 0.01 else "no learning"))
