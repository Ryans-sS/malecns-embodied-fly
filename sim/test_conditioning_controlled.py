"""Conditioning, with a sham control and repeats.

A single conditioning run reported ~4 percentage points of selectivity, which
looked like learning. But the pre-training baselines for the two odours already
differed by a similar amount, and one configuration produced +30 points - the
wrong sign entirely. So the effect has to be compared against its own null:
exactly the same protocol with the reward withheld.

If rewarded runs are not reliably more selective than sham runs, there is no
learning, however good the single-run number looked.

  python sim/test_conditioning_controlled.py [n_seeds]
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

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
CUE_DRIVE = 2.5
KC_THRESH_MULT = 2.2
LEARN_RATE = 0.004
TRAIN_MS = 3000
TEST_SETTLE, TEST_MEASURE = 600, 900
DAN_MV = 1.2

brain = Brain(os.path.join("sim", "brain_malecns.npz"))
mbon = brain.pop("mbon")
kc = brain.pop("kenyon")
W.CUE_DRIVE = CUE_DRIVE
BR.LEARN_RATE = LEARN_RATE
brain.v_thresh[kc] = np.float32(V_THRESH * KC_THRESH_MULT)
world = World(brain, rng=np.random.default_rng(2))
trained, control = world.pads[0], world.pads[2]


def expose(pad, settle, measure, reward=False, seed=81):
    brain.rng = np.random.default_rng(seed)
    brain.v[:] = 0.0
    brain.refrac[:] = 0.0
    brain.rate[:] = 0.0
    brain.alive[:] = True
    acc = np.zeros(len(mbon))
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
        brain.update_plasticity(spk, 1.0 if reward else 0.0)
        if t >= settle:
            acc += brain.rate[mbon]
            n += 1
    return acc / n * 1000.0


def one_run(seed, rewarded):
    brain.w_mult[:] = 1.0
    brain.kc_tag[:] = 0.0
    brain._mb_dirty = False
    brain.edge_w[brain.kc_to_mbon] = brain._base_w_kc
    # Paired design: each odour's pre and post test use the SAME noise seed, so
    # the only difference between them is the training. Using different seeds
    # made the pre/post comparison unpaired and buried a few points of effect
    # under tens of points of run-to-run noise.
    pre_t = expose(trained, TEST_SETTLE, TEST_MEASURE, seed=seed).mean()
    pre_c = expose(control, TEST_SETTLE, TEST_MEASURE, seed=seed + 1).mean()
    expose(trained, 0, TRAIN_MS, reward=rewarded, seed=seed + 2)
    post_t = expose(trained, TEST_SETTLE, TEST_MEASURE, seed=seed).mean()
    post_c = expose(control, TEST_SETTLE, TEST_MEASURE, seed=seed + 1).mean()
    d_t = (post_t - pre_t) / max(pre_t, 1e-9) * 100
    d_c = (post_c - pre_c) / max(pre_c, 1e-9) * 100
    return d_t - d_c, brain.learned_depression


print(f"cue {CUE_DRIVE} mV | KC thresh x{KC_THRESH_MULT} | learn {LEARN_RATE} "
      f"| train {TRAIN_MS} ms | {N_SEEDS} seeds each\n")
print(f"{'seed':>6}{'REWARDED selectivity':>24}{'SHAM selectivity':>20}")
rew, sham = [], []
for k in range(N_SEEDS):
    seed = 200 + k * 11
    r, learned = one_run(seed, True)
    s, _ = one_run(seed, False)
    rew.append(r)
    sham.append(s)
    print(f"{seed:>6}{r:>20.1f} pp{s:>16.1f} pp", flush=True)

rew, sham = np.array(rew), np.array(sham)
print(f"\n  rewarded : mean {rew.mean():+6.2f} pp   sd {rew.std():5.2f}")
print(f"  sham     : mean {sham.mean():+6.2f} pp   sd {sham.std():5.2f}")
diff = rew.mean() - sham.mean()
pooled = np.sqrt((rew.var() + sham.var()) / 2) + 1e-9
print(f"  difference: {diff:+.2f} pp   effect size d = {diff / pooled:+.2f}")
print("\nverdict:", "REAL selective learning - reward beats sham"
      if diff < -1.0 and abs(diff / pooled) > 1.0 else
      "no selective learning distinguishable from sham")
