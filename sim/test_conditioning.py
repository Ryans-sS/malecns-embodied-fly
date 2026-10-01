"""The experiment that decides everything: can this fly be conditioned?

Pair one pad's odour with dopaminergic reward, leave another unpaired, then ask
whether the mushroom-body output response to the trained odour has dropped MORE
than the response to the untrained one. That differential is what a learned
preference is made of. If it is absent, no amount of arena time will produce a
fly that presses the sugar button when hungry.

This sidesteps the argument about overlap metrics entirely: it measures the thing
the overlap was only ever a proxy for.

  python sim/test_conditioning.py [cue_drive ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from sim.brain import Brain
from sim import world as W
from sim.world import World

# (cue drive, KC threshold multiple, learning rate, training ms)
#
# The first run depressed EVERY KC->MBON synapse to the floor: with Kenyon cells
# 23-35% dense and 9 s of unbroken dopamine, every cell fires at some point, gets
# tagged and saturates. Selectivity needs a small cue-specific active set and a
# graded, not saturating, amount of depression.
GRID = [
    (1.5, 1.0, 0.020, 9000),    # the configuration that failed, for reference
    (1.5, 2.2, 0.004, 3000),
    (2.5, 2.2, 0.004, 3000),
    (2.5, 2.8, 0.002, 2000),
]
TEST_SETTLE, TEST_MEASURE = 600, 900
DAN_MV = 1.2

brain = Brain(os.path.join("sim", "brain_malecns.npz"))
mbon = brain.pop("mbon")
kc = brain.pop("kenyon")


def expose(world, pad, ms_settle, ms_measure, reward=False, seed=81):
    """Present one pad's odour; optionally pair it with dopaminergic reward."""
    brain.rng = np.random.default_rng(seed)
    brain.v[:] = 0.0
    brain.refrac[:] = 0.0
    brain.rate[:] = 0.0
    brain.alive[:] = True
    acc = np.zeros(len(mbon))
    n = 0
    for t in range(ms_settle + ms_measure):
        brain.clear_input()
        brain.inject("visual", W.LIGHT_DRIVE)
        brain.inject("mechano_proprio", 0.4)
        if pad is not None:
            brain.ext[pad.cue_L] += np.float32(W.CUE_DRIVE)
            brain.ext[pad.cue_R] += np.float32(W.CUE_DRIVE)
        if reward:
            brain.inject("dan", DAN_MV)
        spk = brain.step(excitability=1.0)
        if reward:
            brain.update_plasticity(spk, 1.0)
        else:
            # keep the tag decaying, but do not teach
            brain.update_plasticity(spk, 0.0)
        if t >= ms_settle:
            acc += brain.rate[mbon]
            n += 1
    return acc / n * 1000.0


import sim.brain as BR
from sim.brain import V_THRESH
for drive, kc_mult, lrate, TRAIN_MS in GRID:
    W.CUE_DRIVE = drive
    BR.LEARN_RATE = lrate
    brain.v_thresh[kc] = np.float32(V_THRESH * kc_mult)
    brain.w_mult[:] = 1.0
    brain.kc_tag[:] = 0.0
    brain._mb_dirty = False
    brain.edge_w[brain.kc_to_mbon] = brain._base_w_kc
    world = World(brain, rng=np.random.default_rng(2))
    trained, control = world.pads[0], world.pads[2]      # SUGAR trained, SALT not

    pre_t = expose(world, trained, TEST_SETTLE, TEST_MEASURE)
    pre_c = expose(world, control, TEST_SETTLE, TEST_MEASURE)

    # training: the trained odour, paired with reward
    expose(world, trained, 0, TRAIN_MS, reward=True, seed=83)
    learned = brain.learned_depression

    post_t = expose(world, trained, TEST_SETTLE, TEST_MEASURE, seed=85)
    post_c = expose(world, control, TEST_SETTLE, TEST_MEASURE, seed=85)

    d_t = (post_t.mean() - pre_t.mean()) / max(pre_t.mean(), 1e-9) * 100
    d_c = (post_c.mean() - pre_c.mean()) / max(pre_c.mean(), 1e-9) * 100

    print(f"\ncue drive {drive:.2f} mV   (MB depression after training: {learned:.4f})")
    print(f"  {'odour':10}{'MBON pre':>11}{'MBON post':>11}{'change':>10}")
    print(f"  {'TRAINED':10}{pre_t.mean():11.2f}{post_t.mean():11.2f}{d_t:+9.1f}%")
    print(f"  {'control':10}{pre_c.mean():11.2f}{post_c.mean():11.2f}{d_c:+9.1f}%")
    sel = d_t - d_c
    print(f"  selectivity (trained minus control): {sel:+.1f} percentage points")
    print("  verdict:", "LEARNS SELECTIVELY - a preference is achievable"
          if sel < -2.0 else
          ("learned, but not selectively - cannot tell the odours apart"
           if learned > 0.01 else "no learning at all"))
