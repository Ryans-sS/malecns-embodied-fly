"""Smelling the request pads, from inside the NeuroMechFly dish.

This is the sensory half of closing the loop: it turns where the fly is standing
into drive on real olfactory receptor neurons, so the brain's own steering output
has something to steer towards.

The partition logic is the one that made chemotaxis work in the three.js arena and
it matters more than it looks. An odour is a combination of GLOMERULI, not a
random handful of receptor neurons. MaleCNS types every ORN by its glomerulus
(ORN_DA1, ORN_VA1d, ... 53 types of ~50 cells), and splitting the cells randomly
cuts every glomerulus across several pads, so each "odour" partially activates
almost all of them and all five pads become the same smell - measured Kenyon-cell
pattern overlap 0.545, against 0.180 once partitioned by glomerulus.

Laterality is real too: each ORN innervates AL(L) or AL(R), and which antenna
hears a pad is set by its bearing in the fly's own frame. A fly with no left/right
difference has nothing to steer on.
"""
from __future__ import annotations

import math

import numpy as np

# How far a pad's odour carries, millimetres. Scaled from the three.js arena's
# 9 mm by the ratio of the two arenas (50 mm dish against 30 mm), so a pad
# occupies about the same fraction of the world as the one chemotaxis was
# verified in.
CUE_R = 15.0
CUE_DRIVE = 2.50               # measured operating point; at 0.75 d-prime is 0.22
CUE_ANTENNA_SHARPNESS = 0.85   # how strongly bearing biases one antenna


class PadOdour:
    """Gives each pad a distinct glomerular identity and smells them bilaterally."""

    def __init__(self, brain, pad_xy, pad_names, rng=None):
        self.brain = brain
        self.rng = rng or np.random.default_rng(2)
        self.pad_xy = list(pad_xy)
        self.names = list(pad_names)

        olf = brain.pop("olfactory")
        glom = {}
        for i in olf:
            glom.setdefault(str(brain.primary_type[i]), []).append(i)
        gnames = sorted(glom)
        self.rng.shuffle(gnames)

        left = brain.pop("olfactory_L")
        right = brain.pop("olfactory_R")
        per = max(1, len(gnames) // max(1, len(self.names)))
        self.cue_L, self.cue_R, self.glomeruli = {}, {}, {}
        for k, name in enumerate(self.names):
            mine = gnames[k * per:(k + 1) * per]
            if not mine:                       # more pads than glomeruli
                self.cue_L[name] = np.empty(0, np.int32)
                self.cue_R[name] = np.empty(0, np.int32)
                self.glomeruli[name] = []
                continue
            cue = np.sort(np.concatenate([glom[g] for g in mine])).astype(np.int32)
            self.glomeruli[name] = mine
            self.cue_L[name] = np.intersect1d(cue, left)
            self.cue_R[name] = np.intersect1d(cue, right)
        self.near = []

    def sense(self, x, y, theta):
        """Inject every pad in range, split between the two antennal lobes.

        Call between brain.clear_input() and brain.step().
        """
        b = self.brain
        self.near = []
        for name, (px, py) in zip(self.names, self.pad_xy):
            d = math.hypot(x - px, y - py)
            if d >= CUE_R:
                continue
            strength = CUE_DRIVE * (1.0 - d / CUE_R)
            # bearing of the pad in the fly's own frame: +1 hard left, -1 hard right
            lat = math.sin(math.atan2(py - y, px - x) - theta)
            gL = 1.0 + CUE_ANTENNA_SHARPNESS * lat
            gR = 1.0 - CUE_ANTENNA_SHARPNESS * lat
            b.ext[self.cue_L[name]] += np.float32(strength * gL)
            b.ext[self.cue_R[name]] += np.float32(strength * gR)
            self.near.append(name)
        return self.near
