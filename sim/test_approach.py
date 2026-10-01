"""Controlled approach assay: does olfactory steering get the fly to a pad?

Releases a freely moving fly at a fixed distance from one pad and measures how
close it gets and how long it spends on the pad. Runs the identical protocol with
the odour cue switched OFF as a control, so the comparison isolates chemotaxis
from the exploratory random walk.

  python sim/test_approach.py [n_trials] [seconds_per_trial]
"""
import math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from sim.brain import Brain, DT_MS
from sim.body import Body
from sim import world as W
from sim.world import World

# --only on / --only off lets a half-finished experiment be completed without
# redoing the half that already ran.
ONLY = None
if "--only" in sys.argv:
    ONLY = sys.argv[sys.argv.index("--only") + 1]
    sys.argv = [a for a in sys.argv if a not in ("--only", ONLY)]

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
TRIAL_S = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
RELEASE_DIST = 11.0     # mm from the pad centre, just inside the cue radius
PHYS_EVERY = 5

brain = Brain(os.path.join("sim", "brain_malecns.npz"))
brain.measure_baseline([
    "mn_proboscis", "mn_leg_front_L", "mn_leg_front_R", "mn_leg_mid_L",
    "mn_leg_mid_R", "mn_leg_hind_L", "mn_leg_hind_R",
])
brain.measure_steer_reference(W.CUE_DRIVE)
print(f"released {RELEASE_DIST:.0f} mm from the SUGAR pad, "
      f"{N_TRIALS} trials x {TRIAL_S:.0f} s per condition\n")


def trial(seed, cues_on):
    body = Body(brain.N, rng=np.random.default_rng(seed + 500))
    world = World(brain, rng=np.random.default_rng(seed + 900))
    pad = world.pads[0]
    # release at a random bearing around the pad, heading tangentially so the fly
    # is not simply pointed at the target
    ang = np.random.default_rng(seed).uniform(0, 2 * math.pi)
    world.x = pad.x + RELEASE_DIST * math.cos(ang)
    world.y = pad.y + RELEASE_DIST * math.sin(ang)
    world.theta = ang + math.pi / 2
    brain.v[:] = 0.0
    brain.refrac[:] = 0.0
    brain.rate[:] = 0.0
    brain.alive[:] = True
    brain.steer_ema = brain.steer_ref[0] - brain.steer_ref[1]
    brain.rng = np.random.default_rng(seed + 7)

    if not cues_on:
        saved = [(p.cue_L, p.cue_R) for p in world.pads]
        empty = np.array([], dtype=np.int32)
        for p in world.pads:
            p.cue_L, p.cue_R = empty, empty

    closest = RELEASE_DIST
    on_pad_ms = 0.0
    steer_mag = 0.0
    steps = int(TRIAL_S * 1000 / DT_MS)
    motor = {}
    for i in range(steps):
        world.apply_senses(body)
        brain.step(excitability=body.excitability)
        steer_mag += abs(brain.steer_signal())
        if i % PHYS_EVERY == 0:
            motor = world.step(DT_MS * PHYS_EVERY, body)
            d = math.hypot(world.x - pad.x, world.y - pad.y)
            closest = min(closest, d)
            if d <= W.PAD_R:
                on_pad_ms += DT_MS * PHYS_EVERY

    if not cues_on:
        for p, (l, r) in zip(world.pads, saved):
            p.cue_L, p.cue_R = l, r
    return closest, on_pad_ms / 1000.0, steer_mag / steps, pad.presses


rows = {}
CONDS = [("odour ON", True), ("odour OFF (control)", False)]
if ONLY == "on":
    CONDS = CONDS[:1]
elif ONLY == "off":
    CONDS = CONDS[1:]

for cond, cues_on in CONDS:
    cl, on, sm, pr = [], [], [], []
    for k in range(N_TRIALS):
        c, o, s, p = trial(1000 + k, cues_on)
        cl.append(c); on.append(o); sm.append(s); pr.append(p)
        print(f"  {cond:20} trial {k+1}: closest {c:5.1f} mm  "
              f"on-pad {o:5.1f} s  presses {p}", flush=True)
    rows[cond] = (np.mean(cl), np.mean(on), np.mean(sm), np.sum(pr))
    print()

print("=" * 72)
print(f"{'condition':22}{'closest (mm)':>14}{'on-pad (s)':>13}{'|steer|':>10}{'presses':>10}")
for cond, (c, o, s, p) in rows.items():
    print(f"{cond:22}{c:14.2f}{o:13.2f}{s:10.3f}{int(p):10d}")

if len(rows) < 2:
    print("")
    print("(single condition only - no verdict without both)")
    sys.exit(0)
c_on, o_on = rows["odour ON"][0], rows["odour ON"][1]
c_off, o_off = rows["odour OFF (control)"][0], rows["odour OFF (control)"][1]
print()
print(f"closest approach improved by {c_off - c_on:+.2f} mm with odour")
print(f"time on pad changed by       {o_on - o_off:+.2f} s with odour")
verdict = ("chemotaxis helps" if (c_off - c_on) > 0.5 or (o_on - o_off) > 0.5
           else "no measurable benefit")
print(f"verdict: {verdict}")
