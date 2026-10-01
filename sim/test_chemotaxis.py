"""Does the fly steer toward an odour, using its real steering neurons?

Holds the fly still at a fixed distance from one pad, places that pad at a known
bearing, and reads the turn command out of DNa01/DNa02. A positive command means
"turn left". If chemotaxis works, a pad on the left should produce a positive
command and a pad on the right a negative one.

  python sim/test_chemotaxis.py
"""
import math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from sim.brain import Brain
from sim.world import World, CUE_DRIVE, CUE_R
from sim.body import Body

SETTLE_MS = 700
MEASURE_MS = 900

brain = Brain(os.path.join("sim", "brain_malecns.npz"))
print("calibrating dead-ahead reference ...")
refL, refR = brain.measure_steer_reference(CUE_DRIVE)
print(f"  symmetric odour: DN_L {refL:.2f} Hz, DN_R {refR:.2f} Hz\n")

world = World(brain, rng=np.random.default_rng(2))
body = Body(brain.N, rng=np.random.default_rng(5))
pad = world.pads[0]           # SUGAR

print(f"{'pad bearing':>14} {'ORN_L':>8} {'ORN_R':>8} {'DN_L':>8} {'DN_R':>8} "
      f"{'turn cmd':>10}  verdict")

results = {}
for label, bearing_deg in (
    ("hard left", 90), ("left", 40), ("ahead", 0),
    ("right", -40), ("hard right", -90),
):
    bearing = math.radians(bearing_deg)
    # place the fly 5 mm from the pad, heading such that the pad sits at `bearing`
    dist = 5.0
    world.theta = 0.0
    world.x = pad.x - dist * math.cos(bearing + world.theta)
    world.y = pad.y - dist * math.sin(bearing + world.theta)
    brain.v[:] = 0.0
    brain.refrac[:] = 0.0
    brain.rate[:] = 0.0
    brain.rng = np.random.default_rng(13)

    accs = {k: 0.0 for k in ("olfactory_L", "olfactory_R", "dn_steer_L", "dn_steer_R")}
    acc_cmd = 0.0
    n = 0
    for t in range(SETTLE_MS + MEASURE_MS):
        world.apply_senses(body)
        brain.step(excitability=1.0)
        if t >= SETTLE_MS:
            for k in accs:
                accs[k] += brain.pop_rate(k)
            acc_cmd += brain.steer_signal()
            n += 1
    cmd = acc_cmd / n
    results[bearing_deg] = cmd
    want = "left" if bearing_deg > 0 else ("right" if bearing_deg < 0 else "-")
    got = "left" if cmd > 0.05 else ("right" if cmd < -0.05 else "-")
    mark = "ok" if (want == got or want == "-") else "WRONG WAY"
    print(f"{label:>14} {accs['olfactory_L']/n:8.2f} {accs['olfactory_R']/n:8.2f} "
          f"{accs['dn_steer_L']/n:8.2f} {accs['dn_steer_R']/n:8.2f} {cmd:+10.3f}  {mark}")

left = np.mean([results[90], results[40]])
right = np.mean([results[-90], results[-40]])
print(f"\nmean turn command with the pad on the LEFT : {left:+.3f}")
print(f"mean turn command with the pad on the RIGHT: {right:+.3f}")
print(f"separation: {left - right:+.3f}  "
      f"({'CHEMOTAXIS WORKS - turns toward odour' if left - right > 0.1 else ''}"
      f"{'turns AWAY from odour' if left - right < -0.1 else ''}"
      f"{'no usable signal' if abs(left - right) <= 0.1 else ''})")
