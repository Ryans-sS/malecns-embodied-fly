"""Find the background drive that puts the network in a biological firing regime.

A connectome on its own is silent: with a 20 ms membrane leak, sub-threshold input
never accumulates. Real brains have spontaneous activity, so the model needs a
tonic drive plus membrane noise. Rather than guessing those two numbers, sweep
them and measure the resulting population firing rate.

Target: a few Hz mean rate, which is what fly neurons actually do at rest.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
from sim.brain import Brain, DT_MS, V_THRESH

b = Brain(os.path.join("sim", "brain_malecns.npz"))
N = b.N
print(f"N={N:,}  threshold={V_THRESH} mV")
print(f"mean |w| per connection = {np.abs(b.edge_w).mean():.3f} mV")
print(f"mean out-degree = {len(b.edge_post)/N:.0f}")
print(f"E/I edge ratio = {(b.edge_w>0).sum()/max(1,(b.edge_w<0).sum()):.2f}\n")
print(f"{'tonic':>7} {'noise':>7} | {'rate@200ms':>11} {'rate@600ms':>11} {'rate@1000ms':>12}  verdict")
for tonic in (0.05, 0.10, 0.15, 0.20, 0.30):
    for noise in (0.3, 0.6):
        b.v[:] = 0.0; b.refrac[:] = 0.0; b.rate[:] = 0.0
        b.alive[:] = True
        marks = {}
        for t in range(1000):
            b.clear_input()
            b.ext[:] = np.float32(tonic)
            spk = b.step(excitability=1.0, noise_mv=noise)
            if t in (199, 599, 999):
                marks[t] = b.rate.mean()*1000.0
        v = [marks[199], marks[599], marks[999]]
        verdict = "silent" if v[2] < 0.2 else ("RUNAWAY" if v[2] > 60 else "ok" if 0.5 <= v[2] <= 25 else "high")
        print(f"{tonic:7.2f} {noise:7.2f} | {v[0]:11.2f} {v[1]:11.2f} {v[2]:12.2f}  {verdict}")
