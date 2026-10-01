"""Where does the time in one simulated millisecond actually go?"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
from sim.brain import Brain, V_THRESH, T_REFRAC_MS, DT_MS, NOISE_MV, TONIC_MV

b = Brain(os.path.join("sim","brain_malecns.npz"))
N = b.N
# settle into a realistic firing regime first
for _ in range(1200):
    b.clear_input(); b.step()
spk = np.flatnonzero((b.v >= V_THRESH*0.0) & b.alive)[:0]
print(f"N={N:,}  resting rate {b.rate.mean()*1000:.2f} Hz\n")

def bench(label, fn, K=120):
    for _ in range(10): fn()
    t0=time.perf_counter()
    for _ in range(K): fn()
    dt=(time.perf_counter()-t0)/K*1000
    print(f"  {label:46}{dt:8.3f} ms")
    return dt

tot = 0.0
tot += bench("rng.normal(0, s, N)  [current noise]", lambda: b.rng.normal(0.0, NOISE_MV, N).astype(np.float32))
tot += bench("rng.standard_normal(N, dtype=float32)", lambda: b.rng.standard_normal(N, dtype=np.float32))
tot += bench("v *= decay", lambda: b.v.__imul__(b.decay))
tot += bench("threshold + flatnonzero", lambda: np.flatnonzero((b.v>=V_THRESH)&(b.refrac<=0)&b.alive))
tot += bench("rate *= decay", lambda: b.rate.__imul__(b.rate_decay))
tot += bench("clear_input (ext[:]=0)", lambda: b.clear_input())

# the event-driven gather, at a realistic spike count
rate = b.rate.mean()*1000
nspk = max(1,int(N*rate/1000.0))
spiking = np.random.default_rng(0).choice(N, size=nspk, replace=False).astype(np.int64)
def gather():
    s0=b.out_start[spiking]; ln=b.out_end[spiking]-s0
    tot_=int(ln.sum())
    starts=np.repeat(s0,ln)
    base=np.concatenate(([0],np.cumsum(ln)[:-1]))
    within=np.arange(tot_,dtype=np.int64)-np.repeat(base,ln)
    g=starts+within
    return np.bincount(b.edge_post[g],weights=b.edge_w[g],minlength=N)
nedge=int((b.out_end[spiking]-b.out_start[spiking]).sum())
print(f"\n  (spike gather with {len(spiking):,} spikers / {nedge:,} edges)")
bench("event gather + bincount", gather, K=60)

print("\n  full step() for reference:")
bench("brain.step()", lambda: b.step(), K=60)
