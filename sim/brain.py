"""Leaky integrate-and-fire engine over the MaleCNS connectome.

Cell dynamics and weight scaling follow the published whole-brain LIF model of
the FlyWire connectome (Shiu et al., Nature 2024): resting potential -52 mV,
threshold -45 mV, membrane time constant 20 ms, 2.2 ms refractory period, and
0.275 mV of postsynaptic potential per anatomical synapse. Potentials here are
kept relative to rest, so rest is 0 and threshold is 7 mV.

Propagation is event driven: only the outgoing synapses of neurons that actually
spiked on a step are touched, which is what makes 166,700 neurons and 25.6M
synapses tractable at 1 ms resolution on a CPU.

What is real: the wiring, the signs (from predicted transmitters), the identity
of every sensor / motor / mushroom-body population, and the plasticity site
(KC->MBON). What is model: the LIF parameters above, the sensory drive
amplitudes, and the per-spike energy cost.
"""
import math
import os

import numpy as np
from sim.kernel import propagate

# 1 ms integration step.
#
# dt=2 was tried and is NOT worth it: runtime is dominated by traversing the
# out-edges of cells that spiked, so a 2 ms step simply processes twice as many
# spikes per step and the total work per unit simulated time is unchanged. It
# measured slower in realtime terms (0.41x vs 0.48x) and drifted the resting rate
# from 1.82 to 2.52 Hz. The rate/sqrt-dt corrections below are kept because they
# are correct, and are inert at dt=1.
DT_MS = 1.0

V_THRESH = 7.0          # mV above rest
V_RESET = 0.0
TAU_M_MS = 20.0
T_REFRAC_MS = 2.2
MV_PER_SYNAPSE = 0.275

# Global synaptic gain and an extra factor on inhibition. These are TUNED, not
# measured: with raw 0.275 mV/synapse this graph (mean out-degree 153, E/I weight
# ratio 1.55) is bistable - silent, or it ignites and saturates near 30 Hz. The
# values below were found by sweeping until the resting population rate sat at
# ~1.8 Hz, which is what fly neurons actually do. See sim/calibrate.py.
W_SCALE = 0.08
I_GAIN = 1.4

# Cap on how much excitatory drive one cell may receive, in mV per global spike.
#
# A single global gain is calibrated for the median neuron (3.8 mV) and badly
# over-drives hubs: the input distribution runs to 95 mV at p99, 307 at p99.9 and
# 2,475 at the maximum. APL - the giant GABAergic feedback neuron that makes the
# mushroom-body code sparse - receives 247x the average, putting its steady state
# at 86 mV against a 7 mV threshold. It therefore sat pinned at ~328 Hz no matter
# what, delivering constant blanket inhibition instead of tracking Kenyon-cell
# activity, so the KC code never decorrelated and the fly could not tell one
# odour from another.
#
# Scaling a cell's incoming weights down when its total drive exceeds this cap is
# homeostatic synaptic scaling, which real neurons do. Excitation and inhibition
# are scaled together so each cell's E/I balance is preserved. Only ~0.1% of
# cells are affected.
HUB_INPUT_CAP = 300.0

# Kenyon-cell spike threshold, as a multiple of the shared threshold.
# 2.2 puts Kenyon-cell density at ~5%, which is what a real KC code is. It does
# not rescue selective learning (see sim/test_conditioning_controlled.py) but it
# is the right physiology and costs nothing measured.
KC_THRESH_MULT = 2.2

# Background state. A connectome alone is silent; real brains are spontaneously
# active. Tonic drive plus membrane noise supplies that (model parameters).
TONIC_MV = 0.12
NOISE_MV = 0.5

# Metabolic cost of one action potential, in arbitrary energy units. Spiking is
# the dominant energy cost of neural tissue; the absolute scale is set so that at
# the calibrated ~1.8 Hz resting rate the brain costs ~0.0012 energy/s, about a
# quarter of the body's resting budget (model, not measurement).
ENERGY_PER_SPIKE = 4.0e-9
ENERGY_BASELINE_PER_NEURON_PER_S = 5.0e-9

# Mushroom-body plasticity. Coincidence of Kenyon-cell activity with
# dopaminergic drive depresses that KC->MBON synapse, which is the established
# direction of the effect in Drosophila. Compartment-specific DAN/MBON pairing
# is simplified to a global dopaminergic gate.
# Steering adaptation. The olfactory-to-DNa01/DNa02 pathway carries a large
# constant left/right bias in this animal, so an absolute imbalance is useless as
# a turn command - it would make the fly circle. Subtracting a slow running mean
# leaves only CHANGES in the imbalance, which is how walking flies actually track
# odour: they compare over time as they move, rather than reading an instantaneous
# left-right difference.
# The turn command used to sit around +0.55 with no odour present, so the fly
# circled. The cause was not the time constant but the SEED: the running mean was
# initialised from the symmetric-odour reference, which is nowhere near the
# free-running value, so it spent the whole run chasing it. Slowing the constant
# made it worse (+0.83). It is now seeded from the first live sample instead, and
# 2.5 s sits between tracking the ~48 Hz constitutive offset and preserving the
# ~70 Hz swing an odour produces.
STEER_ADAPT_TAU_MS = 2500.0
# The adapted imbalance is squashed through tanh, so the turn command is bounded
# no matter how hard the ensemble fires. STEER_SCALE sets how many Hz of
# imbalance counts as a full-strength turn.
STEER_SCALE = 12.0

KC_TAG_TAU_MS = 500.0
LEARN_RATE = 0.020
FORGET_RATE = 2.0e-5
W_MULT_MIN = 0.15


class Brain:
    def __init__(self, npz_path, rng=None):
        z = np.load(npz_path, allow_pickle=True)
        self.ids = z["ids"]
        self.N = N = len(self.ids)
        self.out_start = z["out_start"]
        self.out_end = z["out_end"]
        self.edge_post = z["edge_post"]
        w = z["edge_w"].astype(np.float32) * MV_PER_SYNAPSE * W_SCALE
        w[w < 0] *= I_GAIN
        # homeostatic cap on per-cell input drive (see HUB_INPUT_CAP)
        _post = z["edge_post"]
        _exc = np.bincount(_post, weights=np.maximum(w, 0.0), minlength=N)
        _over = _exc > HUB_INPUT_CAP
        if _over.any():
            _sc = np.ones(N, dtype=np.float32)
            _sc[_over] = (HUB_INPUT_CAP / _exc[_over]).astype(np.float32)
            w *= _sc[_post]
            self.n_capped = int(_over.sum())
        else:
            self.n_capped = 0
        self.edge_w = w
        self.kc_to_mbon = z["kc_to_mbon"]
        self.superclass = z["superclass"]
        self.primary_type = z["primary_type"]
        self.pops = {
            k[4:]: z[k] for k in z.files if k.startswith("pop_")
        }
        self.rng = rng or np.random.default_rng(0)

        self.v = np.zeros(N, dtype=np.float32)
        self.refrac = np.zeros(N, dtype=np.float32)
        self.ext = np.zeros(N, dtype=np.float32)      # injected drive, mV/step
        self.alive = np.ones(N, dtype=bool)
        self.decay = np.float32(np.exp(-DT_MS / TAU_M_MS))

        # plasticity state
        self.w_mult = np.ones(len(self.kc_to_mbon), dtype=np.float32)
        self.kc_tag = np.zeros(len(self.kc_to_mbon), dtype=np.float32)
        self.tag_decay = np.float32(np.exp(-DT_MS / KC_TAG_TAU_MS))
        kc_pre = z["edge_pre"][self.kc_to_mbon]
        self._kc_pre_of_syn = kc_pre
        self._base_w_kc = self.edge_w[self.kc_to_mbon].copy()

        # rolling spike-rate estimate per neuron (for readouts), tau ~200 ms
        self.rate = np.zeros(N, dtype=np.float32)
        self.rate_decay = np.float32(np.exp(-DT_MS / 200.0))

        # Steering readout. sim/probe_steering.py measures which descending
        # neurons actually carry a lateralised odour signal and writes them here;
        # without that file we fall back to the DNa01/DNa02 pair, which is the
        # textbook answer but only four cells and far too noisy to steer with.
        self.steer_idxL = self.steer_idxR = None
        _sr = os.path.join(os.path.dirname(npz_path), "steer_readout.npz")
        if os.path.exists(_sr):
            r = np.load(_sr)
            self.steer_idxL = r["left_idx"].astype(np.int64)
            self.steer_idxR = r["right_idx"].astype(np.int64)
            wl, wr = r["left_w"].astype(np.float32), r["right_w"].astype(np.float32)
            self.steer_wL = wl / max(wl.sum(), 1e-6)
            self.steer_wR = wr / max(wr.sum(), 1e-6)
        # Membrane noise was 61% of the entire runtime: drawing 166,700 Gaussians
        # every simulated millisecond cost ~2 ms, more than propagating 25.6M
        # synapses. Draw a large pool once and take a random slice per step - the
        # per-step distribution is identical and the cost drops to a memcpy.
        self._noise_pool = self.rng.standard_normal(1 << 23, dtype=np.float32)
        self._pool_max = len(self._noise_pool) - N - 1
        self._fired = np.zeros(N, dtype=np.float32)
        # Kenyon cells are near-silent at rest in a real fly - that low
        # spontaneous rate is what makes their odour code sparse and separable.
        # Giving them the same tonic background as everything else made them
        # 8-15% active on every odour and left the evoked patterns 75% similar,
        # i.e. barely discriminable. Hold their background down so their activity
        # is driven by odour rather than by the model's own noise floor.
        self._tonic_scale = np.ones(N, dtype=np.float32)
        if len(self.pops.get("kenyon", [])):
            self._tonic_scale[self.pops["kenyon"]] = 0.10
        # Kenyon cells are coincidence detectors: a real KC needs several
        # projection neurons active together before it fires, which is what makes
        # its odour code sparse. A single shared threshold does not capture that,
        # so theirs is raised. KC_THRESH_MULT is swept in sim/diag_apl.py.
        self.v_thresh = np.full(N, V_THRESH, dtype=np.float32)
        if len(self.pops.get("kenyon", [])):
            self.v_thresh[self.pops["kenyon"]] = V_THRESH * KC_THRESH_MULT
        self._dt_gain = np.float32(DT_MS)          # per-step drive is a rate
        self._noise_gain = np.float32(np.sqrt(DT_MS))  # noise is a random walk
        self._steer_seeded = False
        self._mb_dirty = False
        self.steer_ema = 0.0
        self.steer_out = 0.0
        self.steer_decay = np.float32(np.exp(-DT_MS / STEER_ADAPT_TAU_MS))

        self.spike_count_total = 0
        self.steps = 0

    # -- population helpers ------------------------------------------------
    def pop(self, name):
        return self.pops[name]

    def pop_rate(self, name):
        """Mean firing rate of a population, in Hz."""
        idx = self.pops[name]
        if len(idx) == 0:
            return 0.0
        return float(self.rate[idx].mean()) * 1000.0 / DT_MS

    def inject(self, name, mv_per_step):
        """Drive a sensory population with a depolarising current."""
        idx = self.pops[name]
        if len(idx):
            self.ext[idx] += mv_per_step

    def clear_input(self):
        self.ext[:] = 0.0

    # -- one 1 ms step -----------------------------------------------------
    def step(self, excitability=1.0, noise_mv=NOISE_MV, tonic_mv=TONIC_MV):
        N = self.N
        self.v *= self.decay
        if noise_mv:
            o = int(self.rng.integers(0, self._pool_max))
            self.v += self._noise_pool[o:o + N] * (np.float32(noise_mv) * self._noise_gain)
        self.v += (
            self.ext + np.float32(tonic_mv) * self._tonic_scale
        ) * (np.float32(excitability) * self._dt_gain)

        self.refrac -= DT_MS
        can_fire = (self.refrac <= 0) & self.alive
        spiking = np.flatnonzero((self.v >= self.v_thresh) & can_fire).astype(np.int64)

        if len(spiking):
            self.v[spiking] = V_RESET
            self.refrac[spiking] = T_REFRAC_MS
            # event-driven propagation: gather only the out-edges of spikers
            # Event-driven propagation: only the out-edges of cells that
            # actually spiked are touched. The compiled kernel does the scatter
            # in place and is ~16x faster than the NumPy equivalent; it falls
            # back automatically if numba is not installed.
            propagate(
                spiking, self.out_start, self.out_end,
                self.edge_post, self.edge_w, self.v, N,
            )

        # rate estimate
        self.rate *= self.rate_decay
        if len(spiking):
            self.rate[spiking] += 1.0 - self.rate_decay

        # adaptive steering command from the lateralised descending ensemble
        if self.steer_idxL is not None:
            L = float(self.rate[self.steer_idxL] @ self.steer_wL) * 1000.0 / DT_MS
            R = float(self.rate[self.steer_idxR] @ self.steer_wR) * 1000.0 / DT_MS
        else:
            L = self.pop_rate("dn_steer_L")
            R = self.pop_rate("dn_steer_R")
        raw = L - R
        if not self._steer_seeded:
            # start the running mean where the animal actually is
            self.steer_ema = raw
            self._steer_seeded = True
        self.steer_ema = float(
            self.steer_ema * self.steer_decay + raw * (1.0 - self.steer_decay)
        )
        self.steer_out = math.tanh((raw - self.steer_ema) / STEER_SCALE)

        self.spike_count_total += len(spiking)
        self.steps += 1
        return spiking

    def measure_baseline(self, pops, settle_ms=900, measure_ms=600):
        """Resting rate of each population under tonic drive alone.

        Motor pools sit far above the global mean rate because they receive
        convergent input, so every downstream readout has to be a change from
        this baseline rather than an absolute rate.
        """
        v, refrac, rate = self.v.copy(), self.refrac.copy(), self.rate.copy()
        self.v[:] = 0.0; self.refrac[:] = 0.0; self.rate[:] = 0.0
        acc = {k: 0.0 for k in pops}
        n = 0
        for t in range(settle_ms + measure_ms):
            self.clear_input()
            self.step()
            if t >= settle_ms:
                for k in pops:
                    acc[k] += self.pop_rate(k)
                n += 1
        self.v, self.refrac, self.rate = v, refrac, rate
        self.baseline = {k: acc[k] / max(1, n) for k in pops}
        return self.baseline

    def measure_steer_reference(self, cue_amp, settle_ms=900, measure_ms=700):
        """Steering-DN rates under a SYMMETRIC odour, i.e. an odour straight ahead.

        The fly has a constitutive left/right bias in this pathway: driving either
        antenna alone pushes the steering command rightward, just by different
        amounts. Some of that is real asymmetry and some is MaleCNS being better
        proofread on the right (1,699 right-projecting ORNs against 931 left). Any
        turn signal therefore has to be read as a departure from what a
        dead-ahead odour produces, not as a raw left/right difference.
        """
        v, refrac, rate = self.v.copy(), self.refrac.copy(), self.rate.copy()
        self.v[:] = 0.0; self.refrac[:] = 0.0; self.rate[:] = 0.0
        accL = accR = 0.0
        n = 0
        for t in range(settle_ms + measure_ms):
            self.clear_input()
            self.inject("olfactory_L", cue_amp)
            self.inject("olfactory_R", cue_amp)
            self.step()
            if t >= settle_ms:
                if self.steer_idxL is not None:
                    accL += float(self.rate[self.steer_idxL] @ self.steer_wL) * 1000.0
                    accR += float(self.rate[self.steer_idxR] @ self.steer_wR) * 1000.0
                else:
                    accL += self.pop_rate("dn_steer_L")
                    accR += self.pop_rate("dn_steer_R")
                n += 1
        self.v, self.refrac, self.rate = v, refrac, rate
        self.steer_ref = (accL / max(1, n), accR / max(1, n))
        return self.steer_ref

    def steer_signal(self):
        """Turn command from DNa01/DNa02. Positive means turn left.

        This is the ADAPTED imbalance: instantaneous left-minus-right steering-DN
        activity with a 1.5 s running mean removed. DNa01/DNa02 drive ipsilateral
        turning, so a rising left-DN signal turns the fly left.
        """
        return self.steer_out

    def steer_pair(self):
        """Left- and right-preferring steering-ensemble rates, in Hz.

        These are what actually drives turning. The dn_steer_L/R populations are
        the DNa01/DNa02 pair and are kept only as a reference - four cells, far
        too noisy to steer on, which is why the ensemble exists.
        """
        if self.steer_idxL is None:
            return self.pop_rate("dn_steer_L"), self.pop_rate("dn_steer_R")
        L = float(self.rate[self.steer_idxL] @ self.steer_wL) * 1000.0 / DT_MS
        R = float(self.rate[self.steer_idxR] @ self.steer_wR) * 1000.0 / DT_MS
        return L, R

    def delta_rate(self, name):
        """Firing rate of a population relative to its resting baseline."""
        return self.pop_rate(name) - getattr(self, "baseline", {}).get(name, 0.0)

    # -- mushroom-body plasticity -----------------------------------------
    def update_plasticity(self, spiking, dopamine_drive):
        """Depress KC->MBON synapses whose KC was recently active while
        dopaminergic neurons were firing. dopamine_drive is 0..1."""
        self.kc_tag *= self.tag_decay
        if len(spiking):
            # reused scratch buffer: this runs every millisecond
            self._fired[:] = 0.0
            self._fired[spiking] = 1.0
            self.kc_tag += self._fired[self._kc_pre_of_syn]
        # Fast path: with no dopamine and nothing yet learned there is nothing to
        # write, and this runs every simulated millisecond.
        if dopamine_drive <= 0.0 and not self._mb_dirty:
            return
        if dopamine_drive > 0.0:
            self.w_mult -= LEARN_RATE * dopamine_drive * np.minimum(self.kc_tag, 1.0)
            self._mb_dirty = True
        # slow drift back toward naive
        self.w_mult += FORGET_RATE * (1.0 - self.w_mult)
        np.clip(self.w_mult, W_MULT_MIN, 1.0, out=self.w_mult)
        self.edge_w[self.kc_to_mbon] = self._base_w_kc * self.w_mult

    @property
    def learned_depression(self):
        """0 = naive, 1 = maximally depressed. A scalar summary of what the
        mushroom body has learned."""
        return float(1.0 - self.w_mult.mean())

    # -- damage ------------------------------------------------------------
    def kill_random(self, n, rng=None):
        """Neuron death from energy failure. Permanently silences cells."""
        rng = rng or self.rng
        live = np.flatnonzero(self.alive)
        if len(live) == 0 or n <= 0:
            return 0
        n = int(min(n, len(live)))
        victims = rng.choice(live, size=n, replace=False)
        self.alive[victims] = False
        self.v[victims] = 0.0
        return n

    @property
    def n_dead(self):
        return int((~self.alive).sum())
