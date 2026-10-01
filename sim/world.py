"""The arena: a walled dish with request pads, and the sensory/motor coupling.

The request pads are the point of the experiment. Each pad has

  a CUE - a fixed sparse subset of the 2,639 real olfactory receptor neurons,
          which is what an odour actually is to a fly: a sparse ORN pattern.
          The fly smells the cue whenever it is near the pad.

  a PAYLOAD - what the pad dispenses once the fly has dwelt on it. Delivery
          drives a real sensory population: a gustatory receptor-neuron class for
          the ingestible payloads, the hygrosensory neurons for water.

MaleCNS types its gustatory neurons anatomically (LB* labellar, LgLG* leg, WG*
wing, taste peg, pharyngeal) and does NOT record which tastant each class
responds to. So the mapping from payload to GRN class was not chosen by hand: it
was measured by driving each class and reading the proboscis motor pool
(sim/probe.py). Which pad the fly then favours, and whether that tracks its
internal state, is the measurement - not an input.

Locomotion reads real leg motor neurons: left-vs-right imbalance across the
front/mid/hind leg motor pools steers, their sum sets speed. Ingestion reads the
proboscis muscle motor neurons. The muscle model itself (rate -> velocity) is
model, not data.
"""
import math
import numpy as np

ARENA_R = 30.0          # mm
PAD_R = 3.2
CUE_R = 9.0             # how far the pad's odour carries
DWELL_MS = 600.0        # time on a pad before it dispenses
DELIVER_MS = 6000.0     # how long a dispense lasts
REFILL_MS = 5000.0      # pad cooldown
FEEDING_STOP = 0.12     # locomotion multiplier while feeding

# Sensory drive amplitudes, mV per 1 ms step (model parameters).
# Odour cue drive per antenna. Kept low deliberately: sim/probe.py showed that
# above ~3 mV the network saturates and the steering descending neurons fall
# silent, destroying the turn signal. 0.55 sits in the linear regime.
# 1.80 mV drove the receptor population to ~52 Hz and gave a chemotaxis d-prime of
# 18.7 - far more separation than needed, and it saturated the olfactory system
# badly enough to suppress the feeding pathway. Backed off: d-prime of 2-3 is
# ample, and the brain stays in a sane firing regime.
# 0.75 mV was chosen to stop the cue swamping the feeding pathway - but the real
# cause of that was a spurious mechano_head injection, since removed. Measured at
# 0.75 the steering d-prime is 0.22, i.e. no chemotaxis at all; at 2.5 it is 23.4,
# and feeding is unaffected at every level tested (sim/diag_operating_point.py).
CUE_DRIVE = 2.50
CUE_ANTENNA_SHARPNESS = 0.85   # how strongly bearing biases one antenna over the other
STEER_GAIN = 1.70              # rad/s at a full-strength turn command
WANDER_YAW = 0.14              # exploratory turning, reduced so chemotaxis can win
# 2.5 mV already puts the proboscis ~+22 Hz above rest, well past the +8 Hz it
# needs to swallow, without pinning the receptors at their refractory ceiling the
# way 4 mV did. Response is flat from 3 mV upward (sim/diag_feeding.py).
TASTE_DRIVE = 2.5
HYGRO_DRIVE = 1.5
BITTER_DRIVE = 1.5
WALL_DRIVE = 1.1        # mechanosensory bristles on contact
LIGHT_DRIVE = 0.10

# Locomotion model
MAX_SPEED = 9.0         # mm/s at full leg drive
MAX_YAW = 2.6           # rad/s at full imbalance

# Spontaneous foraging. Baseline leg-motor firing means "standing", so a resting
# fly reads out zero forward drive and would never leave the spot it started on.
# Real flies walk spontaneously and walk MORE when food-deprived, driven by
# central pattern generators and neuromodulation that a static connectome does
# not capture. This bias is therefore an explicit model term, scaled by hunger,
# on top of whatever the leg motor pools are actually doing.
FORAGE_BIAS = 0.28
FORAGE_HUNGER_GAIN = 0.45
# Both thresholds are in Hz ABOVE each pool's resting baseline (see
# Brain.measure_baseline). Absolute rates are meaningless here because motor
# pools rest at 12-22 Hz while the brain as a whole rests at 1.9 Hz.
LEG_RATE_REF = 10.0      # delta-Hz counting as "full" leg drive
PROBOSCIS_FEED_DELTA = 8.0   # delta-Hz above which the fly is actually ingesting

# Payload -> the real sensory populations it drives.
#
# These were MEASURED, not chosen: sim/probe.py drove every gustatory class and
# read the 67 proboscis motor neurons. Taste-peg GRNs were the ONLY class that
# excites feeding (+22.6 Hz at 4 mV). Pharyngeal GRNs suppress it at every
# amplitude (-8 Hz), so they are the aversive channel.
#
# A first run showed why that matters: routing water through the hygrosensory
# neurons alone made drinking physically impossible, because hygro drive
# SUPPRESSES the proboscis (-9.8 Hz). The fly stood on a dispensing water pad and
# its feeding neurons went quiet. So every ingestible payload now also drives the
# taste-peg pathway, which is anatomically reasonable - taste pegs sit on the
# inner labellar surface and contact any liquid the fly is taking up, reporting
# "there is something ingestible here" while the modality-specific channel
# reports what it is.
PAYLOAD_SENSORS = {
    "sucrose": [("grn_tastepeg", TASTE_DRIVE)],
    "water": [("grn_tastepeg", TASTE_DRIVE), ("hygro", HYGRO_DRIVE)],
    "salt": [("grn_tastepeg", TASTE_DRIVE), ("grn_leg", TASTE_DRIVE)],
    "bitter": [("grn_pharyngeal", BITTER_DRIVE)],
    "none": [],
}

# Which need each payload relieves - used only for the hunger-gain feedback and
# for the dopamine signal, both of which are model.
PAYLOAD_NEED = {
    "sucrose": "hunger",
    "water": "thirst",
    "salt": "salt_need",
    "bitter": None,
    "none": None,
}


class Pad:
    def __init__(self, name, x, y, payload, cue_idx, color, cue_L=None, cue_R=None):
        self.name = name
        self.x, self.y = x, y
        self.payload = payload
        self.cue_idx = cue_idx
        # split once at construction - this is read every simulated millisecond
        self.cue_L = cue_L
        self.cue_R = cue_R
        self.color = color
        self.dwell_ms = 0.0
        self.deliver_ms = 0.0
        self.cooldown_ms = 0.0
        self.presses = 0
        self.consumed = 0.0

    @property
    def dispensing(self):
        return self.deliver_ms > 0.0


class World:
    def __init__(self, brain, rng=None):
        self.rng = rng or np.random.default_rng(2)
        self.brain = brain
        self.t_ms = 0.0
        self.x, self.y, self.theta = 0.0, 0.0, 0.3
        self.speed = 0.0
        self.yaw = 0.0
        self.feeding_on = None

        olf = brain.pop("olfactory")
        # An odour is a combination of GLOMERULI, not a random handful of
        # receptor neurons.
        #
        # Cues were previously random ORN subsets. MaleCNS types every olfactory
        # receptor neuron by its glomerulus (ORN_DA1, ORN_VA1d, ... 53 types,
        # ~50 cells each), and a random split cuts every glomerulus across
        # several pads - so each cue partially activated almost all of them,
        # drove almost all projection neurons, and the Kenyon-cell responses to
        # the five pads came out nearly identical. Partitioning by glomerulus
        # instead gives each pad a genuinely distinct olfactory identity, which
        # is the form the antennal lobe is built to resolve.
        glom = {}
        for i in olf:
            glom.setdefault(str(brain.primary_type[i]), []).append(i)
        gnames = sorted(glom)
        self.rng.shuffle(gnames)

        layout = [
            ("SUGAR", "sucrose", "#d98c0a"),
            ("WATER", "water", "#1f8fcf"),
            ("SALT", "salt", "#7a5bd0"),
            ("BITTER", "bitter", "#c0392b"),
            ("BLANK", "none", "#8795a4"),
        ]
        per = max(1, len(gnames) // len(layout))
        self.pads = []
        self.pad_glomeruli = {}
        for k, (name, payload, color) in enumerate(layout):
            ang = 2 * math.pi * k / len(layout) - math.pi / 2
            r = ARENA_R * 0.62
            mine = gnames[k * per:(k + 1) * per]
            cue = np.sort(np.concatenate([glom[g] for g in mine])).astype(np.int32)
            self.pad_glomeruli[name] = mine
            self.pads.append(
                Pad(name, r * math.cos(ang), r * math.sin(ang), payload, cue, color,
                    cue_L=np.intersect1d(cue, brain.pop("olfactory_L")),
                    cue_R=np.intersect1d(cue, brain.pop("olfactory_R")))
            )

        self.events = []

    # -- sensing -----------------------------------------------------------
    def apply_senses(self, body):
        b = self.brain
        b.clear_input()

        # ambient light: constant overhead illumination
        b.inject("visual", LIGHT_DRIVE)

        # proprioception ticks along with movement
        b.inject("mechano_proprio", 0.25 + 0.55 * min(1.0, self.speed / MAX_SPEED))

        # wall contact
        d_wall = ARENA_R - math.hypot(self.x, self.y)
        if d_wall < 1.5:
            b.inject("mechano_tactile", WALL_DRIVE)

        # Pad cues: smelling the buttons, bilaterally.
        #
        # A cue is a fixed sparse ORN subset, but WHICH antenna receives it is set
        # by where the pad is relative to the fly's heading. Splitting the cue
        # across the two antennal lobes is what makes chemotaxis possible at all -
        # a single-antenna fly has no side information to steer on. Laterality
        # comes from real projection targets: each ORN innervates AL(L) or AL(R).
        self.near = []
        for p in self.pads:
            d = math.hypot(self.x - p.x, self.y - p.y)
            if d >= CUE_R:
                continue
            strength = CUE_DRIVE * (1.0 - d / CUE_R)
            # bearing of the pad in the fly's own frame: +1 hard left, -1 hard right
            bearing = math.atan2(p.y - self.y, p.x - self.x) - self.theta
            lat = math.sin(bearing)
            gL = 1.0 + CUE_ANTENNA_SHARPNESS * lat
            gR = 1.0 - CUE_ANTENNA_SHARPNESS * lat
            b.ext[p.cue_L] += np.float32(strength * gL)
            b.ext[p.cue_R] += np.float32(strength * gR)
            self.near.append(p.name)

        # taste / hygro, only while a pad is actually dispensing and the fly is on it
        self.feeding_on = None
        for p in self.pads:
            if not p.dispensing:
                continue
            if math.hypot(self.x - p.x, self.y - p.y) > PAD_R:
                continue
            # hunger raises the gain of the channel that relieves it: real
            # phenomenon in flies, coarse mapping here
            need = PAYLOAD_NEED[p.payload]
            # Hunger raises the gain of the channel that relieves it, but only
            # modestly: a full doubling pushed the taste drive to 7+ mV, well past
            # the 4 mV optimum measured in sim/probe.py, which saturated the
            # taste-peg neurons at 334 Hz and stopped driving the proboscis at all.
            gain = 1.0 + 0.15 * (getattr(body, need) if need else 0.0)
            for popname, amp in PAYLOAD_SENSORS[p.payload]:
                b.inject(popname, amp * gain)
            # NOTE: an earlier version also drove mechano_head here, meaning to
            # represent labellar contact. That population is the whole head
            # mechanosensory class, which drives grooming and SUPPRESSES feeding -
            # sim/probe.py measured -10.55 Hz on the proboscis pool, and in the
            # arena it cancelled the taste signal almost exactly (+29.4 Hz alone
            # vs +3.96 Hz with it), so the fly could never swallow. Grooming and
            # feeding being mutually exclusive is real; using that channel for
            # food contact was not.
            self.feeding_on = p

    # -- acting ------------------------------------------------------------
    def read_motor(self):
        b = self.brain
        left = float(np.mean([
            b.delta_rate("mn_leg_front_L"),
            b.delta_rate("mn_leg_mid_L"),
            b.delta_rate("mn_leg_hind_L"),
        ]))
        right = float(np.mean([
            b.delta_rate("mn_leg_front_R"),
            b.delta_rate("mn_leg_mid_R"),
            b.delta_rate("mn_leg_hind_R"),
        ]))
        drive = float(np.clip((left + right) / (2 * LEG_RATE_REF), 0.0, 1.0))
        imbalance = float(np.clip((right - left) / LEG_RATE_REF, -1.0, 1.0))
        proboscis = b.delta_rate("mn_proboscis")
        return drive, imbalance, proboscis, left, right

    def step(self, dt_ms, body):
        drive, imbalance, proboscis, left, right = self.read_motor()
        forage = FORAGE_BIAS + FORAGE_HUNGER_GAIN * max(body.hunger, body.thirst)
        locomotion = float(np.clip(drive + forage, 0.0, 1.0))
        # Feeding flies stop walking. Without this the fly triggers a dispenser
        # and immediately wanders off the pad before it can ingest anything.
        if self.feeding_on is not None:
            locomotion *= FEEDING_STOP
        self.t_ms += dt_ms
        dt_s = dt_ms / 1000.0

        # exploratory turning so a silent network still wanders rather than
        # standing perfectly still (model: a stand-in for CPG//wind noise)
        wander = float(self.rng.normal(0.0, 0.9))
        # Turn command from DNa01/DNa02, the real steering descending neurons.
        steer = self.brain.steer_signal()
        self.speed = MAX_SPEED * locomotion
        self.yaw = (
            STEER_GAIN * steer                 # chemotaxis, via the steering DNs
            + MAX_YAW * imbalance * 0.35       # leg-pool asymmetry
            + wander * WANDER_YAW              # exploration
        )
        self.steer_cmd = steer

        self.theta += self.yaw * dt_s
        self.x += math.cos(self.theta) * self.speed * dt_s
        self.y += math.sin(self.theta) * self.speed * dt_s

        # arena wall
        r = math.hypot(self.x, self.y)
        if r > ARENA_R - 0.6:
            self.x *= (ARENA_R - 0.6) / r
            self.y *= (ARENA_R - 0.6) / r
            self.theta += math.pi * 0.5

        # pads: dwell -> press -> dispense -> cooldown
        reward_need = 0.0
        punish = 0.0
        for p in self.pads:
            on_pad = math.hypot(self.x - p.x, self.y - p.y) <= PAD_R
            if p.deliver_ms > 0:
                p.deliver_ms -= dt_ms
                if p.deliver_ms <= 0:
                    p.cooldown_ms = REFILL_MS
            elif p.cooldown_ms > 0:
                p.cooldown_ms -= dt_ms
                p.dwell_ms = 0.0
            elif on_pad:
                p.dwell_ms += dt_ms
                if p.dwell_ms >= DWELL_MS:
                    p.dwell_ms = 0.0
                    p.deliver_ms = DELIVER_MS
                    p.presses += 1
                    self.events.append((round(self.t_ms / 1000.0, 1), p.name, "pressed"))
            else:
                p.dwell_ms = max(0.0, p.dwell_ms - dt_ms * 0.5)

        # Reinforcement comes from TASTE CONTACT, not from swallowing.
        #
        # Previously the dopamine signal was gated behind the proboscis motor
        # neurons crossing the feeding threshold, so a fly that never managed to
        # swallow never got a teaching signal and the mushroom body could never
        # learn anything - `learned` sat at 0.0000 in every run to date. In real
        # flies sugar on the labellum drives the reward DANs directly, which is
        # why appetitive conditioning works with brief sugar presentation and no
        # ingestion at all. So contact with a dispensing pad is what teaches;
        # swallowing is what feeds.
        p = self.feeding_on
        ingesting = False
        if p is not None:
            need = PAYLOAD_NEED[p.payload]
            if need:
                reward_need = max(reward_need, getattr(body, need))
            if p.payload == "bitter":
                punish = 1.0
            # ingestion still requires the motor command
            if proboscis >= PROBOSCIS_FEED_DELTA:
                body.ingest(p.payload, dt_s)
                p.consumed += dt_s
                ingesting = True

        return {
            "drive": locomotion,
            "neural_drive": drive,
            "imbalance": imbalance,
            "proboscis": proboscis,
            "leg_L": left,
            "leg_R": right,
            "steer": steer,
            "ingesting": ingesting,
            "reward": reward_need,
            "punish": punish,
        }

    def snapshot(self):
        return {
            "t": round(self.t_ms / 1000.0, 2),
            "arena_r": ARENA_R,
            "fly": {
                "x": round(self.x, 3), "y": round(self.y, 3),
                "theta": round(self.theta, 3),
                "speed": round(self.speed, 3),
            },
            "pads": [
                {
                    "name": p.name, "payload": p.payload,
                    "x": round(p.x, 2), "y": round(p.y, 2), "r": PAD_R,
                    "color": p.color,
                    "dispensing": p.dispensing,
                    "cooldown": round(max(0.0, p.cooldown_ms) / 1000.0, 1),
                    "dwell": round(p.dwell_ms / DWELL_MS, 2),
                    "presses": p.presses,
                    "consumed": round(p.consumed, 1),
                }
                for p in self.pads
            ],
            "events": self.events[-12:],
        }
