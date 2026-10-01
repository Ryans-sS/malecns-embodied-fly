"""Metabolism, homeostasis, repair, growth and death for the simulated fly.

This layer is MODEL, not measurement. The connectome is real data; the numbers
below are a plausible energy budget chosen so that a fly which never eats dies in
minutes rather than milliseconds or days. What it does encode faithfully is the
structure of the problem: spiking costs energy, energy comes only from ingested
food, water is lost continuously and cannot be stored, damage accrues and must be
repaired at a cost, and a body that runs out of fuel kills its own neurons.

Internal state feeds back on the brain two ways, both of which are real
phenomena in flies even though the mapping here is coarse:
  - global excitability falls as energy is depleted,
  - the gain of a need's sensory channel rises with that need (hunger increases
    sugar-neuron sensitivity via dopaminergic modulation).
"""
import numpy as np

from sim.brain import ENERGY_PER_SPIKE, ENERGY_BASELINE_PER_NEURON_PER_S


class Body:
    # capacities (arbitrary but internally consistent units)
    ENERGY_FULL = 1.00
    WATER_FULL = 1.00
    SODIUM_FULL = 1.00
    CROP_MAX = 0.60

    # drains per second
    BASAL_BURN = 0.0010          # scales with mass
    WATER_LOSS = 0.0035          # evaporation, always
    SODIUM_LOSS = 0.00040
    LOCOMOTION_COST = 0.0015     # at full leg drive

    # gains
    CROP_ABSORB = 0.030          # crop -> energy per second
    INGEST_RATE = 0.10           # per second while actively feeding

    # wear and tear
    AGE_DAMAGE = 0.00018         # per second, unavoidable
    STARVE_DAMAGE = 0.010        # per second at zero energy
    REPAIR_RATE = 0.0035         # damage removed per second when repairing
    REPAIR_COST = 0.020          # energy per unit damage repaired
    REPAIR_MIN_ENERGY = 0.45     # only repairs with a comfortable surplus

    GROW_MIN_ENERGY = 0.70       # only grows with a real surplus
    GROW_RATE = 0.0016           # mass per second
    GROW_COST = 0.9              # energy per unit mass

    # death
    LETHAL_DAMAGE = 1.0
    LETHAL_DEAD_FRACTION = 0.30
    NEURON_KILL_PER_S = 900      # at full energy deficit

    def __init__(self, n_neurons, rng=None):
        self.rng = rng or np.random.default_rng(1)
        self.n_neurons = n_neurons
        self.energy = 0.62
        self.water = 0.75
        self.sodium = 0.60
        self.crop = 0.0
        self.damage = 0.0
        self.mass = 1.0
        self.age_s = 0.0
        self.alive = True
        self.cause_of_death = None
        # bookkeeping for the log
        self.ingested = {"sucrose": 0.0, "water": 0.0, "salt": 0.0, "bitter": 0.0}
        self.spikes_last = 0
        self.neural_burn_last = 0.0
        self.repairing = False
        self.growing = False

    # -- needs, 0 (satisfied) .. 1 (desperate) -----------------------------
    @property
    def hunger(self):
        return float(np.clip(1.0 - (self.energy + self.crop) / self.ENERGY_FULL, 0, 1))

    @property
    def thirst(self):
        return float(np.clip(1.0 - self.water / self.WATER_FULL, 0, 1))

    @property
    def salt_need(self):
        return float(np.clip(1.0 - self.sodium / self.SODIUM_FULL, 0, 1))

    @property
    def excitability(self):
        """Global gain on sensory drive.

        An earlier version scaled this hard with energy, which produced a death
        spiral: low energy -> weak network -> no feeding motor command -> lower
        energy. That is backwards. Food-deprived flies become MORE active, not
        less, so the dependence is now weak and hunger pushes it slightly up.
        """
        return float(np.clip(
            0.95 + 0.05 * (self.energy / self.ENERGY_FULL) + 0.08 * self.hunger,
            0.90, 1.10,
        ))

    # -- one tick ----------------------------------------------------------
    def tick(self, dt_s, n_spikes, n_alive_neurons, locomotion_drive, feeding, brain):
        if not self.alive:
            return
        self.age_s += dt_s

        # --- energy out ---
        neural = n_spikes * ENERGY_PER_SPIKE + (
            n_alive_neurons * ENERGY_BASELINE_PER_NEURON_PER_S * dt_s
        )
        self.neural_burn_last = neural
        self.spikes_last = n_spikes
        basal = self.BASAL_BURN * self.mass * dt_s
        moving = self.LOCOMOTION_COST * float(np.clip(locomotion_drive, 0, 1)) * dt_s
        self.energy -= neural + basal + moving

        # --- energy in: crop empties into circulation ---
        if self.crop > 0.0:
            moved = min(self.crop, self.CROP_ABSORB * dt_s)
            self.crop -= moved
            self.energy += moved

        # --- water and salt always leak ---
        self.water -= self.WATER_LOSS * dt_s * (1.0 + 0.5 * float(np.clip(locomotion_drive, 0, 1)))
        self.sodium -= self.SODIUM_LOSS * dt_s

        # --- damage ---
        self.damage += self.AGE_DAMAGE * dt_s
        deficit = max(0.0, -self.energy)
        if self.energy <= 0.0:
            self.damage += self.STARVE_DAMAGE * dt_s
            # a body with no fuel cannot keep all its neurons alive
            n_kill = int(self.NEURON_KILL_PER_S * dt_s * min(1.0, 0.2 + deficit))
            if n_kill:
                brain.kill_random(n_kill, self.rng)

        # --- repair, paid for in energy ---
        self.repairing = False
        if self.damage > 0.0 and self.energy > self.REPAIR_MIN_ENERGY:
            fixed = min(self.damage, self.REPAIR_RATE * dt_s)
            cost = fixed * self.REPAIR_COST
            if cost <= self.energy:
                self.damage -= fixed
                self.energy -= cost
                self.repairing = True

        # --- growth, also paid for ---
        self.growing = False
        if self.energy > self.GROW_MIN_ENERGY:
            dm = self.GROW_RATE * dt_s
            cost = dm * self.GROW_COST
            if cost <= self.energy:
                self.mass += dm
                self.energy -= cost
                self.growing = True

        self.energy = float(np.clip(self.energy, -0.25, self.ENERGY_FULL))
        self.water = float(np.clip(self.water, 0.0, self.WATER_FULL))
        self.sodium = float(np.clip(self.sodium, 0.0, self.SODIUM_FULL))
        self.damage = float(np.clip(self.damage, 0.0, 1.2))

        # --- death ---
        if self.water <= 0.0:
            self._die("desiccation")
        elif self.damage >= self.LETHAL_DAMAGE:
            self._die("accumulated damage")
        elif brain.n_dead / self.n_neurons >= self.LETHAL_DEAD_FRACTION:
            self._die("neural degeneration from starvation")

    def ingest(self, resource, dt_s):
        """Swallow whatever the dispenser is delivering."""
        amount = self.INGEST_RATE * dt_s
        if resource == "sucrose":
            room = self.CROP_MAX - self.crop
            got = max(0.0, min(amount, room))
            self.crop += got
            self.ingested["sucrose"] += got
            # food is wet
            self.water = min(self.WATER_FULL, self.water + got * 0.25)
        elif resource == "water":
            got = min(amount, self.WATER_FULL - self.water)
            self.water += got
            self.ingested["water"] += got
        elif resource == "salt":
            got = min(amount, self.SODIUM_FULL - self.sodium)
            self.sodium += got
            self.ingested["salt"] += got
        elif resource == "bitter":
            # noxious: costs energy and does damage
            self.energy -= amount * 0.20
            self.damage += amount * 0.06
            self.ingested["bitter"] += amount

    def _die(self, cause):
        self.alive = False
        self.cause_of_death = cause

    def snapshot(self):
        return {
            "alive": self.alive,
            "cause_of_death": self.cause_of_death,
            "age_s": round(self.age_s, 1),
            "energy": round(self.energy, 4),
            "crop": round(self.crop, 4),
            "water": round(self.water, 4),
            "sodium": round(self.sodium, 4),
            "damage": round(self.damage, 4),
            "mass": round(self.mass, 4),
            "hunger": round(self.hunger, 3),
            "thirst": round(self.thirst, 3),
            "salt_need": round(self.salt_need, 3),
            "repairing": self.repairing,
            "growing": self.growing,
            "ingested": {k: round(v, 3) for k, v in self.ingested.items()},
        }
