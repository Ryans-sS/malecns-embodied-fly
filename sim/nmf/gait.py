"""Tripod walking, generated locally and steered by descending input.

Real walking is not commanded joint-by-joint from the brain. Pattern generators
in the ventral nerve cord produce the stepping rhythm, and descending neurons
modulate its speed and direction. That is also the only honest option here:
MaleCNS annotates leg motor neurons by segment and side but not by target muscle,
so there is no data-supported mapping from 700 motor neurons onto 66 joint
degrees of freedom.

So this module owns the rhythm, and exposes exactly two inputs - drive and turn -
which the brain supplies from its own motor and descending populations. It is the
same division of labour NeuroMechFly v2 uses for its turning controller.

The rhythm is specified where it is actually observable: as a trajectory of each
FOOT, in the body frame. Joint angles are then solved for. An earlier version
prescribed the joint angles directly as sinusoids, and it did not walk - each
foot was on the ground only 13-31% of the time against the ~50% a tripod needs,
one foot never landed at all, and the fly skated and tipped. The reason is that
swinging a leg fore-and-aft about the body sweeps the foot through an arc, so
holding the other joints at neutral does not hold the foot on the ground: it
lifts away and touches down only where the arc crosses the floor. Prescribing
foot position instead makes "stay on the ground through stance" something the
trajectory states directly, and the per-leg Jacobian carries each leg's own
mirrored joint signs so the two sides behave the same.

flygym 2.1.0 ships no measured walking kinematics (only meshes), so the foot
path is still a model - a flat stance sweep and a raised swing return - but the
quantities in it are the ones gait is measured in: stride length, ground
clearance, and duty factor.
"""
from __future__ import annotations

import mujoco
import numpy as np

LEGS = ["lf", "lm", "lh", "rf", "rm", "rh"]
# Tripod gait: front-left, mid-right, hind-left swing together, then the other three.
TRIPOD = {"lf": 0.0, "rm": 0.0, "lh": 0.0, "rf": 0.5, "lm": 0.5, "rh": 0.5}

# Foot trajectory, millimetres in the body frame. These are gait quantities, not
# joint angles: the body advances one STRIDE_MM per cycle, so walking speed is
# STRIDE_MM * frequency.
STRIDE_MM = 1.80         # fore-aft foot travel during stance
LIFT_MM = 0.20           # ground clearance at mid-swing
SWING_FRAC = 0.35        # fraction of the cycle in swing; stance duty is the rest
# Height of the thorax above the floor during stance, millimetres. Every stance
# foot is placed on this one common plane, which is the whole point: referencing
# each foot to its OWN neutral height instead bakes in the model's neutral pose,
# and that pose is not a level six-point stance. Measured standing, it puts only
# three feet on the ground (both middle legs and one hind) and holds the body at a
# persistent 4 degree nose-up pitch, so the front feet never reach the floor at
# all - lf recorded 10 contact-ticks and rf 1, against lm's 56. A common plane
# makes a level stance the thing being solved for.
BODY_HEIGHT_MM = 1.15
# Where each leg's stroke is centred, fore-aft, relative to its neutral foot
# position. Measured, not chosen: scanning the fore-aft extent each foot can hold
# the stance plane through gives an interval 2.1-2.8 mm wide per leg, but the
# intervals are not centred on the neutral pose - the front legs' lies 0.30 mm
# behind it and the hind legs' 0.25 mm ahead. Centring every stroke on the neutral
# position, as the first version did, ran the front feet off the end of their
# reachable range at full protraction.
STROKE_CENTRE_MM = {"lf": -0.30, "lm": -0.05, "lh": 0.25,
                    "rf": -0.30, "rm": -0.05, "rh": 0.25}
# 4 Hz with a long stride, not 7 Hz with a short one: the reachable stroke is
# 2.1 mm, so the same ~10 mm/s can be had at a step rate the legs can actually
# track. At 7 Hz the measured joint tracking error was large enough that the feet
# never reached the ground - 0.63 of 6 feet down against 3.23 at 4 Hz.
BASE_FREQ_HZ = 4.0       # stepping frequency at full drive
MIN_FREQ_HZ = 1.5
# Body yaw per step cycle at full turn command, radians. Turning is expressed as
# a rotation of the whole body and each leg's stroke is derived from it, rather
# than as a per-side stride gain.
# 0.15 rather than the 0.30 first tried. Every turn level is kinematically
# reachable (the solve returns zero residual at all of them), but at 0.30 a full
# turn asks the outside front leg for a 90 degree joint sweep per cycle, which the
# servo cannot track at the stepping rate - and an untracked turn stops being a
# turn: measured yaw went from +52 deg/s at half command to -19 deg/s at full.
# At 0.15 the worst sweep is 78 degrees and the response stays monotonic.
TURN_PER_CYCLE = 0.15

# Inverse kinematics. The table is solved once at construction and indexed at
# run time, so the physics loop costs a lookup rather than a 6-DOF solve.
N_PHASE = 180
# Turn commands the tables are solved at; the run-time stroke is interpolated
# between them. A table per turn level, rather than one table scaled at run time:
# scaling the solved joint angles also flattens the swing lift, so the inside legs
# of a hard turn stopped clearing the ground and dragged.
TURN_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)
IK_JOINTS = ["thc_yaw", "thc_pitch", "thc_roll", "ctr_pitch", "fti_pitch", "tita_pitch"]
IK_DAMPING = 0.05        # Levenberg damping, keeps the solve away from singularities
IK_STEPS = 80            # Newton iterations per phase sample
JOINT_MARGIN = 0.05      # radians to stay clear of each joint's hard limit
# Six joints solving a three-dimensional foot position leaves a three-dimensional
# null space, and the solver will wander down it: the first version warm-started
# each phase from the previous solution and drifted to 55 degrees of joint
# excursion for a 1.15 mm foot move, which folded the legs up. Solving each phase
# cold from the neutral pose fixes that on its own.
#
# A null-space pull back towards the neutral posture was tried as well and had to
# come out: (I - J+J) built from a DAMPED pseudo-inverse is not an exact null-space
# projector, so the posture term leaked into the task and fought the foot target.
# It made legs look unreachable that are not - with it the front foot appeared
# unable to get below -1.03 mm at full protraction, where in fact it reaches
# -1.35 mm across a 2.1 mm stroke once the solve is left alone to converge.


def _act(leg: str, joint: str) -> str:
    table = {
        "thc_yaw": f"fly/c_thorax-{leg}_coxa-yaw-position",
        "thc_pitch": f"fly/c_thorax-{leg}_coxa-pitch-position",
        "thc_roll": f"fly/c_thorax-{leg}_coxa-roll-position",
        "ctr_pitch": f"fly/{leg}_coxa-{leg}_trochanterfemur-pitch-position",
        "fti_pitch": f"fly/{leg}_trochanterfemur-{leg}_tibia-pitch-position",
        "tita_pitch": f"fly/{leg}_tibia-{leg}_tarsus1-pitch-position",
        "adhesion": f"fly/{leg}_tarsus5-adhesion",
    }
    return table[joint]


def stance_sweep(r_xy, turn: float) -> np.ndarray:
    # The body-frame displacement a planted foot at r_xy must travel through
    # stance: minus the body's own motion, -(d + psi x r), for a body advancing
    # STRIDE_MM and yawing TURN_PER_CYCLE * turn over the cycle.
    psi = TURN_PER_CYCLE * float(turn)
    rx, ry = float(r_xy[0]), float(r_xy[1])
    return np.array([-(STRIDE_MM - psi * ry), -psi * rx])


def foot_offset(p: float, sweep) -> np.ndarray:
    """Where the foot should be at cycle phase p, relative to its neutral spot.

    Phase 0 starts swing. Through swing the foot arcs forward and clear of the
    ground; through stance it sweeps straight back along the floor, which is what
    carries the body forward and what the old joint-space version never did.
    """
    if p < SWING_FRAC:
        u = p / SWING_FRAC                       # return stroke, foot in the air
        return np.array([sweep[0] * (0.5 - u), sweep[1] * (0.5 - u),
                         LIFT_MM * np.sin(np.pi * u)])
    u = (p - SWING_FRAC) / (1.0 - SWING_FRAC)    # stance, foot on the floor
    return np.array([sweep[0] * (u - 0.5), sweep[1] * (u - 0.5), 0.0])


def in_swing(p: float) -> bool:
    return p < SWING_FRAC


def _blend(tab, k, turn):
    # Joint angles at phase row k for an arbitrary turn command.
    if turn <= TURN_LEVELS[0]:
        return tab[0, k]
    if turn >= TURN_LEVELS[-1]:
        return tab[-1, k]
    i = int(np.searchsorted(TURN_LEVELS, turn))
    lo, hi = TURN_LEVELS[i - 1], TURN_LEVELS[i]
    f = (turn - lo) / (hi - lo)
    return tab[i - 1, k] * (1.0 - f) + tab[i, k] * f


class TripodGait:
    """Generates leg joint targets; drive and turn come from the brain."""

    def __init__(self, sim, fly_name: str = "fly"):
        self.sim = sim
        self.fly = fly_name
        model = sim.mj_model
        names = [model.actuator(i).name for i in range(model.nu)]
        self.index = {n: i for i, n in enumerate(names)}
        self.nu = model.nu

        # Neutral pose = whatever the model was built to stand in. Everything
        # below is a departure from it, so the fly keeps its own posture.
        self.neutral = np.array(sim.mj_data.ctrl, dtype=float).copy()
        if not np.any(self.neutral):
            self.neutral = np.zeros(self.nu)

        self.phase = 0.0
        self.ctrl = self.neutral.copy()
        self._adhesion = {leg: 1.0 for leg in LEGS}
        self.table = self._solve_tables()

    # -- inverse kinematics ------------------------------------------------
    def _leg_joints(self, leg):
        """(actuator index, qpos address, dof address, lo, hi) per IK joint."""
        m = self.sim.mj_model
        out = []
        for jname in IK_JOINTS:
            ai = self.index.get(_act(leg, jname))
            if ai is None:
                continue
            ji = int(m.actuator_trnid[ai][0])
            lo, hi = m.jnt_range[ji]
            if not m.jnt_limited[ji]:
                lo, hi = -np.pi, np.pi
            out.append((ai, int(m.jnt_qposadr[ji]), int(m.jnt_dofadr[ji]),
                        float(lo) + JOINT_MARGIN, float(hi) - JOINT_MARGIN))
        return out

    def _solve_tables(self):
        """Solve the foot path into joint angles, once, for every leg and phase.

        Damped-least-squares Newton iteration on the real model Jacobian. Working
        from the model rather than from an analytic arm length means each leg's
        own mirrored axis signs are built in, which is what made the sinusoid
        version behave differently on the left and right sides at zero turn.
        """
        m, d = self.sim.mj_model, self.sim.mj_data
        qpos0 = np.array(d.qpos).copy()
        thorax = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{self.fly}/c_thorax")
        R = np.array(d.xmat[thorax]).reshape(3, 3)       # body -> world
        thorax_pos = np.array(d.xpos[thorax]).copy()

        jacp = np.zeros((3, m.nv))
        tables = {}
        for leg in LEGS:
            joints = self._leg_joints(leg)
            foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                     f"{self.fly}/{leg}_tarsus5")
            q0 = np.array([qpos0[j[1]] for j in joints])
            dofs = [j[2] for j in joints]
            lo = np.array([j[3] for j in joints])
            hi = np.array([j[4] for j in joints])

            d.qpos[:] = qpos0
            mujoco.mj_kinematics(m, d)
            p0 = R.T @ (np.array(d.xpos[foot]) - thorax_pos)   # foot in body frame
            p0[2] = -BODY_HEIGHT_MM          # all six feet onto one ground plane
            centre = np.array([STROKE_CENTRE_MM.get(leg, 0.0), 0.0, 0.0])

            # (turn level, phase, joint); the last phase row is the stand pose
            tab = np.zeros((len(TURN_LEVELS), N_PHASE + 1, len(joints)))
            resid = 0.0
            stance_r = (p0 + centre)[:2]
            for si, tlevel in enumerate(TURN_LEVELS):
                sweep = stance_sweep(stance_r, tlevel)
                for k in range(N_PHASE + 1):
                    # The extra row is mid-stance with no fore-aft offset: the pose
                    # to hold when the brain asks for no walking at all.
                    target = (p0 + centre + (np.zeros(3) if k == N_PHASE
                                             else foot_offset(k / N_PHASE, sweep)))
                    q = q0.copy()                        # cold start: no drift
                    err = target - p0
                    for _ in range(IK_STEPS):
                        for q_i, j in zip(q, joints):
                            d.qpos[j[1]] = q_i
                        mujoco.mj_kinematics(m, d)
                        mujoco.mj_comPos(m, d)
                        err = target - (R.T @ (np.array(d.xpos[foot]) - thorax_pos))
                        if np.linalg.norm(err) < 1e-5:
                            break
                        mujoco.mj_jacBody(m, d, jacp, None, foot)
                        J = R.T @ jacp[:, dofs]          # body-frame Jacobian
                        Jpinv = J.T @ np.linalg.inv(
                            J @ J.T + (IK_DAMPING ** 2) * np.eye(3))
                        q = np.clip(q + Jpinv @ err, lo, hi)
                    resid = max(resid, float(np.linalg.norm(err)))
                    tab[si, k] = q - q0
            tables[leg] = (joints, tab)
            self.ik_residual = max(getattr(self, "ik_residual", 0.0), resid)
            self.ik_excursion = max(getattr(self, "ik_excursion", 0.0),
                                    float(np.abs(tab).max()))

        d.qpos[:] = qpos0
        mujoco.mj_forward(m, d)
        return tables

    # -- control -----------------------------------------------------------
    def _set(self, leg, joint, value):
        i = self.index.get(_act(leg, joint))
        if i is not None:
            self.ctrl[i] = value

    def _neutral_of(self, leg, joint):
        i = self.index.get(_act(leg, joint))
        return self.neutral[i] if i is not None else 0.0

    def step(self, dt_s: float, drive: float, turn: float):
        """Advance the rhythm and write joint targets.

        drive: 0..1, how fast to walk.
        turn:  -1..1, positive turns left. Implemented as a stride-length
               asymmetry between the two sides, which is how an insect turns -
               the inside legs take shorter steps.
        """
        drive = float(np.clip(drive, 0.0, 1.0))
        turn = float(np.clip(turn, -1.0, 1.0))

        # A fly that is not walking keeps all six feet down. Letting the
        # adhesion keep cycling at zero drive made it drift several mm/s and
        # spin on the spot, which read as locomotion that the brain had not
        # commanded.
        if drive <= 0.02:
            for leg in LEGS:
                joints, tab = self.table[leg]
                for j, delta in zip(joints, tab[-1, N_PHASE]):
                    self.ctrl[j[0]] = self.neutral[j[0]] + delta
                self._adhesion[leg] = 1.0
                self._set(leg, "adhesion", 1.0)
            return self.ctrl

        # Drive sets the step RATE and the stride stays full length. Speed is
        # stride x frequency either way, and a long stride keeps the feet clearing
        # the ground and landing properly even when the fly is walking slowly.
        freq = MIN_FREQ_HZ + (BASE_FREQ_HZ - MIN_FREQ_HZ) * drive
        self.phase = (self.phase + freq * dt_s) % 1.0

        for leg in LEGS:
            p = (self.phase + TRIPOD[leg]) % 1.0
            joints, tab = self.table[leg]
            dq = _blend(tab, int(p * N_PHASE) % N_PHASE, turn)
            for j, delta in zip(joints, dq):
                self.ctrl[j[0]] = self.neutral[j[0]] + delta

            # a foot in swing must let go, or the fly drags itself
            self._adhesion[leg] = 0.0 if in_swing(p) else 1.0
            self._set(leg, "adhesion", self._adhesion[leg])

        return self.ctrl

    def apply(self):
        self.sim.mj_data.ctrl[:] = self.ctrl
