"""The physical fly and the dish it lives in, built on NeuroMechFly / MuJoCo.

This replaces the hand-rolled arena and muscle model. What it buys:

  * 66 actuated leg degrees of freedom with real contact physics and tarsal
    adhesion, instead of `speed = k * firing_rate` and feet that slid;
  * 126 live joint angles, so proprioception is a real signal rather than a
    constant injected into the proprioceptive neurons;
  * 2 x 721 ommatidia, so the 6,091 visual sensory neurons finally see something.

flygym 2.1.0 is a composable API: a NeuroMechFly starts bare and you add joints,
actuators, adhesion, vision and cameras explicitly. Note also that flygym's own
ground-contact sensors fail to compile in this version (`unrecognized name
'fly/lf_coxa'` under every ContactBodiesPreset), so they are disabled and contact
is read from MuJoCo directly.
"""
from __future__ import annotations

import math

import numpy as np
from flygym.anatomy import ActuatedDOFPreset, AxisOrder, JointPreset, Skeleton
from flygym.compose import (
    ActuatorType,
    FlatGroundWorld,
    KinematicPosePreset,
    NeuroMechFly,
)
from flygym.utils.math import Rotation3D

# A fly is about 2.5 mm long. The old arena was 30 mm across; this dish is 50 mm
# in radius, roughly 40 body lengths, so there is somewhere to actually go.
DISH_R = 50.0
WALL_H = 4.0
# 36 rather than 72. The wall is a ring of separate box geoms, and every one of
# them is scene-setup work on each rendered frame - which is where half the
# server's wall clock goes. At a 50 mm radius the ring still reads as smooth.
WALL_SEGMENTS = 36
PAD_R = 6.0
PAD_RING = 0.62          # pads sit at this fraction of the dish radius

PADS = [
    ("SUGAR", "sucrose", (0.85, 0.55, 0.04, 1.0)),
    ("WATER", "water", (0.12, 0.56, 0.81, 1.0)),
    ("SALT", "salt", (0.48, 0.36, 0.82, 1.0)),
    ("BITTER", "bitter", (0.75, 0.22, 0.17, 1.0)),
    ("BLANK", "none", (0.53, 0.58, 0.64, 1.0)),
]


def build_fly(name: str = "fly", with_vision: bool = True) -> NeuroMechFly:
    """A fully articulated NeuroMechFly: legs actuated, adhesive tarsi, eyes."""
    fly = NeuroMechFly(name=name)
    skeleton = Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL,
        joint_preset=JointPreset.ALL_BIOLOGICAL,
    )
    fly.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
    fly.add_actuators(
        list(skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ONLY)),
        ActuatorType.POSITION,
        neutral_input=KinematicPosePreset.NEUTRAL,
    )
    # Tarsal adhesion is how a real fly keeps its feet; without it the model
    # skates on the plane.
    fly.add_leg_adhesion()
    if with_vision:
        fly.add_vision()
    fly.add_tracking_camera(name="trackcam")
    # Hovering behind and above the fly, looking the way it is walking.
    #
    # mode="fixed" rather than the default "track": "track" follows the fly's
    # POSITION but holds a fixed orientation in the world, so as the fly turned the
    # camera kept pointing the same way and you watched it side-on, with no view of
    # the ground ahead. "fixed" attaches the camera to the thorax so it yaws with
    # the fly and the view always looks along the heading.
    #
    # It also sat at [-9, 0, 7] with a 50 degree field of view, which put the 2.5 mm
    # fly 11.4 mm away inside a 10.6 mm tall view - a fifth of the frame height,
    # which read as a distant speck. The tilt is set so the ground under the gaze
    # centre is about 1.5 mm ahead of the fly: fly low in frame, path above it.
    fly.add_tracking_camera(
        name="hover",
        mode="fixed",
        pos_offset=np.array([-7.0, 0.0, 4.2]),
        rotation=Rotation3D("xyaxes", (0, -1, 0, 0.44, 0, 0.90)),
        fovy=48,
    )
    # Closer chase view, lower down, for watching the legs work.
    fly.add_tracking_camera(
        name="chase",
        mode="fixed",
        pos_offset=np.array([-4.6, 0.0, 1.7]),
        rotation=Rotation3D("xyaxes", (0, -1, 0, 0.26, 0, 0.97)),
        fovy=58,
    )
    return fly, skeleton


# The model renders bone-white, which vanishes against the grey floor. Tint the
# body so it reads at a distance without hiding the mesh detail.
FLY_COLORS = [
    ("tarsus", (0.16, 0.12, 0.09, 1.0)),
    ("tibia", (0.22, 0.16, 0.10, 1.0)),
    ("femur", (0.30, 0.21, 0.12, 1.0)),
    ("coxa", (0.38, 0.26, 0.14, 1.0)),
    # Barely there: seen from above, which is how the follow camera sees the fly,
    # the wings lie over the whole abdomen and at alpha 0.35 they washed the body
    # out to white against a grey floor - the thing the colouring was for.
    ("wing", (0.82, 0.88, 0.98, 0.13)),
    ("haltere", (0.85, 0.62, 0.20, 1.0)),
    ("eye", (0.78, 0.14, 0.10, 1.0)),
    ("antenna", (0.35, 0.24, 0.13, 1.0)),
    ("aristae", (0.30, 0.20, 0.11, 1.0)),
    ("head", (0.45, 0.30, 0.15, 1.0)),
    ("haustellum", (0.55, 0.40, 0.20, 1.0)),
    ("rostrum", (0.52, 0.37, 0.18, 1.0)),
    ("abdomen", (0.80, 0.49, 0.10, 1.0)),
    ("thorax", (0.62, 0.40, 0.16, 1.0)),
]


def fix_clipping(sim):
    """Stop the views slicing through the far side of the dish.

    Both clipping planes are FRACTIONS of the model extent, and this model reports
    an extent of 1.0, so MuJoCo's default zfar of 50 put the far plane 50 mm away -
    less than the 100 mm width of the dish. Anything beyond it simply vanished,
    which is the "seeing through things" in the eye view: the far wall was not being
    drawn. 400 covers the whole dish from any camera in it.

    znear is raised rather than lowered, to a little more than the fly's own head:
    the eye cameras sit on the head, so without this they fill with the fly's own
    eye and antennae instead of the world.
    """
    sim.mj_model.vis.map.znear = 0.3
    sim.mj_model.vis.map.zfar = 400.0
    extent = sim.mj_model.stat.extent
    return sim.mj_model.vis.map.znear * extent, sim.mj_model.vis.map.zfar * extent


# Keeping the fly inside the dish.
#
# Nothing on the fly collides with anything by the ordinary rules: its geoms ship
# with contype=0 and conaffinity=0, and - measured - this model produces NO
# dynamic contacts at all even with both masks forced to 1. Every contact it has,
# including the feet on the floor, comes from an explicit <pair>, which is how
# flygym wires the 55 ground contacts. So the wall could never stop the fly, and
# it simply walked out of the arena: measured at r = 65.6 mm, and once 97 mm, in
# a dish whose wall stands at 50 mm.
#
# The fix is to declare the pairs. Only the nine central body geoms are paired
# against the wall - thorax, head, abdomen and mouthparts - which is enough to
# stop the body passing through while keeping the pair count to 9 x 36 rather
# than one for all 69 geoms.
def add_wall_contacts(dish, fly_prefix="fly/"):
    """Declare fly-against-wall contact pairs. Call after add_fly, before compiling."""
    root = dish.mjcf_root
    names = [g.name for g in root.geoms if g.name]
    central = [n for n in names if n.startswith(fly_prefix + "c_")]
    walls = [n for n in names if n.startswith("wall_")]
    for w in walls:
        for c in central:
            pair = root.add_pair()
            pair.geomname1 = w
            pair.geomname2 = c
    return len(walls) * len(central)


def color_fly(sim, prefix="fly/"):
    """Tint the fly so it stands out from the floor."""
    m = sim.mj_model
    n = 0
    for i in range(m.ngeom):
        name = m.geom(i).name or ""
        if not name.startswith(prefix):
            continue
        low = name.lower()
        for key, rgba in FLY_COLORS:
            if key in low:
                m.geom_rgba[i] = rgba
                n += 1
                break
        else:
            m.geom_rgba[i] = (0.58, 0.38, 0.16, 1.0)
            n += 1
    return n


def pad_positions(n: int = len(PADS), radius: float = DISH_R * PAD_RING):
    out = []
    for k in range(n):
        ang = 2 * math.pi * k / n - math.pi / 2
        out.append((radius * math.cos(ang), radius * math.sin(ang)))
    return out


class Dish(FlatGroundWorld):
    """A walled circular dish with the five request pads set into the floor."""

    def __init__(self, name: str = "dish"):
        super().__init__(name=name, half_size=DISH_R * 3)
        self.pad_xy = pad_positions()
        self._add_wall()
        self._add_pads()
        self._add_lighting()
        self._add_cameras()

    def _add_wall(self):
        # MuJoCo has no hollow cylinder, so the wall is a ring of thin boxes.
        wb = self.mjcf_root.worldbody
        seg_w = math.pi * DISH_R / WALL_SEGMENTS * 1.15
        for i in range(WALL_SEGMENTS):
            a = 2 * math.pi * i / WALL_SEGMENTS
            wb.add_geom(
                name=f"wall_{i}",
                type=mj_box(),
                size=[0.6, seg_w, WALL_H / 2],
                pos=[DISH_R * math.cos(a), DISH_R * math.sin(a), WALL_H / 2],
                quat=yaw_quat(a),
                rgba=[0.26, 0.31, 0.38, 0.55],
            )

    def _add_pads(self):
        wb = self.mjcf_root.worldbody
        self.pad_geoms = {}
        for (name, _payload, rgba), (x, y) in zip(PADS, self.pad_xy):
            g = wb.add_geom(
                name=f"pad_{name}",
                type=mj_cylinder(),
                size=[PAD_R, 0.12, 0],
                pos=[x, y, 0.12],
                rgba=list(rgba),
                contype=0,          # decorative: the fly walks over it
                conaffinity=0,
            )
            self.pad_geoms[name] = g

    def _add_cameras(self):
        """Views of the whole dish, as opposed to the fly-tracking camera."""
        wb = self.mjcf_root.worldbody
        wb.add_camera(
            name="dish_top",
            pos=[0, 0, DISH_R * 1.75],
            quat=[1, 0, 0, 0],
            fovy=60,
        )
        wb.add_camera(
            name="dish_angled",
            pos=[0, -DISH_R * 1.25, DISH_R * 0.95],
            xyaxes=[1, 0, 0, 0, 0.62, 0.78],
            fovy=55,
        )

    def _add_lighting(self):
        wb = self.mjcf_root.worldbody
        wb.add_light(pos=[0, 0, 60], dir=[0, 0, -1], diffuse=[0.75, 0.75, 0.75],
                     specular=[0.1, 0.1, 0.1], castshadow=False)
        wb.add_light(pos=[40, -40, 40], dir=[-0.6, 0.6, -0.6],
                     diffuse=[0.35, 0.35, 0.4], castshadow=False)


def mj_box():
    import mujoco
    return mujoco.mjtGeom.mjGEOM_BOX


def mj_cylinder():
    import mujoco
    return mujoco.mjtGeom.mjGEOM_CYLINDER


def yaw_quat(a: float):
    """Quaternion for a rotation of `a` about z."""
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


def tune_actuators(sim, kp: float = 40.0, kv: float = 4.0):
    """Give the position actuators usable gains and an actual control range.

    As built, every leg actuator comes out with gainprm kp=1 and - fatally -
    ctrlrange [0, 0], so MuJoCo clamps every command to zero and the legs never
    move at all. kp=1 would also be far too soft against a joint stiffness of 10:
    the fly would sit on its abdomen rather than stand.
    """
    m = sim.mj_model
    n = 0
    for i in range(m.nu):
        name = m.actuator(i).name
        if name.endswith("-adhesion"):
            m.actuator_ctrlrange[i] = [0.0, 1.0]
            m.actuator_ctrllimited[i] = 1
            continue
        m.actuator_gainprm[i][0] = kp
        m.actuator_biasprm[i][1] = -kp
        m.actuator_biasprm[i][2] = -kv
        m.actuator_ctrlrange[i] = [-np.pi, np.pi]
        m.actuator_ctrllimited[i] = 0
        n += 1
    return n


def build_world(fly, spawn=(0.0, 0.0, 0.5)):
    dish = Dish()
    dish.add_fly(
        fly,
        spawn_position=np.array(spawn, dtype=float),
        spawn_rotation=Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)),
        # flygym's ground-contact sensors do not compile in 2.1.0
        add_ground_contact_sensors=False,
    )
    return dish
