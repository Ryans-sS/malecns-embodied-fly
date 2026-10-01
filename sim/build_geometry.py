"""Export real soma positions so the brain can be drawn, not just tabulated.

MaleCNS gives a 3D soma location for 139,662 of the 166,700 neurons, in 8 nm
voxels spanning the whole central nervous system - brain, optic lobes and ventral
nerve cord, about 1.08 mm end to end. This writes them out aligned to the same
neuron ordering the simulation uses, so a live activity value can be looked up per
point with no join at render time.

  python sim/build_geometry.py     ->  sim/brain_geometry.npz
"""
import os, sys
import numpy as np
import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

NPZ = os.path.join("sim", "brain_malecns.npz")
DB = os.path.join("lab", "malecns_v10.duckdb")
OUT = os.path.join("sim", "brain_geometry.npz")

# Group codes, used for colour. Functional identity wins over gross region, so a
# Kenyon cell reads as mushroom body rather than "central brain".
GROUPS = [
    "optic", "central", "vnc", "sensory", "motor",
    "mushroom_body", "projection", "other",
]

z = np.load(NPZ, allow_pickle=True)
ids = z["ids"]
N = len(ids)
pos_in_sim = {int(b): i for i, b in enumerate(ids)}

con = duckdb.connect(DB, read_only=True)
rows = con.execute("""
    SELECT bodyId, soma_x, soma_y, soma_z, superclass, "class"
    FROM cells
    WHERE soma_x IS NOT NULL AND soma_y IS NOT NULL AND soma_z IS NOT NULL
    ORDER BY bodyId
""").fetchall()
con.close()
print(f"{len(rows):,} neurons with a soma position (of {N:,})")

sim_idx, xyz, grp = [], [], []
for body, x, y, zz, sc, kl in rows:
    i = pos_in_sim.get(int(body))
    if i is None:
        continue
    sim_idx.append(i)
    xyz.append((x, y, zz))
    sc = sc or ""
    kl = kl or ""
    if kl in ("Kenyon_Cell", "MBON", "DAN"):
        g = "mushroom_body"
    elif sc.endswith("_motor") or sc.endswith("_efferent"):
        g = "motor"
    elif "sensory" in sc:
        g = "sensory"
    elif sc in ("descending_neuron", "ascending_neuron", "visual_projection",
                "visual_centrifugal", "sensory_ascending"):
        g = "projection"
    elif sc.startswith("ol_"):
        g = "optic"
    elif sc.startswith("cb_"):
        g = "central"
    elif sc.startswith("vnc_"):
        g = "vnc"
    else:
        g = "other"
    grp.append(GROUPS.index(g))

sim_idx = np.array(sim_idx, dtype=np.int32)
xyz = np.array(xyz, dtype=np.float32)
grp = np.array(grp, dtype=np.uint8)

# Centre and scale into a tidy box. MaleCNS voxels are 8 nm; the anterior-posterior
# axis (z) is the long one, so scale all axes by the same factor to keep the shape.
centre = xyz.mean(axis=0)
xyz -= centre
scale = 40.0 / np.abs(xyz).max()
xyz *= scale
# Neuroglancer's y axis points down relative to a natural "up" in a 3D viewer.
xyz[:, 1] *= -1.0

print(f"drawable points: {len(sim_idx):,}")
for g in range(len(GROUPS)):
    n = int((grp == g).sum())
    if n:
        print(f"  {GROUPS[g]:14} {n:>7,}")
print(f"bounding box after scaling: "
      f"x {xyz[:,0].min():.1f}..{xyz[:,0].max():.1f}  "
      f"y {xyz[:,1].min():.1f}..{xyz[:,1].max():.1f}  "
      f"z {xyz[:,2].min():.1f}..{xyz[:,2].max():.1f}")

np.savez_compressed(OUT, sim_idx=sim_idx, xyz=xyz, group=grp,
                    groups=np.array(GROUPS, dtype=object))
print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
