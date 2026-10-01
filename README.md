# MaleCNS embodied fly

Two whole-CNS *Drosophila* connectomes running locally in one web explorer — and
the male one wired up as a spiking brain that drives a physical body.

This is a fork of [murthylab/codex](https://github.com/murthylab/codex), the
connectome explorer behind [codex.flywire.ai](https://codex.flywire.ai), with two
things added:

1. **A second connectome.** Janelia's MaleCNS v1.0 (male brain + optic lobes +
   ventral nerve cord) runs alongside FlyWire FAFB 783 in the same app, switchable
   from the version picker, plus a DuckDB layer that can query both at once.
2. **`sim/` — the connectome as a living animal.** The 166,700-neuron MaleCNS
   network runs as a leaky integrate-and-fire brain with a metabolism, inside an
   arena where it has to find and press request pads to stay alive. Its body is
   [NeuroMechFly](https://github.com/NeLy-EPFL/flygym) in MuJoCo: 66 actuated leg
   joints, tarsal adhesion, and 721 ommatidia per eye.

| Dataset | Version | Neurons | Synapses | Coverage |
|---|---|---|---|---|
| FlyWire FAFB | 783 (Oct 2023) | 139,255 | 34.2 M | female brain + both optic lobes, no VNC |
| Janelia MaleCNS | v1.0 (Jun 2026) | 166,700 | 124.2 M | male brain + optic lobes + **VNC** |

---

## What actually works

Every number below is from a scripted assay in this repo with a control arm, not
from a single demo run. The scripts that produce them are named.

**Chemotaxis** — does the fly find the odour source? (`sim/test_approach.py`,
6 trials per arm, odour ON vs OFF):

| | closest approach | time on pad | pad presses |
|---|---|---|---|
| odour ON | **0.06 mm** | **9.24 s** | **8** |
| odour OFF | 2.75 mm | 1.06 s | 1 |

**Associative learning** — does pairing an odour with reward depress *that
odour's* Kenyon-cell synapses more than another's? (`sim/test_selectivity.py`,
4 seeds, reward vs sham):

| | trained − control weight difference |
|---|---|
| rewarded | **−0.4226 ± 0.0201** |
| sham | 0.0000 ± 0.0000 |

**Walking** — does the NeuroMechFly body hold a tripod gait?
(`sim/nmf/gait.py`, measured over 3 s of walking):

| | |
|---|---|
| feet on the ground, walking | **3.24 of 6** (a tripod is 3) |
| feet on the ground, standing | **6.00 of 6** |
| walking speed | 10.5 mm/s (real flies: 10–20) |
| turn response | linear, −54.6 to +51.7 °/s |
| heading drift at turn = 0 | −0.8 °/s |

---

## Read this before believing anything the fly does

`sim/README.md` separates, line by line, what is **measured connectome data** from
what is **my model layer**. The short version:

- **Real:** every neuron, synapse, synapse count, neurotransmitter prediction,
  cell type and neuropil assignment. Which cells are Kenyon cells, which are
  descending neurons, which gustatory receptor neurons reach the proboscis motor
  pool — all from the dataset.
- **A model:** membrane dynamics, every gain and threshold, the metabolism, the
  learning rule, the arena, and the walking pattern generator. The connectome says
  what is wired to what; it does not say how any of it behaves.

Two examples of why that distinction matters, both of which cost real time here:

- MaleCNS types gustatory neurons *anatomically* and never records which tastant
  they respond to. Which pad counts as food was settled by driving each class and
  reading the 67 proboscis motor neurons (`sim/probe.py`) — taste-peg GRNs are the
  only class that excites feeding; pharyngeal GRNs suppress it.
- flygym ships no measured walking kinematics, so the gait here is a model. Its
  parameters are gait quantities (stride length, ground clearance, duty factor)
  solved onto the real body through the model's own Jacobian, not invented joint
  angles — but it is still a model, not fly motion capture.

---

## Layout

```
codex/      upstream explorer, with MaleCNS support added (8 files changed, 2 added)
lab/        DuckDB build + query layer over both connectomes
sim/        the embodied simulation
  brain.py      166,700-neuron LIF network, mushroom-body plasticity, steering
  body.py       metabolism: energy, water, sodium, damage, cell death
  world.py      arena, request pads, sensorimotor coupling
  nmf/          NeuroMechFly body: gait.py, world.py, serve.py + web UI
  probe*.py     sensor -> motor characterisation
  diag_*.py     diagnostics for each subsystem
  test_*.py     the controlled assays behind the numbers above
LOCAL_SETUP.md   full local setup notes, both datasets
```

## Setup

Python 3.11+ and ~16 GB RAM (both datasets resident is ~6.9 GB).

The explorer and the simulation have separate dependency sets:

```sh
poetry install                                    # the explorer (upstream)

python -m venv .venv                              # the simulation
./.venv/Scripts/pip install -r requirements-sim.txt
```

`flygym==2.1.0` is not interchangeable with other versions — see the note in
`requirements-sim.txt`.

### Data

**The datasets are not in this repo** — they are ~11 GB, and the largest single
file is 6.4 GB against GitHub's 100 MB limit. Fetch them from source:

- **FlyWire FAFB 783** — `./scripts/make_data.sh` (upstream's downloader), or from
  [codex.flywire.ai/api/download](https://codex.flywire.ai/api/download).
  Use requires agreeing to the FlyWire citation guidelines.
- **Janelia MaleCNS v1.0** — from [neuprint.janelia.org](https://neuprint.janelia.org),
  into `static/data/malecns_raw/`. CC-BY 4.0.

Then build the derived artifacts, in this order:

```sh
python lab/build_db.py                 # FAFB      -> lab/flywire_783.duckdb
python lab/build_malecns_db.py         # MaleCNS   -> lab/malecns_v10.duckdb
python lab/build_malecns_rois.py
python lab/gen_malecns_regions.py      # -> codex/data/malecns_regions.py
python lab/export_malecns_to_codex.py  # -> static/data/malecns/
python sim/build_brain.py              # -> sim/brain_malecns.npz  (~45 MB)
python sim/build_geometry.py           # -> sim/brain_geometry.npz
python sim/probe_steering.py           # -> sim/steer_readout.npz  (committed)
```

### Run

```sh
./run.sh                               # explorer      http://127.0.0.1:5000
python sim/run.py                      # arena + brain http://127.0.0.1:5050
python sim/nmf/serve.py --port 5050    # NeuroMechFly  http://127.0.0.1:5050
python sim/run.py --headless 300 --csv out.csv
```

In the explorer, FAFB is the default; append `&data_version=malecns` to any URL
or use the version picker. Each dataset's 3D links point at its own volume —
FAFB at FlyWire, MaleCNS at Janelia's.

---

## Credits

This work stands on four things, none of them mine:

- **[Codex](https://github.com/murthylab/codex)** — Murthy and Seung labs,
  Princeton. The explorer this forks. Apache-2.0.
- **[FlyWire](https://flywire.ai)** — FAFB whole-brain connectome, 783.
  Please follow the FlyWire [citation guidelines](https://codex.flywire.ai/about_flywire)
  if you use it.
- **[Janelia MaleCNS](https://neuprint.janelia.org)** — male CNS connectome v1.0,
  HHMI Janelia Research Campus. CC-BY 4.0.
- **[NeuroMechFly / flygym](https://github.com/NeLy-EPFL/flygym)** — NeLy, EPFL.
  The biomechanical body and its retina.

The leaky integrate-and-fire parameterisation follows
[Shiu et al., *Nature* 2024](https://doi.org/10.1038/s41586-024-07763-9).

## Licence

Apache-2.0, inherited from upstream Codex — see [LICENSE](LICENSE) and
[NOTICE](NOTICE). The connectome datasets are **not** covered by it and carry
their own terms, linked above.
