# Fly connectome explorer — local setup (Ryan)

Two whole-CNS connectomes running locally in one Codex app, plus a DuckDB layer
that can query both at once.

| Dataset | Version | Neurons | Synapses | Coverage |
|---|---|---|---|---|
| **FlyWire FAFB** | 783 (Oct 2023) | 139,255 | 34.2 M | female brain + both optic lobes, no VNC |
| **Janelia MaleCNS** | v1.0 (Jun 2026) | 166,700 | 124.2 M | male brain + optic lobes + **VNC** |

## Run the app

```sh
./run.sh                 # or double-click start-codex.cmd -> http://127.0.0.1:5000
```

FAFB is the default. Switch datasets with the version picker, or append
`&data_version=malecns` to any URL. First MaleCNS request loads a 143 MB pickle
(~15 s); both datasets resident is ~6.9 GB RAM.

Working on both: search, cell info, connectivity, pathways, heatmaps, neuropils,
stats, motifs, NBLAST "similar cells", 3D view. Each dataset's 3D links point at
its own volume — FAFB at FlyWire, MaleCNS at Janelia's.

## Run it as a living animal

`sim/` wires the MaleCNS connectome up as a spiking brain, gives it a metabolism,
and puts it in a 3D arena with five request pads it has to use to stay alive.

```sh
./.venv/Scripts/python.exe sim/build_brain.py          # once: connectome -> runnable substrate
./.venv/Scripts/python.exe sim/run.py                  # 3D viewer on http://127.0.0.1:5050
./.venv/Scripts/python.exe sim/run.py --headless 300    # measure instead of watch
./.venv/Scripts/python.exe sim/probe.py                # sensor -> motor transfer characterisation
```

All 166,700 neurons and 25,582,938 synapses run at 1 ms resolution, event driven,
at roughly a third of realtime. `sim/README.md` is the reference, and it separates
what is real connectome data from what is my model layer - read that before
drawing any conclusion from what the fly does.

## Query it

```sh
./.venv/Scripts/python.exe lab/q.py                            # FAFB REPL
./.venv/Scripts/python.exe lab/q.py malecns                    # MaleCNS REPL
./.venv/Scripts/python.exe lab/q.py "select * from cells limit 5"
./.venv/Scripts/python.exe lab/q.py malecns "select ..."
```

**Both connectomes are attached in every session.** Whichever you open, the other
is available under its name, so one query can span both:

```sql
-- NBLAST crosswalk: does morphology agree with cell typing across two flies?
SELECT m.primary_type AS mcns_type, f.primary_type AS flywire_type,
       count(*) n, round(avg(x.score), 3) avg_nblast
FROM mcns_to_flywire x
JOIN cells m      ON m.bodyId  = x.id
JOIN fafb.cells f ON f.root_id = x.match
WHERE x.rank = 1 AND m.primary_type IS NOT NULL AND f.primary_type IS NOT NULL
GROUP BY 1, 2 ORDER BY n DESC;
```

### FAFB tables (`lab/flywire_783.duckdb`, 296 MB)

```
neurons             139,255    root_id, group, nt_type + 6 neurotransmitter scores
classification      139,255    flow, super_class, class, sub_class, hemilineage, side, nerve
cell_types          138,327    primary_type, additional_type(s)
cell_stats          139,246    length_nm, area_nm, size_nm
connections       3,869,878    pre_root_id, post_root_id, neuropil, syn_count, nt_type
labels              160,045    community annotations w/ author + date
coordinates         238,909    representative XYZ positions
nblast              133,682    morphological-similarity scores
connectivity_tags   134,437
```

Views: **`cells`** (one row per neuron, everything joined), **`edges`** (one row
per connection with pre/post type, side, super_class).

### MaleCNS tables (`lab/malecns_v10.duckdb`, 2.4 GB)

```
annotations         211,577    36 cols: type, instance, class/subclass/superclass, somaSide,
                               statusLabel, hemilineages (Ito-Lee + Truman), dimorphism,
                               receptorType, nerves, synonyms, and matched type names in
                               FlyWire / hemibrain / MANC
neurotransmitters 1,835,518    per-body + per-cell-type NT predictions, consensus_nt
body_stats          191,696    pre, post, downstream, synweight, rank
weights         151,856,684    the FULL segmentation graph, 311.8 M synapses
connections      25,582,938    neuron-to-neuron only, 124,177,617 synapses
edge_rois        29,456,389    the same edges split by brain region (pre, post, neuropil, syn)
rois                    143    the region vocabulary, by synapse count
soma_locations      211,577    soma_x / soma_y / soma_z
neuron_ids          166,700
contra_matches    1,666,990    top-5 NBLAST matches vs the left/right-flipped dataset
mcns_to_flywire     722,560    top-5 NBLAST matches into FlyWire 783
flywire_to_mcns     696,365    top-5 NBLAST matches the other way
```

Views: **`cells`** (one row per neuron: annotations + NT + stats + soma position),
**`edges`** (connections + pre/post type, superclass, side, NT).

### Starter queries

```sql
-- FAFB: biggest hubs by outgoing synapses
SELECT c.root_id, c.primary_type, c.super_class, c.side, sum(e.syn_count) AS out_syn
FROM cells c JOIN connections e ON e.pre_root_id = c.root_id
GROUP BY 1,2,3,4 ORDER BY out_syn DESC LIMIT 20;

-- MaleCNS: biggest hubs (synweight is pre+post, precomputed)
SELECT bodyId, primary_type, superclass, somaSide, synweight
FROM cells ORDER BY synweight DESC NULLS LAST LIMIT 20;

-- MaleCNS: what the VNC sends up to the brain, and where it lands
SELECT er.neuropil, sum(er.syn_count) syn, count(DISTINCT er.post_body) targets
FROM edge_rois er JOIN cells c ON c.bodyId = er.pre_body
WHERE c.superclass = 'ascending_neuron'
GROUP BY 1 ORDER BY syn DESC LIMIT 20;

-- MaleCNS: sexually dimorphic cell types and who they talk to
SELECT c.primary_type, c.dimorphism, count(DISTINCT e.post_body) partners,
       sum(e.syn_count) syn
FROM cells c JOIN edges e ON e.pre_body = c.bodyId
WHERE c.dimorphism IS NOT NULL
GROUP BY 1,2 ORDER BY syn DESC LIMIT 25;

-- MaleCNS: per-region neurotransmitter budget
SELECT er.neuropil, n.consensus_nt, sum(er.syn_count) syn
FROM edge_rois er LEFT JOIN neurotransmitters n ON n.body = er.pre_body
GROUP BY 1,2 ORDER BY er.neuropil, syn DESC;

-- MaleCNS: each neuron's contralateral homolog (NBLAST rank 1)
SELECT c.primary_type, c.bodyId, x.match AS contra_body, round(x.score,3) score
FROM contra_matches x JOIN cells c ON c.bodyId = x.id
WHERE x.rank = 1 AND x.score > 0.7 ORDER BY x.score DESC LIMIT 20;
```

## Rebuilding

```sh
# FAFB
./.venv/Scripts/python.exe -m codex.data.local_data_loader
./.venv/Scripts/python.exe lab/build_db.py

# MaleCNS — in this order, each depends on the last
./.venv/Scripts/python.exe lab/build_malecns_db.py        # Feather -> DuckDB   (~60 s)
./.venv/Scripts/python.exe lab/build_malecns_rois.py      # per-edge regions    (~4 min)
./.venv/Scripts/python.exe lab/build_malecns_matches.py   # NBLAST match tables (~10 s)
./.venv/Scripts/python.exe lab/gen_malecns_regions.py     # region table for Codex
./.venv/Scripts/python.exe lab/export_malecns_to_codex.py 3    # 3 = min synapses
./.venv/Scripts/python.exe -c "from codex.data.local_data_loader import load_and_pickle_neuron_db_versions as f; f(versions=['malecns'])"   # ~6 min, peaks ~15 GB
```

`lab/probe_schema.py <url>` reads any Feather file's schema over HTTP without
downloading it — handy before pulling a multi-GB file.

### Source files

All CC-BY 4.0, no credentials. MaleCNS from
`https://storage.googleapis.com/flyem-male-cns/v1.0/`:

```
connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather   14 MB
connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather         43 MB
connectome-data/flat-connectome/body-stats-male-cns-v1.0-minconf-0.5.feather        778 MB
connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather 1.1 GB
connectome-data/flat-connectome/syn-partners-male-cns-v1.0-minconf-0.5.feather       6.8 GB
nblasts/matches_mcns_v1.0_contra.feather                                             15 MB
nblasts/matches_mcns_v1.0_flywire783.feather                                        7.2 MB
nblasts/matches_flywire783_mcns_v1.0.feather                                        6.4 MB
```

Not downloaded: `syn-points` (13 GB — per-synapse XYZ, nothing here needs it) and
the full NBLAST matrices (18–21 GB each; they are 166k columns wide, so ~110 GB
in memory. The top-5 match tables above carry the usable part).

## Why the numbers are what they are

**166,700 neurons.** MaleCNS defines a neuron as a body with a `superclass`
assigned — that is the project's own criterion, stated in the NBLAST README at
`gs://flyem-male-cns/v1.0/nblasts/`, and it lands within 9 bodies of the
published 166,691. Status-based cuts do not: `Traced` alone is 165,122 and
`Traced+Anchor+Assign` is 167,565. The superclass cut also yields **124,177,617**
neuron-to-neuron synapses against the paper's ~125 M, so both headline figures
agree.

**Per-edge brain regions.** The flat connectome has no ROI column, but
`syn-partners` carries `primary_post` — the neuropil each postsynaptic site sits
in. Aggregating it gives `edge_rois`, which accounts for all 124,177,617
synapses across 143 regions with nothing unassigned.

**UI graph thresholded to ≥3 synapses** (10.5 M of 25.6 M edges, **84% of all
synapses**, 14.0 M per-region rows). Codex holds the connectome as nested Python
dicts and sets, so this is a memory ceiling, not a preference: at ≥3 the index
build peaks at 15.3 GB and serving both datasets sits at 6.9 GB. The complete
25.6 M-edge graph — and the full 151.9 M-edge segmentation graph — stay in
DuckDB. Re-export with a different threshold any time.

## MaleCNS caveats

1. **No per-neuron size.** MaleCNS publishes no cable length, surface area or
   volume, so Codex's "Size" panel is empty. `body_stats` has synapse counts
   (`pre`, `post`, `downstream`, `synweight`) instead.
2. **MaleCNS regions draw no 3D neuropil mesh.** They carry negative segment ids,
   which is Codex's existing convention for "no mesh", so the neuropil mesh layer
   skips them rather than drawing the wrong fly's anatomy. Cell meshes themselves
   are fine — those come from Janelia's own segmentation.
3. **Seven region names are shared with FAFB** (EB, FB, GNG, NO, PB, PRW, SAD) and
   reuse the FAFB entry, since they name the same structure. They do get a 3D
   mesh — the FlyWire one.
4. **"Similar cells" shows contralateral homologs**, not same-side neighbours.
   MaleCNS publishes top-5 matches only against the mirrored dataset; the
   same-side all-by-all exists solely as a 166k-wide matrix that will not fit in
   memory. The contralateral comparison is the more useful one anyway.
5. **No community labels.** The labels Codex shows are synthesised from MaleCNS's
   own `instance`, `synonyms`, and matched type names in FlyWire / hemibrain /
   MANC — all searchable, none crowd-sourced.

## Local patches vs upstream

Nine files. A `git pull` from upstream reintroduces every one of these.

- `codex/data/local_data_loader.py` — force `encoding="utf-8"` on text reads.
  Upstream uses the platform default, which is cp1252 on Windows and crashes on
  `labels.csv.gz`.
- `codex/data/neurotransmitters.py` — added `HA` (histamine) and `UNK`. Codex
  hard-asserts every connection's NT is in this dict; MaleCNS predicts histamine
  for ~6k cells and leaves others unresolved. Inert for FAFB.
- `codex/service/cell_details.py` — skip NT types with no `*_avg` column (so the
  two additions above don't `KeyError` the cell page), and convert coordinates
  per dataset.
- `codex/utils/formatting.py` — `nanometer_to_dataset_coordinates()`. MaleCNS is
  8 nm isotropic; FAFB is 4/4/40 nm.
- `codex/utils/nglui.py` — a MaleCNS Neuroglancer state (EM, segmentation, brain
  and VNC shells) built from Janelia's own published layer sources. Without it,
  MaleCNS body ids were being pointed at the FlyWire volume, which renders the
  wrong brain.
- `codex/data/brain_regions.py` — merge the MaleCNS region table; recognise
  `(L)`/`(R)` as side suffixes alongside FAFB's `_L`/`_R`.
- `codex/data/malecns_regions.py` — **generated** by `lab/gen_malecns_regions.py`
  (idempotent; re-running is safe).
- `codex/data/malecns_roi_naming.py` — new. Codex uppercases region names, which
  would collapse the mushroom body alpha lobe `aL` onto the antennal lobe `AL`;
  this is the one place that policy lives.
- `codex/blueprints/app.py` — an explicit `root_id` now resolves to that cell
  instead of running a free-text search for its digits. A bare id could otherwise
  match another cell's label as a substring (MaleCNS synonym `fru-M-100094`
  contains `10009`) and bounce you to the results page.
