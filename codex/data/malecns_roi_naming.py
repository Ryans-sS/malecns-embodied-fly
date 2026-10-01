"""How MaleCNS ROI names become Codex region keys.

Codex uppercases every neuropil name before looking it up in REGIONS, which
would collapse the mushroom body alpha lobe "aL" onto the antennal lobe "AL".
Prefixing the MB lobes keeps them distinct and reads better in the UI.

Hand-written on purpose: lab/gen_malecns_regions.py and
lab/export_malecns_to_codex.py both import this, so the naming policy has one
home and the generated region table can't drift from the exported CSV.
"""

ROI_RENAMES = {}
for _lobe in ("aL", "a'L", "bL", "b'L", "gL"):
    for _side in ("(L)", "(R)"):
        ROI_RENAMES[_lobe + _side] = "MB-" + _lobe + _side


def to_codex_region(raw_roi):
    """MaleCNS ROI name -> the key Codex stores."""
    return ROI_RENAMES.get(raw_roi, raw_roi).upper()
