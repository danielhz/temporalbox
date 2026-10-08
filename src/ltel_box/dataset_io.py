"""
Textual (JSON) serialization of the synthetic experiment datasets.

The run drivers regenerate every synthetic TBox from (config, seed); this
module gives the datasets a durable on-disk form for the DaRUS deposit: one
directory per dataset holding

  config.json   generator config + oracle parameters + provenance
  tbox.json     the normal-form LTEL TBox (concepts, roles, typed GCI records)
  closure.json  the oracle ground-truth temporal closure, sorted [sub, sup, t]
  abox.json     (RQ2-family only) individuals, memberships, pos/neg triples,
                rigid concepts, support levels; concept ids follow
                Vocabulary.from_tbox order (⊤ = 0, then sorted names)

`scripts/materialize_datasets.py` writes these directories from the archived
run records; the load functions below read them back.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .ontology import (
    NF1, NF2, NF3, NF4, NF5, NF6, NF7, NF10, NF12, NF13,
    NormalFormGCI, NormalFormTBox,
)

_NF_TYPES: dict[str, type] = {
    t.__name__: t for t in (NF1, NF2, NF3, NF4, NF5, NF6, NF7, NF10, NF12, NF13)
}


def gci_to_dict(gci: NormalFormGCI) -> dict:
    return {"nf": type(gci).__name__, **dataclasses.asdict(gci)}


def gci_from_dict(d: dict) -> NormalFormGCI:
    d = dict(d)
    cls = _NF_TYPES[d.pop("nf")]
    return cls(**d)


def tbox_to_dict(tbox: NormalFormTBox) -> dict:
    return {
        "concept_names": sorted(tbox.concept_names),
        "role_names": sorted(tbox.role_names),
        "gcis": [gci_to_dict(g) for g in tbox.nf_gcis],
    }


def tbox_from_dict(d: dict) -> NormalFormTBox:
    return NormalFormTBox(
        concept_names=frozenset(d["concept_names"]),
        role_names=frozenset(d["role_names"]),
        nf_gcis=[gci_from_dict(g) for g in d["gcis"]],
    )


def save_dataset_dir(
    outdir: str | Path,
    *,
    tbox: NormalFormTBox,
    closure: frozenset[tuple[str, str, int]],
    config: dict,
    abox: dict | None = None,
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _dump(outdir / "config.json", config)
    _dump(outdir / "tbox.json", tbox_to_dict(tbox))
    _dump(outdir / "closure.json", sorted([a, b, t] for a, b, t in closure))
    if abox is not None:
        _dump(outdir / "abox.json", abox)
    return outdir


def load_dataset_dir(indir: str | Path) -> dict:
    """Read a dataset directory back: config, tbox, closure (+ abox if present)."""
    indir = Path(indir)
    out = {
        "config": json.loads((indir / "config.json").read_text()),
        "tbox": tbox_from_dict(json.loads((indir / "tbox.json").read_text())),
        "closure": frozenset(
            (a, b, t) for a, b, t in json.loads((indir / "closure.json").read_text())
        ),
    }
    abox_path = indir / "abox.json"
    if abox_path.exists():
        out["abox"] = json.loads(abox_path.read_text())
    return out


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
