"""Round-trip tests for the dataset serialization (ltel_box.dataset_io)."""

from ltel_box.dataset_io import (
    gci_from_dict, gci_to_dict, load_dataset_dir, save_dataset_dir,
    tbox_from_dict, tbox_to_dict,
)
from ltel_box.ontology import (
    NF1, NF2, NF3, NF4, NF5, NF6, NF7, NF10, NF12, NF13,
)
from ltel_box.oracle import temporal_closure
from ltel_box.synthetic import SyntheticConfig, generate_tbox

ALL_NFS = [
    NF1("A", "B"), NF2("A", "B", "C"), NF3("A", "r", "B"), NF4("r", "A", "B"),
    NF5("r", "s", "t"), NF6("A", "B"), NF7("A", "B"), NF10("A", "B"),
    NF12("A", "B"), NF13("A", "B", "C"),
]


def test_gci_round_trip_all_normal_forms():
    for g in ALL_NFS:
        assert gci_from_dict(gci_to_dict(g)) == g


def test_tbox_round_trip():
    tbox = generate_tbox(SyntheticConfig(n_concepts=10, seed=7)).tbox
    back = tbox_from_dict(tbox_to_dict(tbox))
    assert back.concept_names == tbox.concept_names
    assert back.role_names == tbox.role_names
    assert back.nf_gcis == tbox.nf_gcis


def test_dataset_dir_round_trip(tmp_path):
    ds = generate_tbox(SyntheticConfig(n_concepts=10, seed=3))
    closure = temporal_closure(ds.tbox, k=2, loop=0)
    abox = {"individuals": ["i0"], "memberships": [[0, 1, [0, 2]]]}
    save_dataset_dir(tmp_path / "d", tbox=ds.tbox, closure=closure,
                     config={"family": "test", "seed": 3}, abox=abox)
    back = load_dataset_dir(tmp_path / "d")
    assert back["config"] == {"family": "test", "seed": 3}
    assert back["tbox"].nf_gcis == ds.tbox.nf_gcis
    assert back["closure"] == closure
    assert back["abox"] == abox


def test_dataset_dir_without_abox(tmp_path):
    ds = generate_tbox(SyntheticConfig(n_concepts=6, seed=0))
    closure = temporal_closure(ds.tbox, k=1, loop=0)
    save_dataset_dir(tmp_path / "d", tbox=ds.tbox, closure=closure,
                     config={"family": "test"})
    back = load_dataset_dir(tmp_path / "d")
    assert "abox" not in back
    assert back["closure"] == closure
