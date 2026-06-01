"""TDD: local-file mirror for IP names in the observability viewer.

Names are primarily stored in the shared GCS config (`ip_labels`); the user also
wants a local, hand-editable mirror (`viewer/ip_names.yaml`) — "Both". These
cover the pure file + merge helpers (no GCS, no Flask). The GCS config stays
authoritative; on a conflict at startup, the local FILE wins (so a hand-edit
applies), and every rename writes the file through so it always mirrors GCS.
"""
from __future__ import annotations

import viewer.obs_server as obs


# ── labels_to_name_map: GCS ip_labels {ip:{name,..}} -> flat {ip:name} ──────────
def test_labels_to_name_map_extracts_names():
    labels = {
        "1.1.1.1": {"name": "roee"},
        "2.2.2.2": {"name": "dor", "notes": "teammate"},
        "3.3.3.3": {},          # no name -> skipped
        "4.4.4.4": {"name": ""},  # blank -> skipped
    }
    assert obs.labels_to_name_map(labels) == {"1.1.1.1": "roee", "2.2.2.2": "dor"}


# ── load_local_names ────────────────────────────────────────────────────────────
def test_load_local_names_missing_file_is_empty(tmp_path):
    assert obs.load_local_names(tmp_path / "nope.yaml") == {}


def test_load_local_names_malformed_is_empty(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("{unclosed: [oops")
    assert obs.load_local_names(p) == {}


def test_load_local_names_keeps_only_nonempty_strings(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("1.1.1.1: roee\n2.2.2.2:\n3.3.3.3: 123\n")
    got = obs.load_local_names(p)
    assert got.get("1.1.1.1") == "roee"
    assert "2.2.2.2" not in got      # null value dropped
    assert "3.3.3.3" not in got      # non-string dropped


# ── dump + load round trip ──────────────────────────────────────────────────────
def test_dump_then_load_round_trip(tmp_path):
    p = tmp_path / "ip_names.yaml"
    obs.dump_local_names({"9.9.9.9": {"name": "roee"}, "8.8.8.8": {"name": "dor"}}, p)
    assert obs.load_local_names(p) == {"9.9.9.9": "roee", "8.8.8.8": "dor"}


def test_dump_overwrites_atomically(tmp_path):
    p = tmp_path / "ip_names.yaml"
    obs.dump_local_names({"1.1.1.1": {"name": "old"}}, p)
    obs.dump_local_names({"1.1.1.1": {"name": "new"}, "2.2.2.2": {"name": "dor"}}, p)
    assert obs.load_local_names(p) == {"1.1.1.1": "new", "2.2.2.2": "dor"}


# ── compute_label_merge: file wins, only push deltas ────────────────────────────
def test_compute_label_merge_adds_new_only():
    local = {"a": "roee", "b": "dor", "c": "same"}
    gcs = {"a": {"name": "roee"}, "c": {"name": "same"}, "d": {"name": "other"}}
    # a,c already match -> skip; b is new -> push; d not in file -> untouched.
    assert obs.compute_label_merge(local, gcs) == {"b": "dor"}


def test_compute_label_merge_file_overrides_differing_name():
    assert obs.compute_label_merge({"a": "NEW"}, {"a": {"name": "OLD"}}) == {"a": "NEW"}


def test_compute_label_merge_empty_local_pushes_nothing():
    assert obs.compute_label_merge({}, {"a": {"name": "x"}}) == {}


# ── merge_local_into_config: orchestration (file→GCS, then mirror back) ──────────
def test_merge_local_into_config_pushes_only_deltas_and_mirrors_back(monkeypatch, tmp_path):
    p = tmp_path / "ip_names.yaml"
    obs.dump_local_names({"a": {"name": "roee"}, "b": {"name": "dor"}}, p)

    state = {"ip_labels": {"a": {"name": "roee"}}}  # GCS already knows 'a'

    def fake_read_config(client):
        return ({"version": 1, "ip_labels": dict(state["ip_labels"]), "settings": {}}, 1)

    sets = []

    def fake_set_label(client, ip, name, notes=None):
        sets.append((ip, name))
        state["ip_labels"][ip] = {"name": name}
        return {"version": 1, "ip_labels": dict(state["ip_labels"]), "settings": {}}

    mirrored = {}
    monkeypatch.setattr(obs, "read_config", fake_read_config)
    monkeypatch.setattr(obs, "set_label", fake_set_label)
    monkeypatch.setattr(obs, "dump_local_names",
                        lambda labels, path=None: mirrored.update(obs.labels_to_name_map(labels)))

    obs.merge_local_into_config(client=object(), path=p)

    assert sets == [("b", "dor")]            # only the new IP pushed to GCS
    assert mirrored == {"a": "roee", "b": "dor"}  # file mirrors the union afterwards
