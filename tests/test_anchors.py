"""Label anchors: CRUD, import/export, and anchor-override in the classifier."""
from __future__ import annotations

from pathlib import Path

import pytest

from alexandria.anchors import (
    delete_anchor,
    export_anchors,
    get_anchor,
    import_anchors,
    list_anchors,
    score_anchors,
    set_anchor,
)
from alexandria.classify import suggest_metadata
from alexandria.ingest import ingest_file


# ---- CRUD -----------------------------------------------------------------


def test_set_and_get_roundtrips(cfg, conn):
    a = set_anchor(conn, cfg, "category", "bills",
                   "Utility, subscription, or service billing statements.")
    assert a.kind == "category"
    assert a.name == "bills"
    got = get_anchor(conn, "category", "bills")
    assert got is not None
    assert got.description == a.description
    assert got.embed_model == cfg.embeddings.model


def test_set_upserts_and_preserves_created_at(cfg, conn):
    first = set_anchor(conn, cfg, "tag", "utility", "Utility service.")
    updated = set_anchor(conn, cfg, "tag", "utility", "Electric, gas, water, internet bills.")
    assert updated.created_at == first.created_at
    assert updated.updated_at >= first.updated_at
    assert updated.description == "Electric, gas, water, internet bills."


def test_set_rejects_bad_kind(cfg, conn):
    with pytest.raises(ValueError, match="kind must be"):
        set_anchor(conn, cfg, "topic", "physics", "…")


def test_set_rejects_blank_fields(cfg, conn):
    with pytest.raises(ValueError):
        set_anchor(conn, cfg, "tag", "  ", "desc")
    with pytest.raises(ValueError):
        set_anchor(conn, cfg, "tag", "physics", "  ")


def test_delete_and_list(cfg, conn):
    set_anchor(conn, cfg, "tag", "a", "aaa")
    set_anchor(conn, cfg, "tag", "b", "bbb")
    assert len(list_anchors(conn)) == 2
    assert delete_anchor(conn, "tag", "a") is True
    assert delete_anchor(conn, "tag", "a") is False  # idempotent
    remaining = list_anchors(conn)
    assert [(a.kind, a.name) for a in remaining] == [("tag", "b")]


def test_import_export_roundtrip(cfg, conn):
    src = [
        {"kind": "category", "name": "bills",
         "description": "Utility, subscription, service statements."},
        {"kind": "tag", "name": "physics",
         "description": "Physics research including quantum + classical."},
    ]
    r = import_anchors(conn, cfg, src)
    assert r == {"added": 2, "updated": 0, "errors": []}

    exported = export_anchors(conn)
    assert sorted(exported, key=lambda a: (a["kind"], a["name"])) == \
        sorted(src, key=lambda a: (a["kind"], a["name"]))

    # Re-import → all updates, no adds.
    r2 = import_anchors(conn, cfg, src)
    assert r2["added"] == 0 and r2["updated"] == 2


def test_import_collects_errors(cfg, conn):
    r = import_anchors(conn, cfg, [
        {"kind": "category", "name": "ok", "description": "fine"},
        {"kind": "unknown", "name": "bad", "description": "…"},
        {"kind": "tag", "name": "", "description": "blank name"},
    ])
    assert r["added"] == 1
    assert len(r["errors"]) == 2


# ---- similarity math ------------------------------------------------------


def test_score_anchors_returns_matches_above_threshold(cfg, conn):
    set_anchor(conn, cfg, "category", "physics",
               "Physics research including theoretical, experimental, quantum, "
               "and applied topics.")
    set_anchor(conn, cfg, "category", "bills",
               "Utility, subscription, and service billing statements with "
               "amount due and billing period.")

    import numpy as np
    from alexandria.ingest.embed import embed_texts
    doc_vec = embed_texts(
        ["A quantum field theory paper on gauge symmetries and mass generation."],
        cfg.embeddings,
    )[0]
    doc_vec = doc_vec / (np.linalg.norm(doc_vec) or 1.0)

    matches = score_anchors(conn, doc_vec, min_similarity=0.0)
    # Two anchors, both should have some score; the physics one should rank
    # higher than the bills one.
    names_ranked = [m.name for m in matches]
    assert names_ranked.index("physics") < names_ranked.index("bills")


def test_score_anchors_empty_when_none_stored(cfg, conn):
    import numpy as np
    doc_vec = np.zeros(cfg.embeddings.dim, dtype=np.float32)
    doc_vec[0] = 1.0
    assert score_anchors(conn, doc_vec, min_similarity=0.0) == []


# ---- classifier integration ----------------------------------------------


def test_anchor_overrides_neighbor_vote(cfg, conn, corpus_dir: Path):
    """When an anchor matches the doc above threshold, its label wins even
    if neighbor voting would have picked something else."""
    # Seed a mislabeled neighbor: a bills-shaped doc tagged as "misc".
    ingest_file(corpus_dir / "plain.txt", conn, cfg,
                category="misc", tags=["random"])
    ingest_file(corpus_dir / "second-bill.txt", conn, cfg,
                category="misc", tags=["random"])

    # New unlabeled bill.
    unlabeled = corpus_dir / "third-bill.txt"
    unlabeled.write_text(
        "Account statement. Amount due for services this billing period. "
        "Please remit payment by the due date shown. Utility charges apply.\n"
    )
    r = ingest_file(unlabeled, conn, cfg)

    # Anchor for "bills" — a good description that should match the doc.
    set_anchor(conn, cfg, "category", "bills",
               "Utility, subscription, or service billing statement with "
               "account holder, amount due, billing period, and payment "
               "instructions.")

    s = suggest_metadata(r.doc_id, conn, cfg)
    assert s is not None
    # Neighbor vote alone would have suggested "misc" (both neighbors labeled
    # that way). Anchor should override.
    assert s.suggested_category == "bills"
    assert s.category_source == "anchor"


def test_falls_back_to_neighbors_when_no_anchor_matches(cfg, conn, corpus_dir: Path, fake_embed):
    """No anchors defined → neighbor-based suggestion path still works."""
    ingest_file(corpus_dir / "plain.txt", conn, cfg,
                category="bills", tags=["utility"])
    ingest_file(corpus_dir / "second-bill.txt", conn, cfg,
                category="bills", tags=["utility"])
    unlabeled = corpus_dir / "third-bill.txt"
    unlabeled.write_text("Account balance due.\n")
    r = ingest_file(unlabeled, conn, cfg)

    s = suggest_metadata(r.doc_id, conn, cfg)
    assert s is not None
    assert s.suggested_category == "bills"
    assert s.category_source == "neighbor"
    assert s.anchor_matches == []


def test_anchor_matches_surfaced_even_when_neighbors_drive_tags(cfg, conn, corpus_dir: Path):
    """anchor_matches is always populated when anchors clear threshold, even
    if the driving field ends up coming from neighbors."""
    ingest_file(corpus_dir / "plain.txt", conn, cfg,
                category="bills", tags=["utility"])
    ingest_file(corpus_dir / "second-bill.txt", conn, cfg,
                category="bills", tags=["utility"])
    unlabeled = corpus_dir / "third-bill.txt"
    unlabeled.write_text("Account balance due for utility services.\n")
    r = ingest_file(unlabeled, conn, cfg)

    # Only a category anchor. Tag suggestions should still come from neighbors.
    set_anchor(conn, cfg, "category", "bills",
               "Utility, subscription, or service billing statements.")

    s = suggest_metadata(r.doc_id, conn, cfg)
    assert s is not None
    assert s.category_source == "anchor"
    # No tag anchors, so tags fall back to neighbor voting.
    assert s.tag_source in ("neighbor", "none")
    # anchor_matches is populated with the category hit.
    assert any(a.kind == "category" and a.name == "bills" for a in s.anchor_matches)
