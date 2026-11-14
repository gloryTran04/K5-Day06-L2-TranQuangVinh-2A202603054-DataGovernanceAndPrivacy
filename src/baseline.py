"""Baseline pipeline for Track T06 (Dataset Semantic Integrity & Provenance Gate).

Implements the required T06 baseline from docs/tracks/T06.md:
- Parses JSON / validates basic COCO schema (`images`, `annotations`, `categories`)
  and checks that referenced image files exist on disk.
- Does NOT check semantic invariants against the parent release or label map history.

Usage:
    python -m src.main --track t06 --pipeline src.baseline --output out/baseline --run-name baseline --offline --seed 42
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common.contact_sheet import render_contact_sheet
from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.io import sha256_file
from .common.manifest import verify_manifest
from .common.review import make_review_queue, write_review_queue
from .metrics.tracks import get_track

REQUIRED_IMAGE_KEYS = {"id", "file_name", "width", "height"}
REQUIRED_ANN_KEYS = {"id", "image_id", "category_id", "bbox"}
REQUIRED_CAT_KEYS = {"id", "name"}


def load(ctx: RunContext) -> dict:
    manifest = ctx.input_dir / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        ctx.audit.log(actor="baseline", action="provenance_problems", item_id=str(manifest), after=problems[:20])
        if ctx.config.get("strict_provenance", True):
            raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")

    releases_csv = ctx.input_dir / "releases.csv"
    if not releases_csv.exists():
        raise FileNotFoundError(f"Missing releases list: {releases_csv}")

    parent_rel = str(ctx.param("parent_release", "releases/v1.0"))
    parent_dir = ctx.input_dir / parent_rel
    parent_ann_path = parent_dir / "annotations.json"

    return {
        "manifest": pd.read_csv(manifest) if manifest.exists() else None,
        "releases": pd.read_csv(releases_csv),
        "parent_ann_sha256": sha256_file(parent_ann_path) if parent_ann_path.exists() else None,
    }


def _check_schema_and_files(ctx: RunContext, rel_dir: Path) -> tuple[bool, str, str]:
    """Check JSON parseability, basic COCO schema keys, and image file existence."""
    meta_path = rel_dir / "release_meta.json"
    ann_path = rel_dir / "annotations.json"
    if not meta_path.exists() or not ann_path.exists():
        return False, "missing_release_files", "Missing release_meta.json or annotations.json"

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, "corrupt_json", f"JSON parse error: {exc}"

    if not isinstance(meta, dict) or not isinstance(ann, dict):
        return False, "schema_error", "Top-level JSON must be an object"

    for key in ("images", "annotations", "categories"):
        if key not in ann or not isinstance(ann[key], list) or len(ann[key]) == 0:
            return False, "schema_error", f"Missing or empty top-level list '{key}'"

    for img in ann["images"]:
        if not isinstance(img, dict) or not REQUIRED_IMAGE_KEYS.issubset(img.keys()):
            return False, "schema_error", "Image entry missing required COCO keys"
        img_file = ctx.input_dir / str(img["file_name"])
        if not img_file.exists():
            return False, "missing_image_file", f"Referenced image does not exist: {img['file_name']}"

    for item in ann["annotations"]:
        if not isinstance(item, dict) or not REQUIRED_ANN_KEYS.issubset(item.keys()):
            return False, "schema_error", "Annotation entry missing required COCO keys"
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False, "schema_error", "Annotation bbox must be a 4-element list"

    for cat in ann["categories"]:
        if not isinstance(cat, dict) or not REQUIRED_CAT_KEYS.issubset(cat.keys()):
            return False, "schema_error", "Category entry missing required COCO keys"

    first_img = str(ann["images"][0]["file_name"])
    return True, "clean", first_img


def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    mode = str(ctx.param("baseline_mode", "schema_only")).lower()
    parent_sha = data.get("parent_ann_sha256")
    rows = []

    for r in data["releases"].itertuples(index=False):
        rel_id = str(r.release_id)
        rel_dir = ctx.input_dir / str(r.path)
        split = str(getattr(r, "split", "all"))
        ok, code, info = _check_schema_and_files(ctx, rel_dir)

        if not ok:
            decision = "HOLD"
            root_cause = code
            score = 1.0
            reason = info
            evidence_path = "images/frame_000.jpg"
        elif mode == "naive_diff":
            cand_sha = sha256_file(rel_dir / "annotations.json")
            if parent_sha and cand_sha != parent_sha:
                decision = "HOLD"
                root_cause = "raw_file_diff"
                score = 0.8
                reason = "annotations.json SHA256 differs from parent v1.0 (naive diff check)"
            else:
                decision = "PASS"
                root_cause = "clean"
                score = 0.0
                reason = "exact match with parent v1.0"
            evidence_path = info
        else:
            decision = "PASS"
            root_cause = "clean"
            score = 0.0
            reason = "JSON schema valid and all referenced image files exist"
            evidence_path = info

        rows.append(
            {
                "release_id": rel_id,
                "decision": decision,
                "root_cause": root_cause,
                "score": float(score),
                "reason": reason,
                "suggested_action": "HOLD" if decision != "PASS" else "PASS",
                "slice": split,
                "evidence_path": evidence_path,
            }
        )

    return pd.DataFrame(rows)


def rank(ctx: RunContext, scored: pd.DataFrame) -> pd.DataFrame:
    budget = int(ctx.param("budget", 10))
    q = make_review_queue(
        scored,
        id_col="release_id",
        score_col="score",
        budget=budget,
        reason_col="reason",
        action_col="suggested_action",
        slice_col="slice",
        evidence_col="evidence_path",
    )
    write_review_queue(q, ctx.out("review_queue.csv"))
    render_contact_sheet(
        q,
        ctx.out("contact_sheet.html"),
        image_root=ctx.input_dir,
        title="T06 Baseline — Schema & File Existence Gate",
    )
    return q


def decide(ctx: RunContext, final: pd.DataFrame) -> Decision:
    n_blocked = int((final["decision"] != "PASS").sum())
    return Decision(
        status="HOLD",
        reason_code="BASELINE_SCHEMA_ONLY",
        reason=f"Baseline checks syntax/schema and file existence only ({n_blocked}/{len(final)} blocked); semantic invariants not verified.",
        owner=ctx.config.get("owner", "Tran Quang Vinh (2A202603054 - Data Governance Lead)"),
        scope=ctx.track,
        evidence=[str(ctx.out("release_decisions.csv")), str(ctx.out("review_queue.csv"))],
        metrics={"n_releases": int(len(final)), "n_blocked": n_blocked},
    )


def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    data = load(ctx)
    scored = process(ctx, data)
    ctx.stats["n_items"] = len(scored)
    rank(ctx, scored)
    cols = spec.pred_columns + [c for c in scored.columns if c not in spec.pred_columns]
    scored[cols].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(decide(ctx, scored), ctx.out("decision.json"), ctx.out("decision_log.csv"))

