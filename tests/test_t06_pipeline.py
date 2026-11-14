"""Unit and integration tests for Track T06 (Dataset Semantic Integrity & Provenance Gate)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src import main as runner
from src.metrics.tracks import eval_t06
from src.pipeline import verify_coco_yolo_roundtrip

FX_T06 = Path("fixtures/t06_release_gate")

EXPECTED_DECISIONS = {
    "r01": ("HOLD", "stale_map"),
    "r02": ("HOLD", "dropped_frames"),
    "r03": ("PASS", "clean"),
    "r04": ("PASS", "clean"),
    "r05": ("HOLD", "bbox_shift"),
    "r06": ("PASS", "clean"),
    "r07": ("HOLD", "class_swap"),
    "r08": ("HOLD", "id_collision"),
    "r09": ("HOLD", "stale_map"),
    "r10": ("HOLD", "bbox_shift"),
    "r11": ("PASS", "clean"),
    "r12": ("PASS", "clean"),
    "r13": ("PASS", "clean"),
    "r14": ("PASS", "clean"),
    "r15": ("PASS", "clean"),
    "r16": ("HOLD", "dropped_frames"),
    "r17": ("PASS", "clean"),
    "r18": ("HOLD", "id_collision"),
    "r19": ("PASS", "clean"),
    "r20": ("HOLD", "class_swap"),
}


def test_t06_poc_end_to_end_reproducible(tmp_path: Path) -> None:
    out1 = tmp_path / "poc1"
    rc1 = runner.main(["--track", "t06", "--input", str(FX_T06), "--output", str(out1), "--offline", "--seed", "42"])
    assert rc1 == 0

    for fname in [
        "release_decisions.csv",
        "validation_report.json",
        "croissant_metadata.json",
        "review_queue.csv",
        "contact_sheet.html",
        "decision.json",
        "decision_log.csv",
        "audit_log.jsonl",
        "run_info.json",
    ]:
        assert (out1 / fname).exists(), f"Missing output file: {fname}"

    pred = pd.read_csv(out1 / "release_decisions.csv")
    assert len(pred) == 20
    for row in pred.itertuples(index=False):
        exp_dec, exp_rc = EXPECTED_DECISIONS[row.release_id]
        assert row.decision == exp_dec, f"{row.release_id}: expected {exp_dec}, got {row.decision}"
        assert row.root_cause == exp_rc, f"{row.release_id}: expected {exp_rc}, got {row.root_cause}"

    # Reproducibility check with same seed
    out2 = tmp_path / "poc2"
    rc2 = runner.main(["--track", "t06", "--input", str(FX_T06), "--output", str(out2), "--offline", "--seed", "42"])
    assert rc2 == 0
    assert (out1 / "release_decisions.csv").read_text() == (out2 / "release_decisions.csv").read_text()


def test_t06_baseline_end_to_end(tmp_path: Path) -> None:
    out_b = tmp_path / "baseline"
    rc = runner.main(
        [
            "--track",
            "t06",
            "--input",
            str(FX_T06),
            "--pipeline",
            "src.baseline",
            "--output",
            str(out_b),
            "--run-name",
            "baseline",
            "--offline",
            "--seed",
            "42",
        ]
    )
    assert rc == 0
    pred = pd.read_csv(out_b / "release_decisions.csv")
    assert len(pred) == 20
    # Schema-only baseline passes all 20 valid-JSON candidate releases (misses silent semantic corruptions)
    assert (pred["decision"] == "PASS").all()


def test_t06_human_in_the_loop_override_and_audit(tmp_path: Path) -> None:
    out_dir = tmp_path / "poc_hitl"
    assert runner.main(["--track", "t06", "--input", str(FX_T06), "--output", str(out_dir), "--offline", "--seed", "42"]) == 0

    rq = pd.read_csv(out_dir / "review_queue.csv", dtype=str).fillna("")
    assert len(rq) == 10
    # Reviewer inspects top items: approves first item and edits second item to REWORK:class_swap
    rq.loc[0, ["reviewer", "reviewer_decision", "override_value", "review_note", "reviewed_at"]] = [
        "Tran Quang Vinh",
        "approve",
        "",
        "Confirmed class_swap on r07; keep quarantined",
        "2026-11-14T15:35:00Z",
    ]
    target_id = rq.loc[1, "item_id"]
    rq.loc[1, ["reviewer", "reviewer_decision", "override_value", "review_note", "reviewed_at"]] = [
        "Tran Quang Vinh",
        "edit",
        "REWORK:class_swap",
        "Send back to vendor_1 for label map remediation",
        "2026-11-14T15:36:00Z",
    ]
    rq.to_csv(out_dir / "review_queue_reviewed.csv", index=False)

    # Re-run pipeline to apply human review decisions
    assert runner.main(["--track", "t06", "--input", str(FX_T06), "--output", str(out_dir), "--offline", "--seed", "42"]) == 0
    assert (out_dir / "raw_release_decisions.csv").exists()

    final_df = pd.read_csv(out_dir / "release_decisions.csv")
    raw_df = pd.read_csv(out_dir / "raw_release_decisions.csv")
    assert raw_df.loc[raw_df["release_id"] == target_id, "decision"].item() == "HOLD"
    assert final_df.loc[final_df["release_id"] == target_id, "decision"].item() == "REWORK"

    audit_events = [json.loads(line) for line in (out_dir / "audit_log.jsonl").read_text().splitlines() if line.strip()]
    actors = {ev["actor"] for ev in audit_events}
    assert "Tran Quang Vinh" in actors


def test_t06_shock_corrupt_json_and_missing_file(tmp_path: Path) -> None:
    # Copy minimal fixture structure to tmp_path to simulate shock tests safely
    fx_copy = tmp_path / "t06_release_gate"
    shutil.copytree(FX_T06, fx_copy)

    # 1. Corrupt r03/annotations.json by truncating it in half -> provenance check must fail loudly (exit code 2)
    r03_ann = fx_copy / "releases/candidates/r03/annotations.json"
    raw_text = r03_ann.read_text(encoding="utf-8")
    r03_ann.write_text(raw_text[: len(raw_text) // 2], encoding="utf-8")

    rc_strict = runner.main(["--track", "t06", "--input", str(fx_copy), "--output", str(tmp_path / "out_strict"), "--offline"])
    assert rc_strict == 2

    # 2. When strict_provenance is disabled, the release gate catches corrupt JSON and quarantines r03 on HOLD
    cfg_no_strict = tmp_path / "lab_no_strict.yaml"
    cfg_no_strict.write_text(
        "group_id: G06\nowner: 'Tran Quang Vinh'\ntrack: t06\nstrict_provenance: false\noffline: true\n",
        encoding="utf-8",
    )
    out_nostrict = tmp_path / "out_nostrict"
    rc_nostrict = runner.main(
        ["--config", str(cfg_no_strict), "--track", "t06", "--input", str(fx_copy), "--output", str(out_nostrict), "--offline"]
    )
    assert rc_nostrict == 0
    pred = pd.read_csv(out_nostrict / "release_decisions.csv").set_index("release_id")
    assert pred.loc["r03", "decision"] == "HOLD"
    assert pred.loc["r03", "root_cause"] == "corrupt_json"


def test_t06_coco_yolo_roundtrip_verification(tmp_path: Path) -> None:
    parent_dir = FX_T06 / "releases/v1.0"
    res = verify_coco_yolo_roundtrip(parent_dir)
    assert res["verified"] is True
    assert res["n_images_checked"] == 100
    assert res["n_boxes_checked"] == 697
    assert res["n_mismatches"] == 0

    # Inject a silent class swap into a copy of parent YOLO labels and verify detection
    bad_parent = tmp_path / "v1.0_corrupt"
    shutil.copytree(parent_dir, bad_parent)
    f0 = bad_parent / "yolo_labels/frame_000.txt"
    lines = f0.read_text(encoding="utf-8").splitlines()
    parts = lines[0].split()
    parts[0] = str((int(parts[0]) + 1) % 5)
    lines[0] = " ".join(parts)
    f0.write_text("\n".join(lines) + "\n", encoding="utf-8")

    bad_res = verify_coco_yolo_roundtrip(bad_parent)
    assert bad_res["verified"] is False
    assert bad_res["n_mismatches"] >= 1


def test_t06_internal_self_evaluation_metrics(tmp_path: Path) -> None:
    # Build internal self-check ground truth from verified release inspection
    releases_df = pd.read_csv(FX_T06 / "releases.csv")
    gt_rows = []
    for r in releases_df.itertuples(index=False):
        exp_dec, exp_rc = EXPECTED_DECISIONS[r.release_id]
        gt_rows.append(
            {
                "release_id": r.release_id,
                "split": r.split,
                "is_defective": 1 if exp_dec != "PASS" else 0,
                "corruption_type": exp_rc,
            }
        )
    gt_csv = tmp_path / "self_gt_releases.csv"
    pd.DataFrame(gt_rows).to_csv(gt_csv, index=False)

    out_b = tmp_path / "baseline"
    out_p = tmp_path / "poc"
    assert runner.main(["--track", "t06", "--pipeline", "src.baseline", "--output", str(out_b), "--offline"]) == 0
    assert runner.main(["--track", "t06", "--output", str(out_p), "--offline"]) == 0

    res_b = eval_t06(gt_csv, out_b / "release_decisions.csv", {"split": "eval"})
    res_p = eval_t06(gt_csv, out_p / "release_decisions.csv", {"split": "eval"})

    b_metrics = {m.name: m.value for m in res_b.metrics}
    p_metrics = {m.name: m.value for m in res_p.metrics}

    assert b_metrics["Silent-defect Detection F1"] == 0.0
    assert p_metrics["Silent-defect Detection F1"] == pytest.approx(1.0)
    assert p_metrics["Defect recall"] == pytest.approx(1.0)
    assert p_metrics["False-block rate"] == pytest.approx(0.0)
    assert p_metrics["Root-cause accuracy"] == pytest.approx(1.0)

