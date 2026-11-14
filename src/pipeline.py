"""PoC Pipeline — Track T06: Dataset Semantic Integrity & Provenance Gate.

Lab contract = 5 automated steps, run by `python -m src.main ...`:
    01 load      input + provenance check (manifest.csv) + COCO<->YOLO round-trip check
    02 process   semantic invariants & provenance gate across candidate releases
    03 rank      prioritise top-B suspicious releases + render visual HTML contact sheet
    04 review    human-in-the-loop: queue -> reviewer decisions -> overrides + audit log
    05 decide    GO / HOLD / REWORK with reason_code + owner, write outputs
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .common.contact_sheet import render_contact_sheet
from .common.context import RunContext
from .common.decision import Decision, canonical_status, write_decision
from .common.io import sha256_file, write_json
from .common.manifest import verify_manifest
from .common.review import apply_overrides, load_review_decisions, make_review_queue, write_review_queue
from .metrics.tracks import get_track


# ---------------------------------------------------------------- Helpers: COCO <-> YOLO round-trip
def coco_to_yolo_rows(ann_data: dict, classes: list[str]) -> dict[str, list[tuple[int, float, float, float, float]]]:
    """Convert COCO annotations into normalized YOLO rows keyed by image stem."""
    cats = {int(c["id"]): str(c["name"]) for c in ann_data.get("categories", [])}
    name_to_idx = {name: idx for idx, name in enumerate(classes)}
    imgs = {int(i["id"]): i for i in ann_data.get("images", [])}
    out: dict[str, list[tuple[int, float, float, float, float]]] = {
        Path(str(i["file_name"])).stem: [] for i in ann_data.get("images", [])
    }
    for a in ann_data.get("annotations", []):
        img = imgs.get(int(a["image_id"]))
        if not img:
            continue
        cat_name = cats.get(int(a["category_id"]))
        if cat_name not in name_to_idx:
            continue
        cls_idx = name_to_idx[cat_name]
        W, H = float(img["width"]), float(img["height"])
        x, y, w, h = [float(v) for v in a["bbox"]]
        xc = (x + w / 2.0) / W
        yc = (y + h / 2.0) / H
        wn = w / W
        hn = h / H
        stem = Path(str(img["file_name"])).stem
        out.setdefault(stem, []).append((cls_idx, xc, yc, wn, hn))
    return out


def verify_coco_yolo_roundtrip(parent_dir: Path, tol: float = 1e-3) -> dict[str, Any]:
    """Verify round-trip equivalence between COCO annotations.json and yolo_labels/*.txt."""
    ann_path = parent_dir / "annotations.json"
    classes_path = parent_dir / "classes.txt"
    yolo_dir = parent_dir / "yolo_labels"
    if not (ann_path.exists() and classes_path.exists() and yolo_dir.exists()):
        return {"verified": False, "reason": "missing_parent_yolo_artefacts"}

    ann_data = json.loads(ann_path.read_text(encoding="utf-8"))
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = coco_to_yolo_rows(ann_data, classes)

    mismatches = []
    total_boxes = 0
    for stem, exp_rows in sorted(expected.items()):
        txt_path = yolo_dir / f"{stem}.txt"
        if not txt_path.exists():
            mismatches.append({"stem": stem, "error": "missing_yolo_file"})
            continue
        lines = [ln.strip() for ln in txt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) != len(exp_rows):
            mismatches.append({"stem": stem, "error": "box_count_mismatch", "expected": len(exp_rows), "actual": len(lines)})
            continue
        for idx, (ln, exp) in enumerate(zip(lines, exp_rows)):
            parts = ln.split()
            cls_id = int(parts[0])
            vals = [float(v) for v in parts[1:5]]
            total_boxes += 1
            if cls_id != exp[0] or max(abs(vals[k] - exp[k + 1]) for k in range(4)) > tol:
                mismatches.append({"stem": stem, "line": idx, "expected": exp, "actual": (cls_id, *vals)})

    return {
        "verified": len(mismatches) == 0,
        "n_images_checked": len(expected),
        "n_boxes_checked": total_boxes,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:10],
    }


def build_croissant_metadata(ctx: RunContext, data: dict, scored: pd.DataFrame) -> dict[str, Any]:
    """Generate MLCommons Croissant 1.1 JSON-LD metadata with PROV-O lineage and RAI vocabulary."""
    parent_sha = data.get("parent_sha256", "")
    manifest_df: pd.DataFrame | None = data.get("manifest")
    n_files = int(len(manifest_df)) if manifest_df is not None else 0
    return {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "dct": "http://purl.org/dc/terms/",
            "prov": "http://www.w3.org/ns/prov#",
            "rai": "http://mlcommons.org/croissant/RAI/",
        },
        "@type": "sc:Dataset",
        "name": "day06-t06-semantic-integrity-release-gate",
        "version": "1.1.0",
        "description": (
            "Dataset Semantic Integrity & Provenance Gate benchmark (20 candidate releases derived from "
            "parent release v1.0 with verified SHA-256 provenance and semantic invariant validation)."
        ),
        "license": "https://creativecommons.org/publicdomain/zero/1.0/",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "prov:wasDerivedFrom": {
            "@type": "cr:FileObject",
            "name": "releases/v1.0/annotations.json",
            "sha256": parent_sha,
            "version": "v1.0",
        },
        "rai:dataCollection": "Synthetic autonomous driving traffic scenes (CC0, zero real PII).",
        "rai:dataUseCases": "Release gating, silent semantic corruption detection, and provenance auditing.",
        "rai:dataLimitations": "Synthetic thumbnails (320x180); legal/production thresholds require domain owner sign-off.",
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": "manifest.csv",
                "name": "manifest.csv",
                "contentUrl": str(ctx.input_dir / "manifest.csv"),
                "encodingFormat": "text/csv",
                "sha256": sha256_file(ctx.input_dir / "manifest.csv") if (ctx.input_dir / "manifest.csv").exists() else "",
            },
            {
                "@type": "cr:FileObject",
                "@id": "release_decisions.csv",
                "name": "release_decisions.csv",
                "contentUrl": str(ctx.out("release_decisions.csv")),
                "encodingFormat": "text/csv",
            },
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "name": "release_decisions",
                "description": f"Evaluated {len(scored)} candidate releases across {n_files} provenance-tracked files.",
                "field": [
                    {"@type": "cr:Field", "name": "release_id", "dataType": "sc:Text"},
                    {"@type": "cr:Field", "name": "decision", "dataType": "sc:Text"},
                    {"@type": "cr:Field", "name": "root_cause", "dataType": "sc:Text"},
                    {"@type": "cr:Field", "name": "score", "dataType": "sc:Float"},
                ],
            }
        ],
    }


# ---------------------------------------------------------------- 01 load
def load(ctx: RunContext) -> dict:
    manifest = ctx.input_dir / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        ctx.audit.log(actor="pipeline", action="provenance_problems", item_id=str(manifest), after=problems[:20])
        if ctx.config.get("strict_provenance", True):
            raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")
    data: dict[str, Any] = {"manifest": pd.read_csv(manifest) if manifest.exists() else None}

    if ctx.track != "t06":
        return data

    releases_csv = ctx.input_dir / "releases.csv"
    label_maps_path = ctx.input_dir / "releases" / "label_maps.json"
    parent_rel = str(ctx.param("parent_release", "releases/v1.0"))
    parent_dir = ctx.input_dir / parent_rel
    parent_ann_path = parent_dir / "annotations.json"
    parent_meta_path = parent_dir / "release_meta.json"

    for req_path in (releases_csv, label_maps_path, parent_ann_path, parent_meta_path):
        if not req_path.exists():
            raise FileNotFoundError(f"Required T06 input file missing: {req_path}")

    parent_sha256 = sha256_file(parent_ann_path)
    parent_ann = json.loads(parent_ann_path.read_text(encoding="utf-8"))
    parent_meta = json.loads(parent_meta_path.read_text(encoding="utf-8"))
    label_maps = json.loads(label_maps_path.read_text(encoding="utf-8"))

    roundtrip = verify_coco_yolo_roundtrip(parent_dir)
    ctx.stats["coco_yolo_roundtrip"] = {k: v for k, v in roundtrip.items() if k != "mismatches"}
    ctx.audit.log(
        actor="pipeline",
        action="verify_parent_roundtrip",
        item_id=parent_rel,
        after=ctx.stats["coco_yolo_roundtrip"],
        reason="Verified COCO <-> YOLO round-trip on parent release",
    )

    data.update(
        {
            "releases": pd.read_csv(releases_csv),
            "label_maps": label_maps,
            "parent_dir": parent_dir,
            "parent_sha256": parent_sha256,
            "parent_ann": parent_ann,
            "parent_meta": parent_meta,
            "roundtrip": roundtrip,
        }
    )
    return data


# ---------------------------------------------------------------- 02 process
def _inspect_candidate_release(
    ctx: RunContext,
    rel_id: str,
    rel_dir: Path,
    split: str,
    data: dict,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate a single candidate release against parent invariants and provenance rules."""
    parent_ann = data["parent_ann"]
    parent_meta = data["parent_meta"]
    parent_sha256 = data["parent_sha256"]
    label_maps = data["label_maps"]

    p_imgs_by_id = {int(i["id"]): i for i in parent_ann["images"]}
    p_anns_by_id = {int(a["id"]): a for a in parent_ann["annotations"]}
    p_cats = {int(c["id"]): str(c["name"]) for c in parent_ann["categories"]}

    bbox_shift_px_tol = float(ctx.param("bbox_shift_px_tol", 2.0))
    bbox_shift_min_count = int(ctx.param("bbox_shift_min_count", 10))
    dropped_frame_min_count = int(ctx.param("dropped_frame_min_count", 1))

    meta_path = rel_dir / "release_meta.json"
    ann_path = rel_dir / "annotations.json"
    if not meta_path.exists() or not ann_path.exists():
        row = {
            "release_id": rel_id,
            "decision": "HOLD",
            "root_cause": "missing_files",
            "score": 1.0,
            "reason": "Missing release_meta.json or annotations.json",
            "suggested_action": "HOLD",
            "slice": split,
            "evidence_path": "images/frame_000.jpg",
        }
        return row, [], {"release_id": rel_id, "status": "HOLD", "root_cause": "missing_files"}

    try:
        cand_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cand_ann = json.loads(ann_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        row = {
            "release_id": rel_id,
            "decision": "HOLD",
            "root_cause": "corrupt_json",
            "score": 1.0,
            "reason": f"Corrupt JSON file: {exc}",
            "suggested_action": "REWORK",
            "slice": split,
            "evidence_path": "images/frame_000.jpg",
        }
        return row, [], {"release_id": rel_id, "status": "HOLD", "root_cause": "corrupt_json", "error": str(exc)}

    for key in ("images", "annotations", "categories"):
        if key not in cand_ann or not isinstance(cand_ann[key], list):
            row = {
                "release_id": rel_id,
                "decision": "HOLD",
                "root_cause": "corrupt_json",
                "score": 1.0,
                "reason": f"Missing required COCO list '{key}'",
                "suggested_action": "REWORK",
                "slice": split,
                "evidence_path": "images/frame_000.jpg",
            }
            return row, [], {"release_id": rel_id, "status": "HOLD", "root_cause": "corrupt_json"}

    # Check 0: Provenance hash check against parent v1.0
    cand_parent_sha = cand_meta.get("parent_sha256")
    provenance_ok = cand_parent_sha == parent_sha256

    # Check 1: Label map completeness (stale_map) vs class mapping permutation (class_swap)
    lm_ver = str(cand_meta.get("label_map_version") or parent_meta.get("label_map_version", "lm-3"))
    expected_map_raw = label_maps.get(lm_ver, p_cats)
    expected_map = {int(k): str(v) for k, v in expected_map_raw.items()}
    cand_cats = {int(c["id"]): str(c["name"]) for c in cand_ann["categories"]}
    used_cat_ids = {int(a["category_id"]) for a in cand_ann["annotations"]}

    orphan_cat_ids = sorted(used_cat_ids - set(cand_cats.keys()))
    missing_cat_ids = sorted(set(expected_map.keys()) - set(cand_cats.keys()))
    is_stale_map = bool(orphan_cat_ids or missing_cat_ids or set(cand_cats.keys()) != set(expected_map.keys()))

    swapped_cat_ids = sorted(
        cid for cid in set(cand_cats.keys()) & set(expected_map.keys()) if cand_cats[cid] != expected_map[cid]
    )
    is_class_swap = (not is_stale_map) and bool(swapped_cat_ids)

    # Check 2: ID uniqueness (id_collision)
    cand_img_list = cand_ann["images"]
    cand_ann_list = cand_ann["annotations"]
    img_id_counts = Counter(int(i["id"]) for i in cand_img_list)
    ann_id_counts = Counter(int(a["id"]) for a in cand_ann_list)
    dup_img_ids = sorted(k for k, v in img_id_counts.items() if v > 1)
    dup_ann_ids = sorted(k for k, v in ann_id_counts.items() if v > 1)
    n_dup_anns = sum(v - 1 for v in ann_id_counts.values() if v > 1)
    is_id_collision = bool(dup_img_ids or dup_ann_ids)

    # Check 3: Frame completeness & continuity (dropped_frames)
    cand_imgs_by_id = {int(i["id"]): i for i in cand_img_list}
    missing_img_ids = sorted(set(p_imgs_by_id.keys()) - set(cand_imgs_by_id.keys()))
    extra_img_ids = sorted(set(cand_imgs_by_id.keys()) - set(p_imgs_by_id.keys()))

    seq_frames: dict[str, list[int]] = defaultdict(list)
    for img in cand_img_list:
        if "sequence_id" in img and "frame_index" in img:
            seq_frames[str(img["sequence_id"])].append(int(img["frame_index"]))
    seq_gaps: dict[str, list[int]] = {}
    for seq_id, f_idxs in seq_frames.items():
        if len(f_idxs) > 1:
            s_idxs = sorted(f_idxs)
            expected_range = set(range(s_idxs[0], s_idxs[-1] + 1))
            gaps = sorted(expected_range - set(s_idxs))
            if gaps:
                seq_gaps[seq_id] = gaps

    is_dropped_frames = len(missing_img_ids) >= dropped_frame_min_count

    # Check 4: Bounding box shifts & out-of-bounds (bbox_shift) vs annotation-level class swap
    shifted_anns: list[dict[str, Any]] = []
    subpixel_jitter_anns: list[dict[str, Any]] = []
    oob_anns: list[dict[str, Any]] = []
    ann_level_cat_swaps: list[dict[str, Any]] = []

    for a in cand_ann_list:
        aid = int(a["id"])
        img_id = int(a["image_id"])
        x, y, w, h = [float(v) for v in a["bbox"]]
        im = cand_imgs_by_id.get(img_id)
        if im:
            W, H = float(im["width"]), float(im["height"])
            if w <= 0 or h <= 0 or x < -0.5 or y < -0.5 or (x + w) > W + 0.5 or (y + h) > H + 0.5:
                oob_anns.append({"ann_id": aid, "image_id": img_id, "bbox": [x, y, w, h]})

        # Only compare 1-to-1 non-colliding annotation IDs against parent
        if ann_id_counts[aid] == 1 and aid in p_anns_by_id:
            pa = p_anns_by_id[aid]
            if int(pa["image_id"]) == img_id:
                if int(a["category_id"]) != int(pa["category_id"]):
                    ann_level_cat_swaps.append(
                        {
                            "ann_id": aid,
                            "image_id": img_id,
                            "old_cat": int(pa["category_id"]),
                            "new_cat": int(a["category_id"]),
                            "bbox": [x, y, w, h],
                        }
                    )
                px, py, pw, ph = [float(v) for v in pa["bbox"]]
                max_d = max(abs(x - px), abs(y - py), abs(w - pw), abs(h - ph))
                if max_d > bbox_shift_px_tol:
                    shifted_anns.append(
                        {
                            "ann_id": aid,
                            "image_id": img_id,
                            "max_shift_px": round(max_d, 2),
                            "old_bbox": [px, py, pw, ph],
                            "new_bbox": [x, y, w, h],
                            "category_id": int(a["category_id"]),
                        }
                    )
                elif max_d > 0.0:
                    subpixel_jitter_anns.append({"ann_id": aid, "image_id": img_id, "max_shift_px": round(max_d, 3)})

    if not is_class_swap and not is_id_collision and len(ann_level_cat_swaps) > 0:
        is_class_swap = True

    is_bbox_shift = len(shifted_anns) >= bbox_shift_min_count or len(oob_anns) > 0

    # Determine root cause, decision, suspicion score, and visual evidence boxes
    boxes: list[dict[str, Any]] = []
    evidence_path = str(cand_img_list[0]["file_name"]) if cand_img_list else "images/frame_000.jpg"

    if not provenance_ok:
        decision = "HOLD"
        root_cause = "provenance_mismatch"
        score = 1.0
        reason = f"Parent SHA256 mismatch: meta={cand_parent_sha} vs actual={parent_sha256}"
    elif is_class_swap:
        decision = "HOLD"
        root_cause = "class_swap"
        score = 0.99
        swap_desc = ", ".join(f"id {cid}: '{expected_map[cid]}'->'{cand_cats[cid]}'" for cid in swapped_cat_ids)
        if not swap_desc and ann_level_cat_swaps:
            swap_desc = f"{len(ann_level_cat_swaps)} annotations changed category_id"
        reason = f"Silent class swap in label map ({swap_desc})"
        # Find an annotation belonging to one of the swapped category IDs for visual evidence
        target_cids = set(swapped_cat_ids) or {x["new_cat"] for x in ann_level_cat_swaps}
        sample_anns = [a for a in cand_ann_list if int(a["category_id"]) in target_cids]
        if sample_anns:
            sa = sample_anns[0]
            im = cand_imgs_by_id.get(int(sa["image_id"]))
            if im:
                evidence_path = str(im["file_name"])
            for a in cand_ann_list:
                if int(a["image_id"]) == int(sa["image_id"]) and int(a["category_id"]) in target_cids:
                    cid = int(a["category_id"])
                    x, y, w, h = a["bbox"]
                    boxes.append(
                        {
                            "x": x,
                            "y": y,
                            "w": w,
                            "h": h,
                            "color": "red",
                            "label": f"SWAP id={cid}: {expected_map.get(cid)}->{cand_cats.get(cid)}",
                        }
                    )
    elif is_bbox_shift:
        decision = "HOLD"
        root_cause = "bbox_shift"
        max_shift_val = max((s["max_shift_px"] for s in shifted_anns), default=0.0)
        score = round(min(0.98, 0.90 + min(0.08, max_shift_val / 500.0)), 4)
        reason = (
            f"Systematic bbox shift: {len(shifted_anns)}/{len(cand_ann_list)} boxes shifted > {bbox_shift_px_tol}px "
            f"(max shift = {max_shift_val:.1f}px)"
        )
        if shifted_anns:
            worst = max(shifted_anns, key=lambda x: x["max_shift_px"])
            im = cand_imgs_by_id.get(int(worst["image_id"]))
            if im:
                evidence_path = str(im["file_name"])
            for s in shifted_anns:
                if int(s["image_id"]) == int(worst["image_id"]):
                    ox, oy, ow, oh = s["old_bbox"]
                    nx, ny, nw, nh = s["new_bbox"]
                    boxes.append({"x": ox, "y": oy, "w": ow, "h": oh, "color": "lime", "label": f"v1.0 #{s['ann_id']}"})
                    boxes.append(
                        {
                            "x": nx,
                            "y": ny,
                            "w": nw,
                            "h": nh,
                            "color": "red",
                            "label": f"shift {s['max_shift_px']:.1f}px",
                        }
                    )
    elif is_stale_map:
        decision = "HOLD"
        root_cause = "stale_map"
        score = 0.95
        reason = (
            f"Stale label map: meta declares {lm_ver} ({len(expected_map)} classes), "
            f"categories has {len(cand_cats)} classes (missing IDs={missing_cat_ids}, orphan used IDs={orphan_cat_ids})"
        )
        target_orphans = set(orphan_cat_ids) or set(missing_cat_ids)
        orphan_anns = [a for a in cand_ann_list if int(a["category_id"]) in target_orphans]
        if orphan_anns:
            oa = orphan_anns[0]
            im = cand_imgs_by_id.get(int(oa["image_id"]))
            if im:
                evidence_path = str(im["file_name"])
            for a in orphan_anns:
                if int(a["image_id"]) == int(oa["image_id"]):
                    x, y, w, h = a["bbox"]
                    cid = int(a["category_id"])
                    boxes.append(
                        {
                            "x": x,
                            "y": y,
                            "w": w,
                            "h": h,
                            "color": "red",
                            "label": f"ORPHAN cat={cid} ({expected_map.get(cid, '?')})",
                        }
                    )
    elif is_id_collision:
        decision = "HOLD"
        root_cause = "id_collision"
        score = round(0.93 + min(0.04, n_dup_anns * 0.001), 4)
        reason = (
            f"Duplicate annotation/image IDs: {n_dup_anns} colliding annotation rows across "
            f"{len(dup_ann_ids)} duplicate IDs (first IDs: {dup_ann_ids[:5]})"
        )
        if dup_ann_ids:
            first_dup_id = dup_ann_ids[0]
            colliding = [a for a in cand_ann_list if int(a["id"]) == first_dup_id]
            im = cand_imgs_by_id.get(int(colliding[0]["image_id"]))
            if im:
                evidence_path = str(im["file_name"])
            for idx, a in enumerate(colliding):
                if int(a["image_id"]) == int(colliding[0]["image_id"]):
                    x, y, w, h = a["bbox"]
                    boxes.append(
                        {"x": x, "y": y, "w": w, "h": h, "color": "red", "label": f"DUP id={first_dup_id} (#{idx+1})"}
                    )
    elif is_dropped_frames:
        decision = "HOLD"
        root_cause = "dropped_frames"
        score = round(0.90 + min(0.05, len(missing_img_ids) * 0.002), 4)
        reason = (
            f"Dropped frames: {len(missing_img_ids)}/{len(p_imgs_by_id)} parent images missing "
            f"(IDs: {missing_img_ids[:6]}...); sequence gaps in {list(seq_gaps.keys())}"
        )
        first_missing_id = missing_img_ids[0]
        pim = p_imgs_by_id[first_missing_id]
        evidence_path = str(pim["file_name"])
        for pa in parent_ann["annotations"]:
            if int(pa["image_id"]) == first_missing_id:
                x, y, w, h = pa["bbox"]
                boxes.append(
                    {"x": x, "y": y, "w": w, "h": h, "color": "red", "label": f"DROPPED img_id={first_missing_id}"}
                )
    else:
        decision = "PASS"
        root_cause = "clean"
        max_jitter = max((s["max_shift_px"] for s in subpixel_jitter_anns), default=0.0)
        score = round(0.04 + (0.01 if extra_img_ids else 0.0), 4)
        reason = (
            f"All semantic invariants passed (benign changes only: +{len(extra_img_ids)} new images, "
            f"{len(subpixel_jitter_anns)} boxes with sub-pixel jitter <= {max_jitter:.2f}px)"
        )
        if cand_img_list:
            evidence_path = str(cand_img_list[-1]["file_name"])

    row = {
        "release_id": rel_id,
        "decision": decision,
        "root_cause": root_cause,
        "score": float(score),
        "reason": reason,
        "suggested_action": "HOLD" if decision != "PASS" else "PASS",
        "slice": split,
        "evidence_path": evidence_path,
    }
    detail = {
        "release_id": rel_id,
        "split": split,
        "decision": decision,
        "root_cause": root_cause,
        "score": float(score),
        "provenance_ok": provenance_ok,
        "label_map_version": lm_ver,
        "n_images": len(cand_img_list),
        "n_annotations": len(cand_ann_list),
        "n_categories": len(cand_cats),
        "missing_parent_image_ids": missing_img_ids,
        "extra_image_ids": extra_img_ids,
        "sequence_gaps": seq_gaps,
        "orphan_category_ids": orphan_cat_ids,
        "missing_category_ids": missing_cat_ids,
        "swapped_category_ids": swapped_cat_ids,
        "duplicate_annotation_ids": dup_ann_ids,
        "n_duplicate_annotation_rows": n_dup_anns,
        "n_shifted_boxes_gt_tol": len(shifted_anns),
        "max_bbox_shift_px": max((s["max_shift_px"] for s in shifted_anns), default=0.0),
        "n_subpixel_jitter_boxes": len(subpixel_jitter_anns),
        "n_oob_boxes": len(oob_anns),
        "reason": reason,
    }
    return row, boxes, detail


def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    spec = get_track(ctx.track)
    if ctx.track != "t06":
        raise NotImplementedError(
            f"Implement process() for {spec.track} ({spec.title}). It must produce a DataFrame with at least "
            f"{spec.pred_columns} (saved as out/{spec.pred_file}). See docs/tracks/{spec.track.upper()}.md"
        )

    rows = []
    boxes_by_item: dict[str, list[dict[str, Any]]] = {}
    validation_details: list[dict[str, Any]] = []

    for r in data["releases"].itertuples(index=False):
        rel_id = str(r.release_id)
        rel_dir = ctx.input_dir / str(r.path)
        split = str(getattr(r, "split", "all"))
        row, boxes, detail = _inspect_candidate_release(ctx, rel_id, rel_dir, split, data)
        rows.append(row)
        if boxes:
            boxes_by_item[rel_id] = boxes
        validation_details.append(detail)
        ctx.audit.log(
            actor="pipeline",
            action="validate_release",
            item_id=rel_id,
            after={"decision": row["decision"], "root_cause": row["root_cause"], "score": row["score"]},
            reason=row["reason"],
        )

    scored = pd.DataFrame(rows)
    ctx.stats["boxes_by_item"] = boxes_by_item
    ctx.stats["n_defective_detected"] = int((scored["decision"] != "PASS").sum())
    ctx.stats["n_clean_passed"] = int((scored["decision"] == "PASS").sum())
    ctx.stats["root_cause_counts"] = dict(Counter(scored["root_cause"]))

    # Save machine-readable validation report (180m milestone requirement)
    write_json(
        ctx.out("validation_report.json"),
        {
            "track": ctx.track,
            "parent_release": str(ctx.param("parent_release", "releases/v1.0")),
            "parent_sha256": data["parent_sha256"],
            "coco_yolo_roundtrip": data["roundtrip"],
            "summary": {
                "n_releases": len(scored),
                "n_pass": ctx.stats["n_clean_passed"],
                "n_hold": ctx.stats["n_defective_detected"],
                "root_causes": ctx.stats["root_cause_counts"],
            },
            "releases": validation_details,
        },
    )

    # Save Croissant 1.1 provenance metadata (Bonus requirement)
    write_json(ctx.out("croissant_metadata.json"), build_croissant_metadata(ctx, data, scored))

    return scored


# ---------------------------------------------------------------- 03 rank
def rank(ctx: RunContext, scored: pd.DataFrame) -> pd.DataFrame:
    """Rank top-B suspicious items by `score` and render offline HTML contact sheet."""
    budget = int(ctx.param("budget", 10))
    if "score" not in scored.columns:
        raise ValueError(
            "rank(): process() output needs a numeric `score` column (higher = review first), "
            "or adapt rank() to your track"
        )
    q = make_review_queue(
        scored,
        id_col=scored.columns[0],
        score_col="score",
        budget=budget,
        reason_col="reason" if "reason" in scored.columns else None,
        action_col="suggested_action" if "suggested_action" in scored.columns else None,
        slice_col="slice" if "slice" in scored.columns else None,
        evidence_col="evidence_path" if "evidence_path" in scored.columns else None,
    )
    write_review_queue(q, ctx.out("review_queue.csv"))
    boxes_by_item = ctx.stats.pop("boxes_by_item", None)
    if "evidence_path" in q.columns and (q["evidence_path"].astype(str) != "").any():
        render_contact_sheet(
            q,
            ctx.out("contact_sheet.html"),
            image_root=ctx.input_dir,
            boxes_by_item=boxes_by_item,
            title=f"{ctx.track.upper()} Semantic Integrity & Provenance Review Queue",
        )
    return q


# ---------------------------------------------------------------- 04 review
def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    """If a reviewed queue exists (out/review_queue_reviewed.csv), validate and apply it."""
    reviewed_path = ctx.output_dir / "review_queue_reviewed.csv"
    if not reviewed_path.exists():
        alt_path = ctx.param("reviewed_queue_path", None)
        default_outs = {Path("out").resolve(), Path("out/poc").resolve()}
        if alt_path and Path(alt_path).exists() and ctx.output_dir.resolve() in default_outs:
            reviewed_path = Path(alt_path)

    if not reviewed_path.exists():
        ctx.stats["n_reviewed"] = 0
        return scored

    reviewed = load_review_decisions(reviewed_path)
    ctx.stats["n_reviewed"] = int(len(reviewed))

    # Preserve raw predictions before applying human overrides
    scored.to_csv(ctx.out("raw_release_decisions.csv"), index=False)

    # Log human approvals/escalations/skips and normalize edit/reject overrides
    normalized_reviewed = reviewed.copy()
    root_cause_overrides: dict[str, str] = {}
    for idx, r in normalized_reviewed.iterrows():
        dec_type = str(r["reviewer_decision"]).strip().lower()
        ov_val = str(r["override_value"]).strip()
        item_id = str(r["item_id"])
        if dec_type in {"approve", "escalate", "skip"}:
            ctx.audit.log(
                actor=str(r["reviewer"]),
                action=f"review:{dec_type}",
                item_id=item_id,
                before=scored.loc[scored[scored.columns[0]].astype(str) == item_id, "decision"].tolist(),
                after=ov_val or dec_type,
                reason=str(r["review_note"]),
            )
            if dec_type == "escalate" and not ov_val:
                normalized_reviewed.loc[idx, "reviewer_decision"] = "edit"
                normalized_reviewed.loc[idx, "override_value"] = "HOLD"
        elif dec_type in {"edit", "reject"}:
            if ":" in ov_val:
                new_dec, new_rc = [p.strip() for p in ov_val.split(":", 1)]
                normalized_reviewed.loc[idx, "override_value"] = new_dec.upper()
                if new_rc:
                    root_cause_overrides[item_id] = new_rc
            elif not ov_val and dec_type == "reject":
                normalized_reviewed.loc[idx, "override_value"] = "REWORK"

    out = apply_overrides(
        scored,
        normalized_reviewed,
        id_col=scored.columns[0],
        value_col="decision",
        audit=ctx.audit,
    )
    for item_id, new_rc in root_cause_overrides.items():
        mask = out[scored.columns[0]].astype(str) == item_id
        if mask.any() and "root_cause" in out.columns:
            before_rc = out.loc[mask, "root_cause"].tolist()
            out.loc[mask, "root_cause"] = new_rc
            ctx.audit.log(
                actor="reviewer",
                action="override:root_cause",
                item_id=item_id,
                before=before_rc[0] if len(before_rc) == 1 else before_rc,
                after=new_rc,
                reason="Updated root_cause from reviewer override",
            )
    return out


# ---------------------------------------------------------------- 05 decide
def decide(ctx: RunContext, final: pd.DataFrame) -> Decision:
    owner = str(ctx.config.get("owner", "Tran Quang Vinh (2A202603054 - Data Governance Lead)"))
    n_total = int(len(final))
    n_blocked = int(final["decision"].map(lambda s: canonical_status(str(s)) != "GO").sum())
    n_pass = n_total - n_blocked
    rc_counts = dict(Counter(final["root_cause"])) if "root_cause" in final.columns else {}

    # Check if any defective release (root_cause != "clean") was mistakenly given GO/PASS
    unblocked_defects = 0
    if "root_cause" in final.columns:
        unblocked_defects = int(
            final[(final["root_cause"] != "clean") & (final["decision"].map(lambda s: canonical_status(str(s)) == "GO"))].shape[0]
        )

    if unblocked_defects > 0:
        status = "HOLD"
        reason_code = "UNBLOCKED_SILENT_DEFECT"
        reason = f"{unblocked_defects} defective release(s) remain unblocked; halting release pipeline."
    else:
        status = "PASS"
        reason_code = "SEMANTIC_GATE_VERIFIED"
        reason = (
            f"Semantic integrity & provenance gate verified {n_total} candidate releases: "
            f"{n_pass} clean releases approved (PASS), {n_blocked} silent-defect releases quarantined (HOLD)."
        )

    return Decision(
        status=status,
        reason_code=reason_code,
        reason=reason,
        owner=owner,
        scope=ctx.track,
        evidence=[
            str(ctx.out("release_decisions.csv")),
            str(ctx.out("validation_report.json")),
            str(ctx.out("review_queue.csv")),
            str(ctx.out("audit_log.jsonl")),
        ],
        metrics={
            "n_releases": n_total,
            "n_pass": n_pass,
            "n_blocked": n_blocked,
            "n_reviewed": ctx.stats.get("n_reviewed", 0),
            "root_causes": rc_counts,
        },
    )


def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    data = load(ctx)
    scored = process(ctx, data)
    ctx.stats["n_items"] = len(scored)
    queue = rank(ctx, scored)
    final = review(ctx, scored, queue)
    final[spec.pred_columns + [c for c in final.columns if c not in spec.pred_columns]].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(decide(ctx, final), ctx.out("decision.json"), ctx.out("decision_log.csv"))
