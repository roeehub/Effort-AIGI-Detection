#!/usr/bin/env python3
"""
Render an HTML gallery for manual inspection of a frozen Teams target-domain manifest.

This tool is intended for audit, not scoring. It reads the frozen manifest,
selects a deterministic subset of rows per slice, downloads a few representative
frames for each row, renders labeled strip cards, and writes an `index.html`
browser view.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import os
import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

THUMB_SIZE = 192
CARD_PADDING = 12
CARD_GAP = 10
TEXT_GAP = 6
CARD_BG = (24, 24, 24)
PAGE_BG = "#141414"
CARD_BORDER = "#3a3a3a"
TEXT_COLOR = (235, 235, 235)
MUTED_TEXT_COLOR = (185, 185, 185)
REAL_BORDER = (36, 170, 85)
FAKE_BORDER = (214, 78, 78)
MAX_CARD_TEXT_CHARS = 92

_GCS_CLIENT = None
_GCS_BUCKETS: Dict[str, Any] = {}


def _split_gs_uri(uri: str) -> Tuple[str, str]:
    stripped = uri.replace("gs://", "", 1)
    if "/" not in stripped:
        return stripped, ""
    bucket, blob = stripped.split("/", 1)
    return bucket, blob


def _get_gcs_bucket(bucket_name: str):
    global _GCS_CLIENT
    if _GCS_CLIENT is None:
        from google.cloud import storage

        _GCS_CLIENT = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    if bucket_name not in _GCS_BUCKETS:
        _GCS_BUCKETS[bucket_name] = _GCS_CLIENT.bucket(bucket_name)
    return _GCS_BUCKETS[bucket_name]


def _read_text_from_path(path: str) -> str:
    if path.startswith("gs://"):
        bucket_name, blob_path = _split_gs_uri(path)
        return _get_gcs_bucket(bucket_name).blob(blob_path).download_as_text()
    return Path(path).read_text()


def _read_bytes_from_path(path: str) -> bytes:
    if path.startswith("gs://"):
        bucket_name, blob_path = _split_gs_uri(path)
        return _get_gcs_bucket(bucket_name).blob(blob_path).download_as_bytes()
    return Path(path).read_bytes()


def _coerce_list(value: Optional[object]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def _load_manifest_payload(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    text = _read_text_from_path(path)
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    if data is None:
        import yaml

        data = yaml.safe_load(text)

    if isinstance(data, dict):
        rows = data.get("videos")
        if not isinstance(rows, list):
            raise ValueError(f"Manifest {path} must contain a top-level 'videos' list.")
        return data, rows
    if isinstance(data, list):
        return {"summary": {}, "videos": data}, data
    raise ValueError(f"Unsupported manifest format: {path}")


def _normalize_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def _slugify(value: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    if not text:
        return "item"
    return text[:max_len]


def _select_evenly_spaced(items: Sequence[Any], count: int) -> List[Any]:
    if count <= 0 or len(items) <= count:
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]

    step = (len(items) - 1) / float(count - 1)
    indices = sorted({int(round(i * step)) for i in range(count)})
    return [items[idx] for idx in indices]


def _slice_sort_key(slice_name: str) -> Tuple[int, str]:
    priority = [
        "teams_real_all",
        "teams_real_poor_quality",
        "teams_real_lighting_extreme",
        "teams_fake_all",
        "visomaster_enhanced_macro",
        "deeplive_enhanced",
    ]
    if slice_name in priority:
        return (priority.index(slice_name), slice_name)
    return (len(priority), slice_name)


def _select_default_slices(rows: List[Dict[str, Any]]) -> List[str]:
    slices = {
        str(slice_name)
        for row in rows
        for slice_name in _coerce_list(row.get("slices"))
        if str(slice_name).strip()
    }
    return sorted(slices, key=_slice_sort_key)


def _find_rows_for_slice(
    rows: Iterable[Dict[str, Any]],
    slice_name: str,
    split: Optional[str],
) -> List[Dict[str, Any]]:
    target = _normalize_name(slice_name)
    matched = []
    for row in rows:
        row_split = str(row.get("split") or "").strip()
        if split and row_split != split:
            continue
        row_slices = {_normalize_name(item) for item in _coerce_list(row.get("slices"))}
        if target in row_slices:
            matched.append(row)
    matched.sort(key=lambda row: (str(row.get("label")), str(row.get("video_id"))))
    return matched


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Menlo Bold.ttf",
                "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Menlo.ttc",
            "/System/Library/Fonts/Supplemental/Courier New.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _make_thumbnail(path: str, label: str) -> Image.Image:
    image = Image.open(io.BytesIO(_read_bytes_from_path(path))).convert("RGB")
    image.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)

    canvas = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), CARD_BG)
    x = (THUMB_SIZE - image.width) // 2
    y = (THUMB_SIZE - image.height) // 2
    canvas.paste(image, (x, y))

    border = REAL_BORDER if label == "real" else FAKE_BORDER
    draw = ImageDraw.Draw(canvas)
    for i in range(3):
        draw.rectangle((i, i, THUMB_SIZE - 1 - i, THUMB_SIZE - 1 - i), outline=border)
    return canvas


def _wrap_lines(text: str, width: int = MAX_CARD_TEXT_CHARS) -> List[str]:
    wrapped = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    return wrapped or [text]


def _render_card(
    row: Dict[str, Any],
    frame_paths: Sequence[str],
    output_path: Path,
) -> Dict[str, Any]:
    label = str(row.get("label") or "")
    thumbs = [_make_thumbnail(path, label=label) for path in frame_paths]

    title_font = _load_font(20, bold=True)
    body_font = _load_font(14, bold=False)

    strip_width = CARD_PADDING * 2
    if thumbs:
        strip_width += len(thumbs) * THUMB_SIZE + (len(thumbs) - 1) * CARD_GAP

    lines = []
    lines.extend(_wrap_lines(f"video={row.get('video_id', '')}", width=MAX_CARD_TEXT_CHARS))
    lines.extend(
        _wrap_lines(
            " | ".join(
                [
                    str(row.get("label") or ""),
                    f"split={row.get('split', '-')}",
                    f"method={row.get('method', '-')}",
                    f"source={row.get('source_kind', '-')}",
                ]
            ),
            width=MAX_CARD_TEXT_CHARS,
        )
    )
    lines.extend(
        _wrap_lines(
            " | ".join(
                [
                    f"prefix={row.get('prefix', '-')}",
                    f"session={row.get('session_id') or '-'}",
                    f"sequence={row.get('sequence_id') or '-'}",
                    f"rule={row.get('matched_rule') or '-'}",
                ]
            ),
            width=MAX_CARD_TEXT_CHARS,
        )
    )
    slices_text = ",".join(_coerce_list(row.get("slices"))) or "-"
    lines.extend(_wrap_lines(f"slices={slices_text}", width=MAX_CARD_TEXT_CHARS))

    dummy = Image.new("RGB", (16, 16))
    draw = ImageDraw.Draw(dummy)
    title_height = draw.textbbox((0, 0), "Ag", font=title_font)[3]
    body_height = draw.textbbox((0, 0), "Ag", font=body_font)[3]
    text_height = title_height + TEXT_GAP + len(lines) * (body_height + 2)

    card_width = max(strip_width, 1000)
    card_height = CARD_PADDING * 2 + text_height + CARD_GAP + THUMB_SIZE

    card = Image.new("RGB", (card_width, card_height), CARD_BG)
    draw = ImageDraw.Draw(card)
    draw.text((CARD_PADDING, CARD_PADDING), "Frozen Teams Target-Domain Row", fill=TEXT_COLOR, font=title_font)

    y = CARD_PADDING + title_height + TEXT_GAP
    for line in lines:
        draw.text((CARD_PADDING, y), line, fill=MUTED_TEXT_COLOR, font=body_font)
        y += body_height + 2

    x = CARD_PADDING
    y += CARD_GAP
    for thumb in thumbs:
        card.paste(thumb, (x, y))
        x += THUMB_SIZE + CARD_GAP

    output_path.parent.mkdir(parents=True, exist_ok=True)
    card.save(output_path, quality=92)

    return {
        "video_id": row.get("video_id"),
        "label": row.get("label"),
        "method": row.get("method"),
        "split": row.get("split"),
        "source_kind": row.get("source_kind"),
        "prefix": row.get("prefix"),
        "session_id": row.get("session_id"),
        "sequence_id": row.get("sequence_id"),
        "matched_rule": row.get("matched_rule"),
        "slices": _coerce_list(row.get("slices")),
        "card_path": str(output_path),
        "frame_paths": list(frame_paths),
    }


def _write_html(
    output_dir: Path,
    manifest_path: str,
    payload: Dict[str, Any],
    slice_sections: List[Dict[str, Any]],
    split: Optional[str],
    samples_per_slice: int,
    frames_per_video: int,
) -> Path:
    summary = payload.get("summary") or {}
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>Frozen Teams Target-Domain Manifest Audit</title>",
        "<style>",
        f"body {{ background: {PAGE_BG}; color: #eee; font-family: Menlo, Consolas, monospace; padding: 24px; }}",
        "h1, h2 { color: #fff; }",
        ".summary { background: #1e1e1e; border: 1px solid #2d2d2d; border-radius: 10px; padding: 16px; margin-bottom: 24px; }",
        ".section { margin: 28px 0 40px; }",
        ".section-meta { color: #bdbdbd; margin-bottom: 12px; }",
        ".cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 14px; }",
        f".card {{ background: #1b1b1b; border: 1px solid {CARD_BORDER}; border-radius: 10px; padding: 10px; }}",
        ".card img { width: 100%; height: auto; border-radius: 8px; display: block; }",
        ".meta { font-size: 12px; color: #b7b7b7; margin-top: 6px; line-height: 1.5; }",
        "pre { white-space: pre-wrap; word-break: break-word; }",
        "a { color: #8dc2ff; }",
        "</style>",
        "</head><body>",
        "<h1>Frozen Teams Target-Domain Manifest Audit</h1>",
        "<div class='summary'>",
        f"<p><b>Manifest:</b> {html.escape(manifest_path)}</p>",
        f"<p><b>Split filter:</b> {html.escape(split or 'all')} | <b>Samples per slice:</b> {samples_per_slice} | <b>Frames per video:</b> {frames_per_video}</p>",
        "<pre>",
        html.escape(json.dumps(summary, indent=2, sort_keys=True)),
        "</pre>",
        "</div>",
    ]

    for section in slice_sections:
        html_parts.append("<div class='section'>")
        html_parts.append(f"<h2>{html.escape(section['slice'])}</h2>")
        html_parts.append(
            "<div class='section-meta'>"
            f"Candidates in manifest: {section['candidate_count']} | "
            f"Rendered rows: {len(section['cards'])}"
            "</div>"
        )
        html_parts.append("<div class='cards'>")
        for card in section["cards"]:
            rel_path = os.path.relpath(card["card_path"], output_dir)
            meta_parts = [
                f"video={card['video_id']}",
                f"label={card['label']}",
                f"method={card['method']}",
                f"split={card['split']}",
                f"source={card['source_kind']}",
            ]
            if card.get("matched_rule"):
                meta_parts.append(f"rule={card['matched_rule']}")
            html_parts.append(
                "<div class='card'>"
                f"<img src='{html.escape(rel_path)}' alt='{html.escape(str(card['video_id']))}'>"
                f"<div class='meta'>{html.escape(' | '.join(meta_parts))}</div>"
                "</div>"
            )
        html_parts.append("</div></div>")

    html_parts.append("</body></html>")

    html_path = output_dir / "index.html"
    html_path.write_text("\n".join(html_parts))
    return html_path


def build_gallery(
    manifest_path: str,
    output_dir: str,
    split: Optional[str] = "dev",
    slices: Optional[List[str]] = None,
    samples_per_slice: int = 8,
    frames_per_video: int = 4,
) -> Dict[str, Any]:
    if samples_per_slice <= 0:
        raise ValueError("samples_per_slice must be > 0")
    if frames_per_video <= 0:
        raise ValueError("frames_per_video must be > 0")

    payload, rows = _load_manifest_payload(manifest_path)
    target_slices = slices or _select_default_slices(rows)

    output_path = Path(output_dir)
    cards_dir = output_path / "cards"
    output_path.mkdir(parents=True, exist_ok=True)
    cards_dir.mkdir(parents=True, exist_ok=True)

    rendered_cards: Dict[str, Dict[str, Any]] = {}
    slice_sections: List[Dict[str, Any]] = []

    for slice_name in target_slices:
        candidates = _find_rows_for_slice(rows, slice_name=slice_name, split=split)
        selected_rows = _select_evenly_spaced(candidates, samples_per_slice)
        cards = []
        for row in selected_rows:
            row_key = f"{row.get('label')}::{row.get('video_id')}"
            if row_key not in rendered_cards:
                frame_paths = _coerce_list(row.get("frame_paths"))
                selected_frames = _select_evenly_spaced(frame_paths, frames_per_video)
                card_name = (
                    f"{row.get('label', 'row')}_"
                    f"{_slugify(str(row.get('video_id', 'video')))}_"
                    f"{_stable_hash(row_key)}.jpg"
                )
                card_path = cards_dir / card_name
                rendered_cards[row_key] = _render_card(row=row, frame_paths=selected_frames, output_path=card_path)
            cards.append(rendered_cards[row_key])

        slice_sections.append(
            {
                "slice": slice_name,
                "candidate_count": len(candidates),
                "cards": cards,
            }
        )

    selection_payload = {
        "manifest_path": manifest_path,
        "split": split,
        "samples_per_slice": samples_per_slice,
        "frames_per_video": frames_per_video,
        "slice_sections": [
            {
                "slice": section["slice"],
                "candidate_count": section["candidate_count"],
                "cards": [
                    {
                        **card,
                        "card_path": os.path.relpath(card["card_path"], output_path),
                    }
                    for card in section["cards"]
                ],
            }
            for section in slice_sections
        ],
    }
    selection_path = output_path / "selection.json"
    selection_path.write_text(json.dumps(selection_payload, indent=2, sort_keys=True))

    html_path = _write_html(
        output_dir=output_path,
        manifest_path=manifest_path,
        payload=payload,
        slice_sections=slice_sections,
        split=split,
        samples_per_slice=samples_per_slice,
        frames_per_video=frames_per_video,
    )

    return {
        "html_path": str(html_path),
        "selection_path": str(selection_path),
        "rendered_card_count": len(rendered_cards),
        "slice_count": len(slice_sections),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an HTML audit gallery for a frozen Teams manifest.")
    parser.add_argument("--manifest", required=True, help="Path to the frozen manifest JSON/YAML.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for cards, index.html, and selection.json. Defaults next to the manifest.",
    )
    parser.add_argument("--split", default="dev", help="Optional split filter (default: dev). Use 'all' for no filter.")
    parser.add_argument(
        "--slices",
        default=None,
        help="Comma-separated slice filter. Defaults to all slices found in the manifest.",
    )
    parser.add_argument("--samples-per-slice", type=int, default=8, help="Rows to render per slice.")
    parser.add_argument("--frames-per-video", type=int, default=4, help="Frames to render per selected row.")
    args = parser.parse_args()

    manifest_path = args.manifest
    manifest_stem = Path(manifest_path).stem
    output_dir = args.output_dir or str(Path("DeepfakeBench/training/arena/visual_audits") / manifest_stem)
    split = None if args.split.strip().lower() == "all" else args.split.strip()
    slices = [item.strip() for item in args.slices.split(",") if item.strip()] if args.slices else None

    result = build_gallery(
        manifest_path=manifest_path,
        output_dir=output_dir,
        split=split,
        slices=slices,
        samples_per_slice=args.samples_per_slice,
        frames_per_video=args.frames_per_video,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
