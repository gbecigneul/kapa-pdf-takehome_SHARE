import tempfile
import os
import re
import json
from pathlib import Path
from typing import List

from ..loader.types import LoadedPDF
from .base import PDFtoMarkdown


class UnstructuredConverter(PDFtoMarkdown):
    def convert(self, doc: LoadedPDF) -> str:  # type: ignore[override]
        # Minimal integration using Unstructured's partition API.
        # No heuristic post-processing.
        try:
            from unstructured.partition.pdf import partition_pdf  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Unstructured is not installed. Install and system deps, then restart the app:\n"
                "pip install \"unstructured[pdf]\"\n"
                "brew install poppler tesseract\n"
                "Docs: https://github.com/Unstructured-IO/unstructured"
            ) from exc

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(doc.raw_bytes)
            tmp.flush()

            # Prefer hi_res for better structure; allow override via env
            strategy = os.getenv("UNSTRUCTURED_PDF_STRATEGY", "hi_res")
            include_meta = True
            elements = partition_pdf(filename=tmp.name, strategy=strategy, include_metadata=include_meta)

        def to_markdown() -> str:
            # Allow configuring heading level for Unstructured Title elements
            try:
                title_level = int(os.getenv("UNSTRUCTURED_TITLE_LEVEL", "2"))
            except ValueError:
                title_level = 2
            if title_level < 1 or title_level > 6:
                title_level = 2
            title_prefix = "#" * title_level + " "

            lines: List[str] = []
            for el in elements:
                # Prefer robust access via attributes with fallbacks
                text = getattr(el, "text", "") or str(el)
                cls = el.__class__.__name__
                # Inspect metadata for HTML with heading tags to determine depth
                level_from_html = None
                meta = getattr(el, "metadata", None)
                if meta is not None and hasattr(meta, "to_dict"):
                    mdict = meta.to_dict()
                    html = mdict.get("text_as_html")
                    if isinstance(html, str):
                        m = re.search(r"<h([1-6])[^>]*>", html, re.IGNORECASE)
                        if m:
                            try:
                                level_from_html = int(m.group(1))
                            except Exception:
                                level_from_html = None

                if cls == "Title" or level_from_html is not None:
                    cleaned = re.sub(r"\s+", " ", text.strip())
                    if level_from_html is None:
                        # Fallback when no explicit level is present
                        level = title_level
                    else:
                        level = level_from_html
                    level = min(max(level, 1), 6)
                    lines.append(f"{'#' * level} {cleaned}")
                elif cls == "ListItem":
                    lines.append(f"- {text.strip()}")
                elif cls == "PageBreak":
                    lines.append("\n---\n")
                elif cls == "Table":
                    # Preserve table content without attempting to reformat
                    lines.append("```table\n" + text.strip() + "\n```")
                else:
                    lines.append(text.strip())
            return "\n\n".join([ln for ln in lines if ln is not None])

        # Write raw Unstructured output for inspection (before Markdown conversion)
        try:
            log_dir = Path("data") / "md_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            safe_name = doc.name.replace("/", "_").replace("\\", "_")

            raw_entries = []
            for el in elements:
                meta = getattr(el, "metadata", None)
                meta_dict = None
                if meta is not None and hasattr(meta, "to_dict"):
                    try:
                        meta_dict = meta.to_dict()
                    except Exception:
                        meta_dict = None
                raw_entries.append(
                    {
                        "type": el.__class__.__name__,
                        "text": getattr(el, "text", None) or str(el),
                        "metadata": meta_dict,
                    }
                )
            (log_dir / f"{safe_name}.unstructured.raw.json").write_text(
                json.dumps(raw_entries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # Also write a quick text outline
            outline_lines: List[str] = []
            for i, el in enumerate(elements):
                outline_lines.append(
                    f"[{i:04d}] {el.__class__.__name__}: {(getattr(el, 'text', '') or '').strip()}"
                )
            (log_dir / f"{safe_name}.unstructured.raw.txt").write_text(
                "\n\n".join(outline_lines), encoding="utf-8"
            )
        except Exception:
            pass

        return to_markdown()


