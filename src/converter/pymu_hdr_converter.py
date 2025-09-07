from __future__ import annotations

from typing import Callable, Optional, Literal, List, Tuple
from pathlib import Path
import json

import pymupdf
import pymupdf4llm

from ..loader.types import LoadedPDF
from .base import PDFtoMarkdown


HeaderMode = Literal["identify", "font_size", "toc", "custom", "auto"]


class PymuHeaderConverter(PDFtoMarkdown):
    def __init__(
        self,
        mode: HeaderMode = "identify",
        *,
        max_levels: int = 3,
        size_h1: float = 14.0,
        size_h2: float = 10.0,
        custom_hdr: Optional[Callable[[dict, Optional[pymupdf.Page]], str]] = None,
        page_chunks: bool = False,
        debug: bool = False,
        debug_output_dir: Optional[Path] = None,
    ):
        self.mode = mode
        self.max_levels = max_levels
        self.size_h1 = size_h1
        self.size_h2 = size_h2
        self.custom_hdr = custom_hdr
        self.page_chunks = page_chunks
        self.debug = debug
        self.debug_output_dir = debug_output_dir

    def _build_hdr_info(self, pdf: pymupdf.Document):
        if self.mode == "auto":
            toc = pdf.get_toc()
            if toc:
                if self.debug:
                    print("[pymu-hdr][auto] Using TOC-based headings (toc entries:", len(toc), ")")
                def hdr_func(span: dict, page: Optional[pymupdf.Page] = None) -> str:
                    if page is None:
                        return ""
                    page_no = page.number + 1
                    toc_here = [t for t in toc if t and t[-1] == page_no]
                    if not toc_here:
                        return ""
                    text = span.get("text", "")
                    chosen = ""
                    for lvl, title, _ in toc_here:
                        if text.startswith(title) or title.startswith(text):
                            chosen = "#" * int(lvl) + " "
                            break
                    if self.debug:
                        print(
                            "[pymu-hdr][auto->toc]",
                            {"page": getattr(page, "number", None), "text": text[:80], "level": chosen.strip()},
                        )
                    return chosen
                return hdr_func
            else:
                if self.debug:
                    print("[pymu-hdr][auto] No TOC found; using font_size with thresholds",
                          {"size_h1": self.size_h1, "size_h2": self.size_h2})
                size_h1 = self.size_h1
                size_h2 = self.size_h2
                def hdr_func(span: dict, page: Optional[pymupdf.Page] = None) -> str:
                    sz = span.get("size", 0)
                    level = ""
                    if sz > size_h1:
                        level = "# "
                    elif sz > size_h2:
                        level = "## "
                    if self.debug:
                        print(
                            "[pymu-hdr][auto->font_size]",
                            {"page": getattr(page, "number", None), "size": sz, "text": span.get("text", "")[:80], "level": level.strip()},
                        )
                    return level
                return hdr_func
        if self.mode == "identify":
            hdr = pymupdf4llm.IdentifyHeaders(pdf, max_levels=self.max_levels)
            if self.debug:
                try:
                    # Attempt to log header thresholds or info if available
                    info = getattr(hdr, "header_info", None)
                    if info is not None:
                        print("[pymu-hdr][identify] header_info:", info)
                except Exception:
                    pass
            return hdr

        if self.mode == "font_size":
            size_h1 = self.size_h1
            size_h2 = self.size_h2

            def hdr_func(span: dict, page: Optional[pymupdf.Page] = None) -> str:
                sz = span.get("size", 0)
                level = ""
                if sz > size_h1:
                    level = "# "
                elif sz > size_h2:
                    level = "## "
                if self.debug:
                    print(
                        "[pymu-hdr][font_size]",
                        {"page": getattr(page, "number", None), "size": sz, "text": span.get("text", "")[:80], "level": level.strip()},
                    )
                return level

            return hdr_func

        if self.mode == "toc":
            toc = pdf.get_toc()

            def hdr_func(span: dict, page: Optional[pymupdf.Page] = None) -> str:
                if page is None:
                    return ""
                page_no = page.number + 1
                toc_here = [t for t in toc if t and t[-1] == page_no]
                if not toc_here:
                    return ""
                text = span.get("text", "")
                chosen = ""
                for lvl, title, _ in toc_here:
                    if text.startswith(title) or title.startswith(text):
                        chosen = "#" * int(lvl) + " "
                        break
                if self.debug:
                    print(
                        "[pymu-hdr][toc]",
                        {"page": getattr(page, "number", None), "text": text[:80], "toc_here": [(int(l), t) for l, t, _ in toc_here], "level": chosen.strip()},
                    )
                return chosen

            return hdr_func

        if self.mode == "custom":
            if self.custom_hdr is None:
                raise ValueError("custom_hdr must be provided when mode='custom'")
            return self.custom_hdr

        raise ValueError(f"Unknown mode: {self.mode!r}")

    def convert(self, doc: LoadedPDF) -> str:  # type: ignore[override]
        pdf = pymupdf.open(stream=doc.raw_bytes, filetype="pdf")
        hdr_info = self._build_hdr_info(pdf)
        log_entries: List[dict] = []

        # If hdr_info is a callable, wrap it to capture decisions
        if callable(hdr_info) and self.debug:
            underlying = hdr_info

            def logging_wrapper(span: dict, page: Optional[pymupdf.Page] = None) -> str:  # type: ignore[no-redef]
                level = underlying(span, page)
                try:
                    log_entries.append(
                        {
                            "page": getattr(page, "number", None),
                            "text": span.get("text", ""),
                            "size": span.get("size", None),
                            "level": level.strip(),
                        }
                    )
                except Exception:
                    pass
                return level

            hdr_info = logging_wrapper

        if self.debug:
            print(
                "[pymu-hdr] convert",
                {"doc": doc.name, "mode": self.mode, "max_levels": self.max_levels, "size_h1": self.size_h1, "size_h2": self.size_h2, "page_chunks": self.page_chunks},
            )

        markdown = pymupdf4llm.to_markdown(pdf, hdr_info=hdr_info, page_chunks=self.page_chunks)

        # Persist logs if requested
        if self.debug and self.debug_output_dir is not None:
            try:
                out_dir = Path(self.debug_output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                base = f"{doc.name}.pymu_hdr.{self.mode}"
                meta_path = out_dir / f"{base}.meta.json"
                log_path = out_dir / f"{base}.log.jsonl"
                meta = {
                    "doc": doc.name,
                    "mode": self.mode,
                    "max_levels": self.max_levels,
                    "size_h1": self.size_h1,
                    "size_h2": self.size_h2,
                    "page_chunks": self.page_chunks,
                }
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                if log_entries:
                    with log_path.open("w", encoding="utf-8") as f:
                        for row in log_entries:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception:
                pass

        return markdown


