import tempfile
import re
from pathlib import Path
import os

from ..loader.types import LoadedPDF
from .base import PDFtoMarkdown

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode #, VlmPipelineOptions

pipeline_options = PdfPipelineOptions(do_table_structure=True)

pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE  # use more accurate TableFormer model
pipeline_options.do_ocr = False
pipeline_options.table_structure_options.do_cell_matching = False

# from docling.pipeline.vlm_pipeline import VlmPipeline

class DoclingConverter(PDFtoMarkdown):
    def convert(self, doc: LoadedPDF) -> str:  # type: ignore[override]
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(doc.raw_bytes)
            tmp.flush()
            
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                        # pipeline_cls=VlmPipeline
                    )
                }
            )
            result = converter.convert(tmp.name)
        md_pre = result.document.export_to_markdown()

        # Group contiguous sections by the most frequent heading level Docling emits
        # (e.g., '# ' or '## '). Inside groups, convert any heading lines to bold text
        # so the chunker sees only a single top heading per group.
        def _detect_base_heading_level(md: str) -> int | None:
            counts = {
                1: len(re.findall(r'(?m)^#\s+', md)),
                2: len(re.findall(r'(?m)^##\s+', md)),
                3: len(re.findall(r'(?m)^###\s+', md)),
            }
            level, cnt = max(counts.items(), key=lambda x: x[1])
            return level if cnt > 0 else None

        def wrap_docling_md(md: str, max_chars: int = 8000, overlap: int = 1) -> str:
            base = _detect_base_heading_level(md)
            if base is None:
                return md.rstrip() + "\n"

            pattern = rf'(?m)^#{{{base}}}\s+.*$'
            matches = list(re.finditer(pattern, md))
            sections: list[tuple[str, str]] = []
            if not matches:
                sections = [("Document", md.rstrip() + "\n")]
            else:
                if matches[0].start() > 0:
                    pre = md[:matches[0].start()].strip()
                    if pre:
                        sections.append(("Preamble", pre + "\n"))
                for i, m in enumerate(matches):
                    start = m.start()
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
                    block = md[start:end].rstrip() + "\n"
                    title = re.sub(rf'^#{{{base}}}\s+', '', m.group()).strip()
                    # Preserve the leading base-level heading line as plain text (no '#')
                    block = re.sub(rf'(?m)^#{{{base}}}\s+(.*)$', r'\1', block, count=1)
                    # Convert any remaining headings (any level) to plain text (keep content)
                    block = re.sub(r'(?m)^#{1,6}\s+(.*)$', r'\1', block)
                    sections.append((title, block))

            def render(cand: list[tuple[str, str]]) -> str:
                heading = cand[0][0] if cand else "Chunk"
                body_parts = []
                for t, b in cand:
                    if b:
                        body_parts.append(f"**{t}**\n\n{b}")
                body = "\n\n".join(body_parts)
                return f"# {heading}\n\n" + body

            chunks: list[str] = []
            i, n = 0, len(sections)
            # dynamic overlap selection: repeat as many prior sections as possible
            # such that the repeated text is <= max_repeat_pct of a chunk.
            max_repeat_pct = float(os.getenv("DOCLING_WRAP_MAX_REPEAT_PCT", "0.50"))
            max_repeat_sections_cap = int(os.getenv("DOCLING_WRAP_MAX_REPEAT_SECTIONS", "8"))
            while i < n:
                if chunks and i > 0:
                    allowed = int(max_chars * max_repeat_pct)
                    best_overlap = 0
                    max_try = min(i, max_repeat_sections_cap)
                    for c in range(max_try, 0, -1):
                        # Render only the would-be repeated prefix
                        rep_text = render(sections[i - c:i])
                        if len(rep_text) <= allowed:
                            best_overlap = c
                            break
                    if best_overlap == 0:
                        best_overlap = 1
                    start = i - best_overlap
                else:
                    start = i
                j = i
                best = render(sections[start:j + 1])
                while j + 1 < n:
                    nxt = render(sections[start:j + 2])
                    if len(nxt) <= max_chars:
                        j += 1
                        best = nxt
                    else:
                        break
                chunks.append(best)
                i = j + 1
            return ("\n\n".join(chunks)).rstrip() + "\n"

        wrap_max = int(os.getenv("DOCLING_WRAP_MAX_CHARS", "4800"))
        wrap_overlap = int(os.getenv("DOCLING_WRAP_OVERLAP", os.getenv("DOCLING_WRAP_OVERLAP_SECTIONS", "1")))
        md = wrap_docling_md(md_pre, max_chars=wrap_max, overlap=wrap_overlap)



        # Write debug logs to data/md_logs/<filename>.docling.{pre,post}.md
        try:
            log_dir = Path("data") / "md_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            safe_name = doc.name.replace("/", "_").replace("\\", "_")
            (log_dir / f"{safe_name}.docling.pre.md").write_text(md_pre, encoding="utf-8")
            (log_dir / f"{safe_name}.docling.post.md").write_text(md, encoding="utf-8")
        except Exception:
            # Best-effort logging; ignore filesystem errors
            pass
        return md


