import tempfile

from ..loader.types import LoadedPDF
from .base import PDFtoMarkdown


class MarkItDownConverter(PDFtoMarkdown):
    def convert(self, doc: LoadedPDF) -> str:  # type: ignore[override]
        # Minimal, no post-processing or heuristics
        from markitdown import MarkItDown  # optional dependency; import lazily

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(doc.raw_bytes)
            tmp.flush()

            md = MarkItDown(enable_plugins=False)
            result = md.convert(tmp.name)
        return result.text_content




