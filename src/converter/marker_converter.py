import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..loader.types import LoadedPDF
from .base import PDFtoMarkdown


class MarkerConverter(PDFtoMarkdown):
    def convert(self, doc: LoadedPDF) -> str:  # type: ignore[override]
        """
        Convert PDF to Markdown using the marker-pdf CLI.

        Supports two variants:
        - marker_single <file> --output_dir <dir>
        - marker <input_dir> --output_dir <dir>
        """

        marker_single_exec = shutil.which("marker_single")
        marker_exec = shutil.which("marker")

        if marker_single_exec is None and marker_exec is None:
            raise RuntimeError(
                "Neither 'marker_single' nor 'marker' CLI found. Install with `pip install marker-pdf`."
            )

        # Prepare temp input and output directories
        with tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
            in_dir_path = Path(in_dir)
            out_dir_path = Path(out_dir)

            # Write PDF to input directory using a stable, user-facing name
            base_stem = Path(doc.name).stem or "document"
            in_pdf_path = in_dir_path / f"{base_stem}.pdf"
            in_pdf_path.write_bytes(doc.raw_bytes)

            # Build command depending on available CLI
            if marker_single_exec is not None:
                cmd = [marker_single_exec, str(in_pdf_path), "--output_dir", str(out_dir_path)]
            else:
                # marker expects an input directory
                cmd = [marker_exec, str(in_dir_path), "--output_dir", str(out_dir_path)]  # type: ignore[arg-type]

            res = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=1200,
            )
            if res.returncode != 0:
                raise RuntimeError(f"marker failed (code {res.returncode}): {res.stderr.strip()}")

            # Find a produced Markdown file; prefer matching stem, else first .md
            candidates = list(out_dir_path.rglob("*.md"))
            preferred = [p for p in candidates if p.stem == base_stem]
            chosen = preferred[0] if preferred else (candidates[0] if candidates else None)
            if chosen is None:
                raise RuntimeError("marker ran but no Markdown was produced in the output directory")

            return chosen.read_text(encoding="utf-8", errors="ignore")


