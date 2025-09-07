from dataclasses import dataclass


@dataclass(frozen=True)
class InMemoryPDF:
    """
    A class to represent an in-memory PDF document.
    Attributes:
        filename: The name of the PDF file.
        data: The byte content of the PDF.
    """

    filename: str
    data: bytes


@dataclass(frozen=True)
class PdfPageRange:
    """
    A class to represent a range of pages in a PDF document.
    Attributes:
        start_idx: The starting index of the page range (inclusive).
        end_idx: The ending index of the page range (inclusive). If start_idx == end_idx, this represents a single page.
        data: The byte content of the PDF in this page range.
    """

    start_idx: int
    end_idx: int
    data: bytes

    def __post_init__(self):
        if self.start_idx > self.end_idx:
            raise ValueError(
                f"Invalid page range: start index {self.start_idx} is greater than end index {self.end_idx}"
            )


@dataclass(frozen=True)
class PdfConversionJobMetadata:
    """
    Class to hold the data needed to process a single PDF conversion job.
    """

    filename: str

    # the page data extracted from PyMuPDF for this PDF. Each index of the list is a page and each page has an info dict.
    page_info: list[dict[str, str]]

    # the index of the PDF in the batch provided during batch conversion
    pdf_batch_idx: int

    # the range of pages + data to process in this job
    page_range: PdfPageRange

    # the index of the page range in the PDF.
    # When PDFs are being converted, we split them into pieces. This var is the index of this specific piece, which we use for re-assembling the final markdown results.
    # See `PdfMarkdownConverter._get_page_ranges` for more details on the list of page ranges is calculated.
    page_range_idx: int
