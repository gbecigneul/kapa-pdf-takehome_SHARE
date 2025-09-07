import io
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Union

import pymupdf
import pymupdf4llm
from google.genai import types

from ..loader.types import LoadedPDF
from .base import PDFtoMarkdown
from .exceptions import EmptyConversionError, UnexpectedStopError
from .google_client import GeminiModel, GoogleAiClient
from .openai_client import OpenAIClient
from .types import InMemoryPDF, PdfConversionJobMetadata, PdfPageRange
from .utils import num_tokens_from_string


class GoogleAiClientBuilder:
    """
    A builder class to create instances of GoogleAiClient.
    """

    def build(self) -> GoogleAiClient:
        return GoogleAiClient()


class OpenAIClientBuilder:
    """
    A builder class to create instances of OpenAIClient.
    """

    def build(self) -> OpenAIClient:
        return OpenAIClient(max_retries=0, timeout=180)


class GeminiConverter(PDFtoMarkdown):
    system_instruction = "You are an expert at analyzing PDF documents and converting them into clean, valid, well-formatted Markdown."
    initial_prompt = """\
You are provided a PDF file; transcribe it into valid Markdown format following the instructions below.
Text handling:
    - Create a single root ATX-style header (#) that's specific and informative so that it provides context for the entire document. Nest all other headers under this root header.
    - Ignore text emphasis in markdown headers
    - Ignore figures and images in the document.
    - Ignore conversion of Table of Contents, list of figures or list of tables, but you may use the Table of Contents to help identify the main sections and hierarchy of the document.
    - Ignore page headers and page footers.
    - Ignore page numbers
    - Ignore any watermarks, logos, or decorative elements
    - Ignore copyright, license, and disclaimer text

Table handling:
    - Use basic, minimal GFM table syntax with leading and trailing pipes (|) and do NOT include any extra whitespace or tabs for alignment.
    - For cells that span multiple rows, create a new row for each cell combination, duplicating the content as needed
    - If a table is split across multiple PDF pages, merge them into one
    - If the PDF contains a nested table, break it up into multiple valid tables, each with its own header row

Special guidelines:
    - Pay special attention to use ATX-style markdown headers (ie: H2, H3, etc.) when doing the conversion.
    - Don't attempt to summarize the document. Preserve the original content to the best of your ability.
    - Do not surround your output with triple backticks
    - Remember to keep the markdown tables minimal and simple, while duplicating cells across rows as needed if a cell spans multiple rows.
    - IMPORTANT: Use the full range of your token output capacity if needed.
"""
    subsequent_prompt = """\
You are provided a PDF file; transcribe it into valid Markdown format following the instructions below.
Text handling:
    - IMPORTANT: Pay special attention to use ATX-style markdown headers (ie: H2, H3, etc.) when doing the conversion. All generated ATX-headers MUST be depth 2 (##) or greater (###, ####, etc.)
    - Ignore text emphasis in markdown headers
    - Ignore figures and images in the document.
    - Ignore page headers and page footers.
    - Ignore page numbers
    - Ignore any watermarks, logos, or decorative elements
    - Ignore copyright, license, and disclaimer text

Table handling:
    - Use basic, minimal GFM table syntax with leading and trailing pipes (|) and do NOT include any extra whitespace or tabs for alignment.
    - For cells that span multiple rows, create a new row for each cell combination, duplicating the content as needed
    - If a table is split across multiple PDF pages, merge them into one.
    - If the PDF contains a nested table, break it up into multiple valid tables, each with its own header row

Special instructions:
    - Pay special attention to use ATX-style markdown headers (ie: H2, H3, etc.) when doing the conversion.
    - Don't attempt to summarize the document. Preserve the original content to the best of your ability.
    - Do not surround your output with triple backticks.
    - Remember to keep the markdown tables minimal and simple, while duplicating cells across rows as needed if a cell spans multiple rows.
    - IMPORTANT: Use the full range of your token output capacity if needed.
"""
    PRIMARY_MODEL = GeminiModel.GEMINI_2_0_FLASH

    # Safety buffer for token estimation so we don't hit Gemini's output limit
    # This number has been tuned through trial and error to be conservative enough to avoid hitting the limit
    # but not so conservative that it causes unnecessary batch splits. It can still hit this limit in some cases
    # if PDF elements are very verbose to represent property in markdown format. This number comes from trial and error.
    TOKEN_LIMIT_BUFFER_FACTOR = 0.9
    MAX_CONCURRENT_WORKERS = 50  # TODO: validate this number
    MAX_BATCH_SIZE = 25  # Up to 25 distinct PDFs in a batch, inclusive

    def __init__(
        self,
        google_client_builder: Optional[GoogleAiClientBuilder] = None,
        openai_client_builder: Optional[OpenAIClientBuilder] = None,
    ):
        super().__init__()
        self.google_client_builder = google_client_builder or GoogleAiClientBuilder()
        self.openai_client_builder = openai_client_builder or OpenAIClientBuilder()

    def convert(self, doc: LoadedPDF) -> str:
        """
        Convert a local PDF file to Markdown format using the Gemini API. Convenience method for a single PDF.

        Args:
            pdf_data: The data of the PDF file to be converted.
        Returns:
            A string with the converted Markdown content.
        Raises:
            FileNotFoundError: If the PDF file does not exist.
            Exception: If a fatal error occurs during conversion.
        """

        pdf = InMemoryPDF(filename=doc.name, data=doc.raw_bytes)
        result = self.convert_batch(pdfs=[pdf])

        if not result:
            raise EmptyConversionError(
                f"Empty conversion result for {doc.name}. No content returned."
            )

        return result[0]

    def convert_batch(self, pdfs: list[InMemoryPDF]) -> list[str]:
        """
        Convert multiple PDFs in parallel to Markdown format. Returns a list of Markdown strings where the index corresponds to the input PDF.
        Note that if a fatal error occurs during extraction or conversion for a single PDF, that PDF will be skipped and an empty string will be returned at its index.
        The other PDFs will still be processed.

        Args:
            pdfs: A list of PDFs in-memory to be converted
        Returns:
            A list of strings with the converted Markdown content, where the index corresponds to the input PDF.
            An empty string will be returned for any PDFs that failed to convert.
        """
        if len(pdfs) == 0:
            raise ValueError("Empty list of PDFs provided for batch conversion.")

        # pre-extract per-PDF data serially up-front, as PyMu doesn't allow for use in multi-threaded environments.
        # Flatten all PDFs and their pieces into a single list of jobs for the thread pool to process.
        jobs: list[PdfConversionJobMetadata] = []
        for pdf_batch_idx, pdf in enumerate(pdfs):
            name = pdf.filename
            data = pdf.data
            try:
                doc = pymupdf.open(stream=data, filetype="pdf")
            except Exception as e:
                # If this is the only PDF in the batch, re-raise the error
                if len(pdfs) == 1:
                    raise e
                else:
                    # if batch, skip this PDF
                    logging.warning(
                        f"Skipping {name}: could not open PDF due to: ({e})"
                    )
                    continue

            try:
                page_info = pymupdf4llm.to_markdown(doc, page_chunks=True)

                # determine how to split the PDF into pieces for processing, then form the splits
                page_ranges = self._get_page_ranges(page_info)
                if len(page_ranges) == 1:
                    page_range: tuple[int, int] = page_ranges[0]
                    page_range_data: bytes = doc.tobytes()

                    jobs.append(
                        PdfConversionJobMetadata(
                            filename=name,
                            page_info=page_info,
                            pdf_batch_idx=pdf_batch_idx,
                            page_range=PdfPageRange(
                                start_idx=0,
                                end_idx=page_range[1],
                                data=page_range_data,
                            ),
                            page_range_idx=0,
                        )
                    )
                else:
                    for page_range_idx, (start, end) in enumerate(page_ranges):
                        with pymupdf.open() as tmp:
                            # create a temporary PDF with the specified page range
                            tmp.insert_pdf(doc, from_page=start, to_page=end)

                            # extract the bytes from the temporary PDF
                            page_range_data: bytes
                            with io.BytesIO() as buf:
                                tmp.save(buf)
                                buf.seek(0)
                                page_range_data = buf.getvalue()

                        jobs.append(
                            PdfConversionJobMetadata(
                                filename=name,
                                page_info=page_info,
                                pdf_batch_idx=pdf_batch_idx,
                                page_range=PdfPageRange(
                                    start_idx=start,
                                    end_idx=end,
                                    data=page_range_data,
                                ),
                                page_range_idx=page_range_idx,
                            )
                        )
            except Exception as e:
                if len(pdfs) == 1:
                    # Re-raise error if this is only PDF
                    raise e
                else:
                    # if batch, skip this PDF
                    logging.warning(
                        f"Skipping {name}: could not extract text from PDF due to: ({e})"
                    )
                    continue
            finally:
                doc.close()

        # single ThreadPoolExecutor for all LLM batch conversions
        results: dict[tuple[int, int], str] = {}
        max_workers = min(len(jobs), self.MAX_CONCURRENT_WORKERS)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # submit all jobs to the thread pool, and associate each one with its
            # corresponding PDF batch index and page range index for reassembly
            futures = {
                pool.submit(self._execute_pdf_job, job): (
                    job.pdf_batch_idx,
                    job.page_range_idx,
                )
                for job in jobs
            }

            for future in as_completed(futures):
                batch_idx, page_range_idx = futures[future]
                markdown = future.result()
                results[(batch_idx, page_range_idx)] = markdown

        # group the results by PDF
        grouped: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for (batch_idx, page_idx), md in results.items():
            grouped[batch_idx].append((page_idx, md))

        # reassemble the markdown for each PDF
        # IMPORTANT: critically, we iterate over the original len of the PDF input.
        # This means that if a PDF throws a fatal exception during extraction / conversion and was skipped,
        # we'll have no markdown data for it (at its index) and it will return an empty string for its index position.
        markdowns: list[str] = []
        for pdf_idx in range(len(pdfs)):
            # sort the pieces by page_range_idx and concatenate them
            md_parts = [
                md for _, md in sorted(grouped.get(pdf_idx, []), key=lambda x: x[0])
            ]
            pdf_markdown = "\n".join(md_parts)
            markdowns.append(pdf_markdown)

        return markdowns

    def _get_page_ranges(
        self, page_info: list[dict[str, str]]
    ) -> list[tuple[int, int]]:
        """
        Analyze page markdown to form the list of page ranges used to split and process a PDF in parallel.
        Page ranges formed are based on estimated token count of PDF's extracted text layer.

        Args:
            page_markdown_map: A dictionary mapping page numbers to their markdown content.

        Returns:
            A list of tuples, each representing a range of pages (start, end) for each batch.
            Pages are 0-indexed and ranges are inclusive.
            Ex: [(0, 2), (3, 5), (6, 7)]
        """
        # max number of pages in a batch that is reasonable given Gemini's limits, from trial and error
        # this is important so that if the PDF doesn't have embedded text, we have an upper limit
        MAX_PAGE_RANGE_SIZE = 15
        batches: list[tuple[int, int]] = []
        num_pages = len(page_info)
        if num_pages == 0:
            return []

        # Calculate token limit with buffer
        token_limit = (
            self.PRIMARY_MODEL.get_output_token_limit() * self.TOKEN_LIMIT_BUFFER_FACTOR
        )

        batch_start = 0
        current_batch_tokens = 0.0

        for page_idx in range(num_pages):
            page_markdown = page_info[page_idx].get("text", "")
            page_tokens = self._estimate_tokens_of_page(page_markdown)

            # Check if the current page *alone* exceeds the limit. Unlikely but sanity check.
            # Note: we fallback from gemini-flash-2.0 to gpt-4.1-mini, the later which has a much larger token output limit (~64k, 8x bigger) so if
            # Gemini returns a size-related error, we'll handle it
            if page_tokens > token_limit:
                logging.warning(
                    f"Page {page_idx} estimated tokens ({page_tokens}) exceeds batch limit ({token_limit}). "
                    f"Including it in its own potentially oversized batch."
                )
                # If a previous batch was being built, finalize it
                if page_idx > batch_start:
                    batches.append((batch_start, page_idx - 1))

                # Add the oversized page as its own batch
                batches.append((page_idx, page_idx))
                # Start next batch from the next page
                batch_start = page_idx + 1
                current_batch_tokens = 0.0
                continue  # Move to the next page

            # Check if adding this page exceeds the token limit or max pages
            would_exceed_token_limit = (
                current_batch_tokens + page_tokens
            ) > token_limit and page_idx > batch_start
            would_exceed_max_page_size = (
                page_idx - batch_start + 1
            ) > MAX_PAGE_RANGE_SIZE

            if would_exceed_token_limit or would_exceed_max_page_size:
                # Finalize the current batch (up to the previous page)
                batches.append((batch_start, page_idx - 1))

                # Start a new batch with the current page
                batch_start = page_idx
                current_batch_tokens = page_tokens
            else:
                # Accumulate page into the current batch
                current_batch_tokens += page_tokens

        # Add the last batch if there are remaining pages
        if batch_start < num_pages:
            batches.append((batch_start, num_pages - 1))

        # Sanity check for empty batches list if num_pages > 0
        if num_pages > 0 and not batches:
            logging.warning(
                "Batch calculation resulted in empty batch list for non-empty PDF. Creating single batch."
            )
            batches.append((0, num_pages - 1))
        return batches

    def _execute_pdf_job(self, job: PdfConversionJobMetadata) -> str:
        """
        Convert a PDF batch job to Markdown. Returns the markdown string.
        """
        pdf_name = job.filename
        page_range_idx = job.page_range_idx
        start, end = job.page_range.start_idx, job.page_range.end_idx
        pdf_bytes = job.page_range.data
        page_info = job.page_info

        # Try Gemini first
        try:
            pdf_part = types.Part.from_bytes(
                data=pdf_bytes, mime_type="application/pdf"
            )
            prompt = (
                self.initial_prompt if page_range_idx == 0 else self.subsequent_prompt
            )

            # Instantiate a new client instance for each batch to avoid state issues
            # in a multi-threaded environment. The Google GenAi docs don't explicitly state that the client is thread-safe.
            # Like this, we can be sure that each thread has its own isolated client instance.
            client = self.google_client_builder.build()

            logging.debug(f"Sending batch {page_range_idx} to Gemini API...")

            # Convert PDF to markdown via Gemini API
            # NOTE: Gemini flash 2.0 has a strange bug where it chokes when generating tables when the temperature is set to 0. Careful.
            # We explicitly don't set a temperature and this avoids the bug. See more:
            # https://discuss.ai.google.dev/t/gemini-2-0-not-completing-responses/64886
            # https://discuss.ai.google.dev/t/gemini-2-0-flash-has-a-weird-bug/65119/20
            response = client.generate_content(
                model=self.PRIMARY_MODEL,
                contents=[pdf_part, prompt],
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    temperature=None,
                    http_options=types.HttpOptions(
                        timeout=120 * 1000
                    ),  # 120s, API expects milliseconds
                ),
            )

            finish_reason = response.candidates[-1].finish_reason
            if finish_reason not in (
                types.FinishReason.FINISH_REASON_UNSPECIFIED,
                types.FinishReason.STOP,
            ):
                raise UnexpectedStopError(
                    f"Gemini stopped unexpectedly for {pdf_name}, batch {page_range_idx} and page_range: {(start, end)} for reason: {finish_reason}"
                )

            text = response.text.strip() if response.text else ""
            text = self._strip_outer_fence(text)
            if not text:
                raise EmptyConversionError(
                    f"No content returned from Gemini for {pdf_name}, batch {page_range_idx}"
                )

            return text

        except Exception as e_gemini:
            logging.warning(f"Gemini failed for batch {page_range_idx}: {e_gemini}")
            # Fallback 1: OpenAI
            try:
                logging.debug(
                    f"Attempting OpenAI fallback for batch {page_range_idx}..."
                )
                prompt = (
                    self.initial_prompt
                    if page_range_idx == 0
                    else self.subsequent_prompt
                )

                # Instantiate a new client instance for each batch to avoid state issues
                # in a multi-threaded environment. The Google GenAi docs don't explicitly state that the client is thread-safe.
                # Like this, we can be sure that each thread has its own client instance.
                openai_client = self.openai_client_builder.build()
                result = openai_client.convert_pdf(
                    pdf_data=pdf_bytes, prompt=prompt, filename=pdf_name
                )
                markdown = result.get("text", "").strip()
                markdown = self._strip_outer_fence(markdown)

                if not markdown:
                    raise EmptyConversionError(
                        f"OpenAI returned no content for {pdf_name}, batch {page_range_idx}"
                    )

                return markdown

            except Exception as e_oi:
                logging.warning(
                    f"OpenAI fallback failed for batch {page_range_idx}: {e_oi}"
                )
                # Fallback 2: PyMuPDF markdown
                logging.debug(
                    f"Attempting PyMuPDF fallback for batch {page_range_idx}..."
                )

                pymu_markdown = "\n".join(
                    page_info[p].get("text", "") for p in range(start, end + 1)
                ).strip()

                return pymu_markdown

    def _estimate_tokens_of_page(self, page_markdown: str) -> float:
        """
        Estimate the number of tokens for a resulting page's markdown using tiktoken (or similar) and a conversion factor for Gemini.
        """
        has_table = bool(re.search(r"\|.*\|", page_markdown))

        if has_table:
            # Gemini tables are more verbose and duplicate cells across rows if needed, so we need more token overhead
            CONVERSION_FACTOR = 1.4
        else:
            CONVERSION_FACTOR = 1.15

        # Add a token overhead multiplier, which accounts for an increase in tokens for the real markdown format from Gemini vs. PyMuPDF
        # (ei: Gemini might add more tokens for table row duplication, formatting, etc.)
        # and the difference between how tiktoken and Gemini count tokens. This number comes from trial and error.
        return num_tokens_from_string(page_markdown) * CONVERSION_FACTOR

    def _strip_outer_fence(self, s: str) -> str:
        """
        Attempt to strip the outer fence from a string, if it exists. Preserves the inner content, including any code blocks inside the string.

        This is useful for cleaning up the output from the LLM, which may include triple backticks (```) at the beginning and end of the string.
        Gemini Flash 2.0 often wraps its output in:

        ```markdown
        ...

        ```

        fences despite us querying against this behavior, so we need to strip them out.
        """
        # matches an opening ``` plus optional label (up to the first newline), greedily captures EVERYTHING (including any ``` inside),
        # then the very last closing ```.
        pattern = re.compile(r"\A```[^\n]*\n([\s\S]*)\n```$")
        s_stripped = s.strip("\n")
        m = pattern.match(s_stripped)  # try to match an outer fence
        return m.group(1) if m else s
