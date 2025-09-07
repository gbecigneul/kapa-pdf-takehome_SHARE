import io
import logging
import os
import pathlib
from enum import StrEnum
from typing import Optional, Union

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiModel(StrEnum):
    """
    Supported Gemini models
    """

    GEMINI_2_0_FLASH = "gemini-2.0-flash"

    def get_output_token_limit(self) -> int:
        """
        Get the output token limit for the model
        """
        match self:
            case self.GEMINI_2_0_FLASH:
                return 8192
            case _:
                raise ValueError(f"Unknown or unsupported model: {self}")


class GoogleAiClient:
    """
    Class for making calls to Google LLMs (Gemini)
    """

    # As of April 9th 2025, the Gemini APIs have a 20 MB limit for processing files in-memory.
    # Any larger, and the file first must be uploaded to the Google Files API before processing w/ Gemini.
    FILE_MB_THRESHOLD = 20

    def __init__(self):
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def count_tokens(
        self, text: str, model: GeminiModel = GeminiModel.GEMINI_2_0_FLASH
    ) -> int:
        """
        Count the number of tokens in a string using Google's `count_tokens`.
        """
        if not text or not text.strip():
            return 0

        response = self._client.models.count_tokens(
            model=model.value,
            contents=[text],
        )
        return response.total_tokens

    def count_tokens_in_file(
        self,
        file: Union[str, pathlib.Path, os.PathLike, io.IOBase],
        prompt: Optional[str],
        model: GeminiModel = GeminiModel.GEMINI_2_0_FLASH,
        system_instruction: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> int:
        """
        Count the number of tokens in a file using Google's `count_tokens` for use with Gemini flash 2.0.
        Uses Google's Files API for files over FILE_MB_THRESHOLD.

        NOTE: if counting the tokens in a PDF this returns the number of tokens that Google uses to represent the file,
        NOT necessarily the number of tokens in the text of the document.

        Args:
            file: Can be a file path (string or Path object) or a file-like object (opened in binary or text mode)
            prompt: A prompt to include along with the file, which influences the response + token count.
            system_instruction: Optional system instruction to provide context for the model, which influences the response + token count.
            mime_type: Optional MIME type of the file. If not provided, will attempt to infer from file extension

        Returns:
            int: The number of tokens in the file

        Raises:
            FileNotFoundError: If the file path doesn't exist
        """
        file_bytes = None
        file_size_mb = 0

        if isinstance(file, (str, pathlib.Path, os.PathLike)):
            # input is a local file path
            file_path = pathlib.Path(file)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

            # If mime_type not provided, try to infer from file extension
            if mime_type is None:
                import mimetypes

                mime_type, _ = mimetypes.guess_type(str(file_path))
                if mime_type is None:
                    # Fallback to plain text
                    mime_type = "text/plain"

            file_bytes = file_path.read_bytes()

        elif hasattr(file, "read"):
            # handle file-like objects.
            # read the entire content; convert to bytes if necessary.
            content = file.read()
            file_bytes = (
                content if isinstance(content, bytes) else content.encode("utf-8")
            )
            file_size_mb = len(file_bytes) / (1024 * 1024)

            # restore file internal pointer
            try:
                file.seek(0)
            except Exception:
                pass

            if mime_type is None:
                mime_type = "text/plain"
        else:
            raise TypeError(
                "Expected a file path (string or Path) or a file-like IO object"
            )

        # Process based on file size
        if file_size_mb > self.FILE_MB_THRESHOLD:
            # For files exceeding the threshold, use the Files API.
            doc_io = io.BytesIO(file_bytes)
            doc = self.upload_file(file=doc_io, config={"mime_type": mime_type})
        else:
            # For smaller files, process the contents inline.
            doc = types.Part.from_bytes(
                data=file_bytes,
                mime_type=mime_type,
            )

        # Per Gemini documentation, put the document being analyzed before the prompt if using both
        contents = [doc, prompt] if prompt else [doc]

        # Count tokens
        response = self._client.models.count_tokens(
            model=model.value,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0,
            ),
        )
        return response.total_tokens

    def upload_file(
        self,
        *,
        file: Union[str, pathlib.Path, os.PathLike, io.IOBase],
        config: Optional[types.UploadFileConfigOrDict] = None,
    ) -> types.File:
        """
        Upload a file to the Google Files API.
        Passthrough to the underlying Google API client w/ matching interface.
        """
        return self._client.files.upload(
            file=file,
            config=config,
        )

    def generate_content(
        self,
        *,
        model: GeminiModel = GeminiModel.GEMINI_2_0_FLASH,
        contents: Union[types.ContentListUnion, types.ContentListUnionDict],
        config: Optional[types.GenerateContentConfigOrDict] = None,
    ) -> types.GenerateContentResponse:
        """
        Generate content via a Google genai model using the Gemini API.
        Passthrough to the underlying Google API client w/ matching interface.
        """
        return self._client.models.generate_content(
            model=model.value,
            contents=contents,
            config=config,
        )
