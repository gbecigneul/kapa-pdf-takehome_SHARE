import logging
from typing import Dict, List, Optional, Type

import openai
from openai import OpenAI

from .llm_types import LLMGeneration, Message, TokenUsage

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    class for making calls to Openai
    """

    def __init__(self, max_retries: int = 2, timeout: int = 60):
        # max_retries = 2 and timeout = 600 are the openai sdkd defaults
        self._client = OpenAI(
            max_retries=max_retries,  # default
            timeout=timeout,  # default
        )

        # Certain errors are retried by default
        # 408 Request Timeout
        # 409 Conflict
        # 429 Ratelimit
        # 500 Internal errors

    def _chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        stop: Optional[List[str]],
        stream: bool,
        stream_options: Optional[Dict],
        response_format: Optional[dict],
        num_generations: int = 1,
    ):
        """
        This function can be mocked easily for tests
        """
        if stop:
            return self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                stream=stream,
                stream_options=stream_options,
                response_format=response_format,
                n=num_generations,
            )
        else:
            return self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                stream_options=stream_options,
                response_format=response_format,
                n=num_generations,
            )

    def chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        response_format: Optional[dict] = None,
    ) -> LLMGeneration:
        """
        Call the OpenAI chat completions API.

        Args:
            messages: List of messages to send to the model.
            model: LLMModel to use for the completion.
            temperature: Temperature for sampling.
            max_tokens: Maximum number of tokens to generate.
            stop: List of strings to stop the generation.
            response_format: Response format for the completion. See chat_completion_parsed for enforced format.
        """
        try:
            response = self._chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                stream=False,
                stream_options=None,
                response_format=response_format,
            )

            return LLMGeneration(
                text=response.choices[0].message.content,
                token_usage=TokenUsage(
                    total_tokens=response.usage.total_tokens,
                    prompt_tokens=response.usage.prompt_tokens,
                    generation_tokens=response.usage.completion_tokens,
                ),
            )
        except (openai.RateLimitError, openai.InternalServerError) as exc:
            raise Exception(
                f"Exception occured for model {model} and messages {messages}"
            ) from exc
        except Exception as exc:
            raise Exception(
                f"Exception occured for model {model} and messages {messages}"
            ) from exc

    def convert_pdf(
        self,
        pdf_data: bytes,
        prompt: str,
        temperature: float = 0,
        filename: str = "temp",
    ) -> LLMGeneration:
        """
        Convert a PDF file to markdown using OpenAI's chat completion API w/ gpt-4.1-mini.
        We explicitly use the gpt-4.1-mini model for this task as it strikes the best balance between
        latency, cost, and quality of result.

        Args:
            pdf_data: The PDF data represented in bytes. ie: output of `pdf_file.read_bytes()`
            prompt: The prompt to use for the conversion.
            filename: The name of the file to use for the conversion. This is used to set the filename in the OpenAI API request.

        Returns:
            LLMGeneration: The generated markdown and token usage information.
        """
        data = base64.b64encode(pdf_data).decode("utf-8")

        # required input format for base64 files for the OpenAI API
        # seeMore: https://platform.openai.com/docs/guides/pdf-files?api-mode=chat
        file_data = f"data:application/pdf;base64,{data}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "file",
                        "file": {
                            # `filename` doesn't seem to matter unless you can use something descriptive / specific.
                            "filename": f"{filename}.pdf",
                            "file_data": file_data,
                        },
                    },
                ],
            }
        ]

        response = self._client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=temperature,
            response_format={"type": "text"},  # Markdown in plain text
        )

        resulting_markdown = response.choices[0].message.content.strip()

        return LLMGeneration(
            text=resulting_markdown,
            token_usage=TokenUsage(
                total_tokens=response.usage.total_tokens,
                prompt_tokens=response.usage.prompt_tokens,
                generation_tokens=response.usage.completion_tokens,
            ),
        )
