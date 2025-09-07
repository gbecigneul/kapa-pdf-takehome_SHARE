from typing import List, TypedDict

from pydantic import BaseModel


class Message(TypedDict):
    """
    Represents a message to be sent to an LLM
    """

    role: str
    content: str


class TokenUsage(TypedDict):
    """
    Represents the token usage of an LLM call
    """

    total_tokens: int
    generation_tokens: int
    prompt_tokens: int


class LLMGeneration(TypedDict):
    """
    Represents the output of an LLM
    """

    text: str
    token_usage: TokenUsage


class LLMMultiGeneration(TypedDict):
    """
    Represents multiple versions of the output of an LLM
    E.g. when using higher temperatures one can get multiple different outputs for the same input
    """

    texts: List[str]
    token_usage: TokenUsage


class LLMParsedGeneration(TypedDict):
    """
    Represents the output of an LLM
    """

    output: BaseModel
    refusal: str  # If OpenAI refuses to generate a response e.g. because of abusive queries, this will be populated with a human-readable reason
    token_usage: TokenUsage
