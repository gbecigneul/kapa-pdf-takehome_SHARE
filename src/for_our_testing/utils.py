from typing import List

import tiktoken

from .llm_types import Message

# This number of tokens will be reserved for the generation
# in the context window of the used model. However, it is not used
# to limit the number of tokens that the LLM can generate. We just
# allow it to use whatever is left in the context window. If we exceed
# the context window the model will simply stop generating. This way
# we can allow longer generations because usually we do not fill
# the context window completely with retrieval results and we
# just give the generation a minimum window of 768 tokens.
RESERVED_GENERATION_TOKENS = 768


def num_tokens_from_message(
    message: Message, encoding_name: str = "cl100k_base"
) -> int:
    """
    Calcuate the number tokens for the content of a 'Message'
    Every message comes with some extra token overhead see: https://github.com/openai/tiktoken/issues/66
    which at the most can be +6 tokens, so we add that to the tokens of the content
    """

    return num_tokens_from_string(message["content"], encoding_name) + 6


def num_tokens_from_messages(
    messages: List[Message],
    encoding_name: str = "cl100k_base",
    exclude_roles: List[str] = [],
) -> int:
    """
    Calcuate the number tokens for the content of a list of 'Messages'
    'exclude_roles' can be used to not include messages of certain types in the token count.
    This is used for example for the custom chat endpoint to not count the 'context' place holder messages
    which are replaced instead of sent to Openai.
    """

    tokens = 0
    for message in messages:
        if message["role"] in exclude_roles:
            continue  # No op
        tokens += num_tokens_from_message(message, encoding_name)

    return tokens


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """
    Returns the number of tokens in a text string.
    Allows all special characters
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens
