"""Custom tools for the LangChain code agent.

This module defines two tools: `generate_code` and `explain_code`.  Each tool
wraps a call to a large language model via LangChain’s `ChatOpenAI` class.  The
`generate_code` tool generates source code for a specified task and programming
language, while the `explain_code` tool provides a human‑readable explanation
for a given code snippet.

Both tools rely on an environment variable `OPENAI_API_KEY` to authenticate
against the OpenAI API.  You can modify the chat model parameters (such as
temperature or model name) by changing the `DEFAULT_MODEL` or passing your
own model when constructing the agent.
"""

from __future__ import annotations

import os
from typing import Optional

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.tools import tool

# Default model configuration.  Developers can customise this when creating the
# agent by passing a different model to `create_agent`.
DEFAULT_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo-0125")


def _get_chat_model(temperature: float = 0.0) -> ChatOpenAI:
    """Instantiate a ChatOpenAI model using the default settings.

    The function reads the `OPENAI_API_KEY` from the environment.  If the key
    is not set, an informative exception will be raised by `ChatOpenAI`.

    Args:
        temperature: Sampling temperature for the model.  Lower values
            encourage more deterministic responses.

    Returns:
        A configured `ChatOpenAI` instance.
    """
    return ChatOpenAI(model_name=DEFAULT_MODEL_NAME, temperature=temperature)


@tool
def generate_code(query: str, model_name: Optional[str] = None) -> str:
    """Generate source code for a task in a given programming language.

    This tool formulates a prompt instructing the language model to produce
    **only** the requested code. The input should be a string that contains the language and the task, separated by a comma.

    Args:
        query: A comma-separated string containing the programming language and a
            description of the desired behaviour. For example: "python,compute the factorial of a number".
        model_name: Optional override for the chat model to use.  If not
            provided, the `DEFAULT_MODEL_NAME` will be used.

    Returns:
        A string containing the generated source code.
    """
    language, task = query.split(",", 1)
    chat_model = ChatOpenAI(model_name=model_name or DEFAULT_MODEL_NAME, temperature=0.0)
    system_prompt = (
        "You are an expert software engineer. "
        "Write {language} code to accomplish the described task. "
        "Provide only the code, without comments or explanation."
    ).format(language=language)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=task),
    ]
    response = chat_model.invoke(messages)
    # Some models return a `content` attribute while others use `text`
    return getattr(response, "content", str(response))


@tool
def explain_code(code: str, language: str, model_name: Optional[str] = None) -> str:
    """Explain what the provided source code does.

    The tool sends both the code and its language to the language model and asks
    for a detailed, clear explanation.  The agent will respond in natural
    language and may include step‑by‑step descriptions or summaries.

    Args:
        code: The snippet of code to explain.
        language: The programming language of the snippet.
        model_name: Optional override for the chat model to use.  If not
            provided, the `DEFAULT_MODEL_NAME` will be used.

    Returns:
        A natural language explanation of the code.
    """
    chat_model = ChatOpenAI(model_name=model_name or DEFAULT_MODEL_NAME, temperature=0.2)
    system_prompt = (
        "You are a senior software engineer and educator. "
        "Given a piece of {language} code, describe what the code does. "
        "Provide a clear and concise explanation that would be understandable to a beginner."
    ).format(language=language)
    prompt = f"Here is the code:\n{code}"
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ]
    response = chat_model.invoke(messages)
    return getattr(response, "content", str(response))
