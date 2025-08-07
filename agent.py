"""Definition of the LangChain agent used in this project.

The function `create_agent` returns an `AgentExecutor` configured with a chat
model and the custom tools defined in `tools.py`.  It uses LangChain’s
``initialize_agent`` helper to build a ReAct‑style agent that can decide
whether to call one of the tools or simply respond directly.
"""

from __future__ import annotations

from typing import Optional

from langchain.chat_models import ChatOpenAI
import tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub


def create_agent(model_name: Optional[str] = None, verbose: bool = False):
    """Construct an agent configured with the custom tools.

    Args:
        model_name: Optional override for the chat model to use.  If
            ``None``, the default model defined in ``tools.DEFAULT_MODEL_NAME``
            will be used.
        verbose: If ``True``, the agent will print its thought process and
            intermediate steps to the console.  This can be helpful for
            debugging but will clutter the output.

    Returns:
        An instance of ``AgentExecutor`` ready to handle user queries.
    """
    # Determine which model to use.  The default model name is defined in
    # tools.DEFAULT_MODEL_NAME; if a custom model_name is provided, use that.
    llm = ChatOpenAI(
        model_name=model_name or tools.DEFAULT_MODEL_NAME,
        temperature=0.0,
    )

    # Gather the tool functions defined in tools.py.  Because they are
    # decorated with @tool, LangChain will automatically convert them into
    # Tool objects with a description taken from the docstring.  The agent
    # reads those descriptions to decide when to invoke a tool.
    available_tools = [
        tools.generate_code,
        tools.explain_code,
    ]

    # Initialise a ReAct agent using the zero‑shot prompting strategy.  The
    # agent will consider the tool descriptions when deciding whether to use a
    # tool.  Setting ``handle_parsing_errors=True`` helps when the model
    # incorrectly formats tool calls.
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(
        tools=available_tools,
        llm=llm,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(agent=agent, tools=available_tools, verbose=verbose)
    return agent_executor
