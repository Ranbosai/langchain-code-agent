# LangChain Code Agent

## Overview

This repository contains a simple example of building an **AI agent** with the
[LangChain framework](https://www.langchain.com/).  The agent is capable of
generating code in many popular programming languages and explaining what the
generated code does.  It accomplishes this by combining a language model (LLM)
with custom tools that encapsulate the two core tasks:

1. **`write_code` tool** – given a description of a programming task and a target
   language, the tool prompts the language model to produce only the source
   code required to accomplish the task.  It can generate code for any
   language supported by the underlying LLM (Python, JavaScript, Java, C++,
   etc.).
2. **`explain_code` tool** – given a snippet of code and its programming
   language, the tool asks the language model to return a natural language
   explanation describing what the code does.  This is useful for learning
   unfamiliar language constructs or auditing generated code.

LangChain provides abstractions for building agents that use language models
alongside arbitrary tools.  According to the official LangChain website,
their code‑generation capabilities **“accelerate software development by
automating code writing, refactoring, and documentation for your team”**【46645545831789†L296-L299】.
The example in this repository demonstrates how to expose those capabilities via
a pair of simple tools that can be orchestrated by an agent.

## Repository contents

| File                    | Purpose                                                   |
|------------------------|-----------------------------------------------------------|
| `agent.py`             | Creates a LangChain agent using a chat model and the two custom tools. |
| `tools.py`             | Defines the `write_code` and `explain_code` tools using LangChain’s `@tool` decorator. |
| `cli.py`               | Command‑line interface for generating and explaining code without writing Python. |
| `requirements.txt`     | Lists the Python dependencies required to run the agent. |
| `LICENSE`              | MIT licence for open source use.                          |

## Installation

This project requires **Python 3.8+**.  To install the dependencies locally:

```bash
git clone https://github.com/your-user/langchain-code-agent.git
cd langchain-code-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note:** The agent uses OpenAI’s chat models by default.  You will need
> to supply an API key.  Set an environment variable named
> `OPENAI_API_KEY` before running any scripts.

## Usage

The project provides a small command‑line utility to interact with the
agent:

### Generate code

```
python cli.py write --language python --task "sort a list of integers"
```

This command will ask the agent to write Python code that sorts a list of
integers.  You can specify any programming language supported by the model
(e.g. `javascript`, `java`, `c++`).

### Explain code

```
python cli.py explain --language python --code """
def factorial(n):
    return 1 if n == 0 else n * factorial(n-1)
"""
```

The agent will return a human‑readable explanation of what the provided code
does.

### Running through the agent directly

If you wish to experiment without the CLI, you can import `create_agent`
from `agent.py` and call the agent with a list of messages.  The
`Build an Agent` tutorial in LangChain’s documentation shows how to set up
custom tools and call them from a language model【396155099479905†L509-L603】.

## Adding new languages

The `write_code` tool uses a prompt template that includes the target
programming language.  You can therefore specify any language supported by
OpenAI’s models.  To add language‑specific behaviour (e.g. using different
code style conventions), modify the prompt in `tools.py`.

## License

This project is licensed under the [MIT License](LICENSE).
