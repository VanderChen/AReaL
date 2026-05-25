"""
Prompt templates for MBPP GRPO workflow.
"""

SYSTEM_PROMPT = """You are a helpful assistant.
You can use the python_executor tool by writing code inside ```python\n...\n``` blocks.
The tool's output will be placed within ```output\n...\n```.
Please solve the task step by step, and provide your final Python solution in a ```python\n...\n``` block.
"""

MBPP_PROMPT = """A conversation between User and Assistant.
User: Write a Python function for: {prompt}
Assistant:
"""

ANSWER_PATTERN = r"```python\n(.*?)\n```"
