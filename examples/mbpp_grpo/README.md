# MBPP GRPO with Tool-Integrated Reasoning (TIR)

This example demonstrates how to train a model on the **MBPP (Mostly Basic Python Problems)** dataset using the **GRPO** algorithm, integrated with a **Python tool** for code verification during the reasoning process.

## Overview

The "Tool-Integrated Reasoning" (TIR) workflow allows the model to:
1.  **Think** about the programming task.
2.  **Write** draft Python code.
3.  **Execute** the code using a `python_executor` tool.
4.  **Observe** the output or error messages.
5.  **Refine** the solution based on the feedback before providing the final answer.

This iterative process is optimized using Group Relative Policy Optimization (GRPO), where the reward is based on whether the final code passes the hidden unit tests provided by the MBPP dataset.

## Directory Structure

- `mbpp_workflow.py`: The core logic for multi-turn tool-use and reasoning.
- `train_mbpp.py`: Training script with a custom reward function that executes code against MBPP test cases.
- `mbpp_grpo_config.yaml`: Configuration for GRPO training, including model paths and hyperparameters.
- `prompts.py`: Templates for the system prompt and task-specific instructions.
- `tool_manager.py` & `tools/`: Infrastructure for routing and executing tool calls (specifically Python).

## Requirements

Ensure you have the project dependencies installed:

```bash
uv sync
```

## Dataset

This example uses the sanitized MBPP dataset located at `datasets/mbpp/sanitized`. It automatically handles loading the parquet files for training and testing.

## Running Training

To start the GRPO training, use the following command:

```bash
uv run python examples/mbpp_grpo/train_mbpp.py --config examples/mbpp_grpo/mbpp_grpo_config.yaml
```

### Key Configuration Parameters

In `mbpp_grpo_config.yaml`:
- `actor.path`: The base model (default: `Qwen/Qwen2.5-Coder-1.5B-Instruct`).
- `gconfig.n_samples`: Number of samples per prompt for GRPO (default: 4).
- `mbpp.max_turns`: Maximum number of tool-use turns allowed (default: 2).
- `train_dataset.batch_size`: Batch size for training (default: 128).

## Reward Function

The reward function in `train_mbpp.py` works as follows:
1.  Extracts the final Python code block from the model's response.
2.  Appends the unit tests (`assert ...`) from the dataset to the code.
3.  Executes the combined script in a separate process.
4.  Assigns a reward of **1.0** if the code executes without errors, and **0.0** otherwise.

## Notes

- **Safety**: The `PythonExecutor` runs code in a separate process but is not fully sandboxed. Use with caution in untrusted environments.
- **Model Choice**: While `Qwen2.5-Coder` is recommended, any instruct-tuned model with chat template support can be used by setting `mbpp.is_chat_model: true`.
