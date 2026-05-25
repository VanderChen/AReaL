import sys
import re
import os
from typing import Any
from datasets import load_dataset

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer

from examples.mbpp_grpo.mbpp_workflow import MBPPGRPOConfig  # isort: skip
from examples.mbpp_grpo.tools.python_tool import PythonExecutor  # isort: skip

logger = logging.getLogger("MBPP Training")

def extract_code(text: str) -> str:
    # Try to extract from ```python ... ```
    pattern = r"```python\n(.*?)\n```"
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return ""

def mbpp_reward_fn(prompt, completions, prompt_ids, completion_ids, **kwargs):
    # 'test_list' comes from kwargs (the dataset)
    test_list = kwargs.get("test_list", [])
    code = extract_code(completions)
    if not code:
        logger.warning("No code block found in completions for reward calculation")
        return 0.0

    # Prepare tests
    if isinstance(test_list, (list, tuple)):
        tests = "\n".join(test_list)
    else:
        # Handle numpy array or other iterables
        tests = "\n".join([str(t) for t in test_list])
    
    full_code = f"{code}\n{tests}"
    logger.info(f"Reward function verifying code with tests:\n{full_code}")

    try:
        executor = PythonExecutor()
        result, report = executor.execute(
            full_code.split("\n"), 
            get_answer_from_stdout=True, 
            runtime=executor.runtime
        )
        if report == "Done":
            logger.info("Reward verification PASSED (1.0)")
            return 1.0
        else:
            logger.info(f"Reward verification FAILED: {report}")
            return 0.0
    except Exception as e:
        logger.error(f"Error in reward function execution: {e}")
        return 0.0

def load_mbpp_dataset(path, split):
    # Support for JSONL files as seen in deploy.yaml
    jsonl_filename = "mbpp_train.jsonl" if split == "train" else "mbpp_test.jsonl"
    jsonl_path = os.path.join(path, jsonl_filename)
    
    if os.path.exists(jsonl_path):
        logger.info(f"Loading dataset from {jsonl_path}")
        dataset = load_dataset("json", data_files={split: jsonl_path}, split=split)
        # Rename 'text' or other columns to 'prompt' if necessary, 
        # but sanitized MBPP usually has 'prompt' and 'test_list'.
        # If it's raw MBPP, we might need more processing.
        return dataset

    # Fallback to parquet
    data_files = {split: os.path.join(path, f"{split}-*.parquet")}
    logger.info(f"Loading dataset from {data_files}")
    return load_dataset("parquet", data_files=data_files, split=split)

def main(args):
    config, _ = load_expr_config(args, MBPPGRPOConfig)

    logger.info("Starting MBPP training")
    logger.info(f"Configuration: {config.experiment_name}")
    logger.info(f"Model: {config.actor.path}")

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Load datasets
    train_dataset = load_mbpp_dataset(config.train_dataset.path, "train")
    valid_dataset = load_mbpp_dataset(config.valid_dataset.path, "test")

    from dataclasses import asdict
    mbpp_config_dict = asdict(config.mbpp)

    workflow_kwargs = dict(
        reward_fn="examples.mbpp_grpo.train_mbpp.mbpp_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        mbpp_config=mbpp_config_dict,
        enable_thinking=True,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    # Create trainer
    with PPOTrainer(config, train_dataset, valid_dataset) as trainer:
        # Run training
        trainer.train(
            workflow="examples.mbpp_grpo.mbpp_workflow.MBPPWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.mbpp_grpo.mbpp_workflow.MBPPWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )

if __name__ == "__main__":
    main(sys.argv[1:])
