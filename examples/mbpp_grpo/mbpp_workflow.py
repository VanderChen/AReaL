import ast
import copy
import re
import uuid
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api import (
    AsyncRewardWrapper,
    InferenceEngine,
    ModelRequest,
    ModelResponse,
    RolloutWorkflow,
)
from areal.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    dataclass,
    field,
)
from areal.utils import logging, stats_tracker
from areal.utils.hf_utils import apply_chat_template

from examples.mbpp_grpo.prompts import MBPP_PROMPT, SYSTEM_PROMPT  # isort: skip
from examples.mbpp_grpo.tool_manager import ToolCallStatus, ToolManager  # isort: skip

logger = logging.getLogger("MBPP workflow")


@dataclass
class MBPPConfig:
    max_turns: int = field(default=2)
    max_length: int = field(default=2048)
    tool_timeout: float = field(default=30)
    enable_tools: str = field(default="python")
    is_chat_model: bool = field(default=False)


@dataclass
class MBPPGRPOConfig(GRPOConfig):
    mbpp: MBPPConfig = field(default_factory=MBPPConfig)


class MBPPWorkflow(RolloutWorkflow):
    """Tool-Integrated Reasoning Workflow for MBPP."""

    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        mbpp_config: MBPPConfig | dict,
        enable_thinking: bool = False,
    ):
        super().__init__()
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer
            # Revert to standard call as force_download is not supported in this version
            tokenizer = load_hf_tokenizer(tokenizer)
        
        if isinstance(mbpp_config, dict):
            mbpp_config = MBPPConfig(**mbpp_config)
            
        self.reward_fn = reward_fn
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.tokenizer = tokenizer
        self.mbpp_config = mbpp_config
        self.enable_thinking = enable_thinking
        
        self.max_length = mbpp_config.max_length
        self.max_turns = mbpp_config.max_turns
        
        # Lazy initialization markers
        self.start_markers = ["```python", "<python>"]
        self.end_markers = ["```", "</python>"]
        self.async_reward_fn = None

    def _build_tool_manager(self) -> ToolManager:
        return ToolManager(
            self.mbpp_config.tool_timeout, 
            self.mbpp_config.enable_tools, 
            debug_mode=False
        )

    @staticmethod
    def _process_tool_result(tool_result) -> str:
        try:
            # Handle string result from PythonTool
            if isinstance(tool_result, str):
                res = tool_result
            else:
                run_result, run_status = ast.literal_eval(str(tool_result))
                res = run_result if run_status == "Done" else f"Error: {run_status}"
        except Exception:
            res = str(tool_result)
        return f"\n```output\n{res}\n```\n"

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Run a complete MBPP inference episode."""
        # Simple health check before starting
        try:
            # Try to see if we can get a response from health
            # Note: This is synchronous in current RemoteInfEngine but we can wrap it
            import requests
            # We don't have addresses here, but engine.agenerate will fail if not ready.
            # Let's just log that we are starting.
            logger.info("[START_EPISODE] Ensuring engine connection...")
        except Exception:
            pass

        if self.async_reward_fn is None:
            if isinstance(self.reward_fn, str):
                from areal.utils.dynamic_import import import_from_string
                reward_fn = import_from_string(self.reward_fn)
            else:
                reward_fn = self.reward_fn
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)

        tool_manager = self._build_tool_manager()
        try:
            # Initialize conversation history
            prompt = data["prompt"]

            # Prepare input
            if self.mbpp_config.is_chat_model:
                system_prompt = SYSTEM_PROMPT.format(
                    tool_descriptions=tool_manager.get_tool_descriptions_prompt()
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please write a Python function for the following task: {prompt}\nYou can use the python_executor tool to verify your code with test cases."},
                ]
                input_ids = apply_chat_template(
                    self.tokenizer,
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            else:
                input_ids = self.tokenizer.encode(
                    MBPP_PROMPT.format(prompt=prompt),
                    add_special_tokens=False,
                )

            # Update markers from manager
            self.start_markers = tool_manager.get_all_start_markers()
            self.end_markers = tool_manager.get_all_end_markers()

            # Run single trajectory
            return await self._multi_round_response(
                engine, input_ids, data, tool_manager
            )
        finally:
            await tool_manager.acleanup()

    async def _multi_round_response(self, engine, prompt_ids, data, tool_manager=None):
        if tool_manager is None:
            tool_manager = self.tool_manager

        prompt_str = self.tokenizer.decode(prompt_ids)
        logger.info(f"--- New Episode ---")
        logger.info(f"Full Prompt:\n{prompt_str}")
        
        completions_str = ""
        has_tool = False
        tool_call_count = 0
        tool_success_count = 0
        stop_reason = None
        max_len = self.max_length
        turn = 0
        # State flag for each episode: whether waiting for tool start marker
        waiting_for_tool_start = True
        tool_start_idx = -1

        # initialize seq, logprobs, loss_mask, versions
        context_ids = copy.deepcopy(prompt_ids)
        seq = copy.deepcopy(prompt_ids)
        logprobs = [0.0] * len(context_ids)
        loss_mask = [0] * len(context_ids)
        versions = [-1] * len(context_ids)
        output_ids = []

        while turn <= self.max_turns:
            if len(context_ids) >= max_len:
                logger.warning(f"Reached max_length {max_len}, stopping.")
                break

            # Generate response
            logger.info(f"Turn {turn}, generating (waiting_for_tool_start={waiting_for_tool_start})...")
            try:
                resp, stop_reason = await self._generate_response(
                    engine, context_ids, max_len, waiting_for_tool_start
                )
            except Exception as e:
                logger.error(f"[DIAGNOSTIC] Generation failed: {e}")
                # Explicitly check engine health
                try:
                    # We can't access addresses easily, but we can try to ping
                    # RemoteInfEngine handles some of this, but let's try a direct ping if possible
                    pass
                except Exception:
                    pass
                raise
            context_ids.extend(resp.output_tokens)
            seq.extend(resp.output_tokens)
            logprobs.extend(resp.output_logprobs)
            loss_mask.extend([1] * resp.output_len)
            versions.extend(resp.output_versions)

            cur_completions_str = self.tokenizer.decode(resp.output_tokens)
            logger.info(f"[ROUND {turn}] Generated Snippet: {cur_completions_str} [stop_reason: {stop_reason}]")
            completions_str += cur_completions_str
            output_ids.extend(resp.output_tokens)

            # End token, truncate
            if context_ids[-1] in [
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
            ]:
                logger.info("EOS detected, finishing episode.")
                break

            # State transition logic: detect if tool start marker is encountered
            if waiting_for_tool_start and stop_reason == "stop":
                # Check if tool start marker is detected (case-insensitive)
                lower_cur = cur_completions_str.lower()
                tool_start_marker = None
                for marker in self.start_markers:
                    if lower_cur.endswith(marker.lower()):
                        tool_start_marker = marker
                        break
                
                if tool_start_marker:
                    logger.info(f"MATCHED tool start marker: {tool_start_marker}")
                    waiting_for_tool_start = False
                    # Use the actual end of completions_str
                    tool_start_idx = len(completions_str) - len(cur_completions_str) + cur_completions_str.lower().rfind(tool_start_marker.lower())
                    continue
                else:
                    logger.info("Stop reason was 'stop' but no start marker found at end of snippet.")

            # If tool call is detected, execute tool call
            if (
                not waiting_for_tool_start
                and stop_reason == "stop"
                and tool_start_idx != -1
            ):
                tool_input = completions_str[tool_start_idx:]
                logger.info(f"Executing tool call with input: {tool_input[:200]}...")
                tool_results, tool_status = await self._execute_tools(
                    tool_input, tool_manager
                )
                if tool_status == ToolCallStatus.NOT_FOUND:
                    logger.warning("Tool call detected but no suitable tool found.")
                    continue
                
                turn += 1
                has_tool = True
                tool_call_count += 1
                if (
                    tool_status == ToolCallStatus.SUCCESS
                    and "Error" not in tool_results
                ):
                    tool_success_count += 1
                
                processed_result = self._process_tool_result(tool_results)
                logger.info(f"Tool result ({tool_status}): {processed_result[:200]}...")
                
                # Append tool response token IDs
                tool_rsp_token_ids = self.tokenizer.encode(
                    processed_result, add_special_tokens=False
                )
                # Concatenate to seq
                context_ids.extend(tool_rsp_token_ids)
                seq.extend(tool_rsp_token_ids)
                logprobs.extend([0.0] * len(tool_rsp_token_ids))
                loss_mask.extend([0] * len(tool_rsp_token_ids))
                versions.extend([-1] * len(tool_rsp_token_ids))
                completions_str += processed_result

                # After tool execution completes, reset state flag to prepare for next tool call detection
                waiting_for_tool_start = True
            
            if turn > self.max_turns:
                logger.info(f"Reached max_turns {self.max_turns}, finishing episode.")
                break

        logger.info(f"Episode finished. Tool calls: {tool_call_count}, Successes: {tool_success_count}")
        reward = await self.async_reward_fn(
            completions=completions_str,
            prompt_ids=prompt_ids,
            completion_ids=output_ids,
            tool_using=has_tool,
            tool_status=tool_call_count,
            **data,
        )
        logger.info(f"Final Reward: {reward}")

        res = dict(
            input_ids=torch.tensor(seq[:max_len]).unsqueeze(0),
            logprobs=torch.tensor(logprobs[:max_len]).unsqueeze(0),
            loss_mask=torch.tensor(loss_mask[:max_len]).unsqueeze(0),
            versions=torch.tensor(versions[:max_len]).unsqueeze(0),
            attention_mask=torch.ones(len(seq[:max_len]), dtype=torch.bool).unsqueeze(
                0
            ),
            rewards=torch.tensor([float(reward)]),
        )
        return res

    async def _generate_response(
        self,
        engine: InferenceEngine,
        input_ids: list[int],
        max_len: int,
        waiting_for_tool_start: bool,
    ) -> tuple[ModelResponse, str]:
        """Generate response with tool call detection support"""

        # Select stop condition based on state flag
        if waiting_for_tool_start:
            stop_markers = [marker for marker in self.start_markers]
        else:
            stop_markers = [marker for marker in self.end_markers]

        # Set generation config, add tool call stop tokens
        gconfig = self.gconfig.new(
            n_samples=1, stop=[marker for marker in stop_markers]
        )

        # Generate response
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=gconfig,
            tokenizer=self.tokenizer,
        )

        resp = await engine.agenerate(req)
        return resp, resp.stop_reason

    def _detect_tool_start_marker(self, text: str) -> str | None:
        """Detect if text ends with tool start marker"""
        for marker in self.start_markers:
            if text.endswith(marker):
                return marker
        return None

    async def _execute_tools(
        self, response: str, tool_manager=None
    ) -> tuple[str, ToolCallStatus]:
        """Execute tool call"""
        if tool_manager is None:
            tool_manager = self.tool_manager
        return await tool_manager.aexecute_tool_call(response)
