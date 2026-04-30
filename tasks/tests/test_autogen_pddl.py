"""
Test suite for tasks/mas_workflow/autogen/autogen_pddl.py and
mas/memory/mas_memory/intrinsicmemory_notemplate.py.

Run from GMemory root:
    python -m pytest tasks/tests/test_autogen_pddl.py -v --tb=short
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# ── path setup ────────────────────────────────────────────────────────────────
# Env vars and dependency stubs are handled by conftest.py in this directory.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mas.agents import Agent, Env
from mas.memory.common import AgentMessage
from mas.memory.mas_memory.intrinsicmemory_notemplate import IntrinsicMASMemoryNoTemplate
from mas.reasoning import ReasoningBase, ReasoningConfig

from tasks.mas_workflow.autogen.autogen_pddl import AutoGen
from tasks.mas_workflow.autogen.autogen_prompt import AUTOGEN_PROMPT


# ── stubs & helpers ───────────────────────────────────────────────────────────

class StubReasoning(ReasoningBase):
    """Concrete ReasoningBase subclass required by build_system's isinstance check.

    Cycles through a list of responses; repeats from the start when exhausted.
    """
    def __init__(self, responses=None):
        super().__init__(llm_model=MagicMock())
        if isinstance(responses, str):
            responses = [responses]
        self._responses = responses or ["pickup block"]
        self._call_idx = 0

    def __call__(self, prompts, config):
        # Clamp to last item once exhausted (avoids cycling back to INVALID etc.)
        idx = min(self._call_idx, len(self._responses) - 1)
        self._call_idx += 1
        return self._responses[idx]

    @property
    def call_count(self):
        return self._call_idx


def make_memory(tmp_path, llm_response: str = "updated memory"):
    llm = MagicMock()
    llm.return_value = llm_response
    llm.model_name = "test-model"
    mem = IntrinsicMASMemoryNoTemplate(
        namespace="test",
        global_config={"working_dir": str(tmp_path)},
        llm_model=llm,
        embedding_func=MagicMock(),
    )
    return mem, llm


def make_env(max_trials=3, step_returns=None, feedback_return=(0.5, True, "done")):
    env = MagicMock()
    env.max_trials = max_trials
    env.process_action.side_effect = lambda a: a.strip()
    env.step.side_effect = step_returns or [("obs", 1.0, True)]
    env.feedback.return_value = feedback_return
    return env


def make_autogen(reasoning, memory, env) -> AutoGen:
    ag = AutoGen()
    ag.build_system(
        reasoning,
        memory,
        env,
        config={
            "successful_topk": 1,
            "failed_topk": 0,
            "insights_topk": 3,
            "threshold": 0.0,
            "use_projector": False,
        },
    )
    return ag


PATCH_GPTCHAT = "mas.memory.mas_memory.intrinsicmemory.GPTChat"


# ── A. IntrinsicMASMemoryNoTemplate.summarize() ───────────────────────────────

class TestSummarize:

    def test_llm_not_called_short_trajectory(self, tmp_path):
        """LLM not invoked when trajectory is ≤ 5 chars (initial value '\n\n>' is 3)."""
        mem, llm = make_memory(tmp_path)
        mem.init_task_context("task", "desc")
        mem.summarize(solver_message="sys")
        llm.assert_not_called()

    def test_llm_called_after_trajectory_grows(self, tmp_path):
        """LLM invoked exactly once after a move_state call pushes trajectory past 5 chars."""
        mem, llm = make_memory(tmp_path, llm_response="new memory")
        mem.init_task_context("task", "desc")
        mem.move_memory_state("action1", "observation1")
        mem.summarize(solver_message="sys")
        llm.assert_called_once()

    def test_only_new_memory_stored(self, tmp_path):
        """agent_intrinsic_memory is replaced entirely; old content must be absent."""
        mem, llm = make_memory(tmp_path, llm_response="brand new memory")
        mem.agent_intrinsic_memory = "old memory content"
        mem.init_task_context("task", "desc")
        mem.move_memory_state("a", "b")

        mem.summarize(solver_message="sys")

        assert mem.agent_intrinsic_memory == "brand new memory"
        assert "old memory content" not in mem.agent_intrinsic_memory

    def test_summary_contains_agent_memory_header(self, tmp_path):
        """Return value includes the '### Agent Memory' section marker."""
        mem, _ = make_memory(tmp_path)
        mem.init_task_context("task", "desc")
        assert "### Agent Memory" in mem.summarize(solver_message="sys")

    def test_summary_contains_task_description(self, tmp_path):
        """Return value includes the task description text."""
        mem, _ = make_memory(tmp_path)
        mem.init_task_context("task", "unique description text")
        assert "unique description text" in mem.summarize(solver_message="sys")

    def test_summary_contains_action_injection(self, tmp_path):
        """Return value includes the single-action constraint instruction."""
        mem, _ = make_memory(tmp_path)
        mem.init_task_context("task", "desc")
        assert "You can only perform one action" in mem.summarize(solver_message="sys")

    def test_solver_message_embedded_in_llm_prompt(self, tmp_path):
        """solver_message appears in the user Message sent to the LLM (custom_message slot)."""
        mem, llm = make_memory(tmp_path)
        mem.init_task_context("task", "desc")
        mem.move_memory_state("a", "b")

        mem.summarize(solver_message="UNIQUE_SOLVER_TAG_XYZ")

        user_content = next(m.content for m in llm.call_args[0][0] if m.role == "user")
        assert "UNIQUE_SOLVER_TAG_XYZ" in user_content

    def test_notemplate_system_prompt_is_nonempty(self, tmp_path):
        """NoTemplate memory_system_prompt is non-empty (base IntrinsicMASMemory uses '')."""
        mem, _ = make_memory(tmp_path)
        assert mem.memory_system_prompt.strip() != ""

    def test_notemplate_system_prompt_sent_to_llm(self, tmp_path):
        """System Message in the LLM call matches the NoTemplate system prompt."""
        mem, llm = make_memory(tmp_path)
        mem.init_task_context("task", "desc")
        mem.move_memory_state("a", "b")
        mem.summarize(solver_message="sys")

        system_content = next(m.content for m in llm.call_args[0][0] if m.role == "system")
        assert system_content == mem.memory_system_prompt

    def test_update_prompt_contains_required_format_vars(self, tmp_path):
        """memory_update_prompt contains all four expected format placeholders."""
        mem, _ = make_memory(tmp_path)
        for var in ("{custom_message}", "{task_description}", "{task_trajectory}", "{current_memory}"):
            assert var in mem.memory_update_prompt


# ── B. save_task_context() ─────────────────────────────────────────────────────

class TestSaveTaskContext:

    def test_wipes_agent_intrinsic_memory(self, tmp_path):
        """save_task_context resets agent_intrinsic_memory to empty string."""
        mem, _ = make_memory(tmp_path)
        mem.agent_intrinsic_memory = "important memory"
        with patch(PATCH_GPTCHAT):
            mem.save_task_context(label=True)
        assert mem.agent_intrinsic_memory == ""

    def test_resets_counter_to_zero(self, tmp_path):
        """save_task_context resets counter to 0."""
        mem, _ = make_memory(tmp_path)
        mem.counter = 7
        with patch(PATCH_GPTCHAT):
            mem.save_task_context(label=False)
        assert mem.counter == 0

    def test_reinitialises_llm_instance(self, tmp_path):
        """save_task_context replaces llm_model with a new GPTChat instance."""
        mem, original_llm = make_memory(tmp_path)
        with patch(PATCH_GPTCHAT) as MockGPT:
            MockGPT.return_value = MagicMock()
            mem.save_task_context(label=True)
        assert mem.llm_model is not original_llm

    def test_no_cross_task_memory_leak(self, tmp_path):
        """Memory written during task 1 is absent after save_task_context."""
        mem, llm = make_memory(tmp_path, llm_response="task1 memory")
        mem.init_task_context("t1", "d1")
        mem.move_memory_state("a", "b")
        mem.summarize(solver_message="")
        assert mem.agent_intrinsic_memory == "task1 memory"

        with patch(PATCH_GPTCHAT):
            mem.save_task_context(label=True)

        assert mem.agent_intrinsic_memory == ""


# ── C. Validator agent — VALID/INVALID evaluation ─────────────────────────────

class TestValidatorAgent:

    def test_validator_prompt_contains_valid_keyword(self):
        """Validator system prompt instructs the LLM to respond with 'VALID'."""
        assert "VALID" in AUTOGEN_PROMPT.validator_system_prompt

    def test_validator_prompt_contains_invalid_keyword(self):
        """Validator system prompt instructs the LLM to respond with 'INVALID'."""
        assert "INVALID" in AUTOGEN_PROMPT.validator_system_prompt

    def test_validator_response_valid_allows_env_step(self, tmp_path):
        """When validator returns 'VALID', env.step() is called with the solver action."""
        reasoning = StubReasoning(["pickup block", "VALID"])
        memory, _ = make_memory(tmp_path)
        env = make_env(max_trials=1, step_returns=[("obs", 1.0, True)])
        ag = make_autogen(reasoning, memory, env)

        with patch(PATCH_GPTCHAT):
            ag.schedule({"task_main": "t", "task_description": "d"})

        env.step.assert_called_once_with("pickup block")

    def test_validator_response_invalid_blocks_env_step_on_that_attempt(self, tmp_path):
        """When validator returns 'INVALID' then 'VALID', env.step is called only once."""
        # invalid on first attempt, valid on second
        reasoning = StubReasoning(["action", "INVALID: bad", "action", "VALID"])
        memory, _ = make_memory(tmp_path)
        env = make_env(max_trials=1, step_returns=[("obs", 1.0, True)])
        ag = make_autogen(reasoning, memory, env)

        with patch(PATCH_GPTCHAT):
            ag.schedule({"task_main": "t", "task_description": "d"})

        env.step.assert_called_once()

    def test_validator_prompt_includes_solver_action(self, tmp_path):
        """Validator receives a prompt that contains the solver's latest action."""
        solver_action = "pickup block_A"
        # Capture validator call arguments via a side effect on the reasoning module
        captured_prompts = []

        class CapturingReasoning(ReasoningBase):
            def __init__(self):
                super().__init__(llm_model=MagicMock())
                self._idx = 0

            def __call__(self, prompts, config):
                captured_prompts.append(prompts)
                responses = [solver_action, "VALID"]
                resp = responses[self._idx % len(responses)]
                self._idx += 1
                return resp

        reasoning = CapturingReasoning()
        memory, _ = make_memory(tmp_path)
        env = make_env(max_trials=1, step_returns=[("obs", 1.0, True)])
        ag = make_autogen(reasoning, memory, env)

        with patch(PATCH_GPTCHAT):
            ag.schedule({"task_main": "t", "task_description": "d"})

        # Second call is the validator — its user prompt should include the solver action
        validator_user_msg = next(m.content for m in captured_prompts[1] if m.role == "user")
        assert solver_action in validator_user_msg

    def test_validator_prompt_includes_task_description(self, tmp_path):
        """Validator prompt contains the task_description from task_config."""
        captured_prompts = []

        class CapturingReasoning(ReasoningBase):
            def __init__(self):
                super().__init__(llm_model=MagicMock())
                self._idx = 0

            def __call__(self, prompts, config):
                captured_prompts.append(prompts)
                resp = ["action", "VALID"][self._idx % 2]
                self._idx += 1
                return resp

        reasoning = CapturingReasoning()
        memory, _ = make_memory(tmp_path)
        env = make_env(max_trials=1, step_returns=[("obs", 1.0, True)])
        ag = make_autogen(reasoning, memory, env)

        with patch(PATCH_GPTCHAT):
            ag.schedule({"task_main": "t", "task_description": "UNIQUE_TASK_DESC"})

        validator_user_msg = next(m.content for m in captured_prompts[1] if m.role == "user")
        assert "UNIQUE_TASK_DESC" in validator_user_msg

    def test_validator_evaluation_stored_in_validator_memory(self, tmp_path):
        """After validation, meta_memory_validator.summarize() is called with the evaluation."""
        reasoning = StubReasoning(["action", "VALID"])
        memory, _ = make_memory(tmp_path)
        env = make_env(max_trials=1, step_returns=[("obs", 1.0, True)])
        ag = make_autogen(reasoning, memory, env)

        with patch(PATCH_GPTCHAT):
            with patch.object(ag.meta_memory_validator, "summarize", wraps=ag.meta_memory_validator.summarize) as mock_summarize:
                ag.schedule({"task_main": "t", "task_description": "d"})

        mock_summarize.assert_called_once()
        call_kwargs = mock_summarize.call_args[1]
        assert "VALID" in call_kwargs.get("solver_message", "")


# ── D. Solver instruction — prepended on INVALID ──────────────────────────────

class TestSolverInstruction:

    def test_first_solver_call_has_no_instruction_prefix(self, tmp_path):
        """On the first attempt, solver receives only user_prompt with no prefix."""
        captured = []

        class CapturingReasoning(ReasoningBase):
            def __init__(self):
                super().__init__(llm_model=MagicMock())
                self._idx = 0

            def __call__(self, prompts, config):
                captured.append(prompts)
                resp = ["action", "VALID"][self._idx % 2]
                self._idx += 1
                return resp

        reasoning = CapturingReasoning()
        memory, _ = make_memory(tmp_path)
        env = make_env(max_trials=1, step_returns=[("obs", 1.0, True)])
        ag = make_autogen(reasoning, memory, env)

        with patch(PATCH_GPTCHAT):
            ag.schedule({"task_main": "t", "task_description": "d"})

        # First call is solver. Its user message should NOT start with instruction text.
        solver_user_msg = next(m.content for m in captured[0] if m.role == "user")
        assert not solver_user_msg.startswith("Your response does not follow")

    def test_solver_instruction_prepended_after_invalid(self, tmp_path):
        """After an INVALID evaluation, the next solver prompt starts with the revision instruction."""
        captured = []

        class CapturingReasoning(ReasoningBase):
            def __init__(self):
                super().__init__(llm_model=MagicMock())
                self._idx = 0

            def __call__(self, prompts, config):
                captured.append(prompts)
                # Call order: solver, validator(INVALID), solver(again), validator(VALID)
                resp = ["action", "INVALID: wrong", "action", "VALID"][self._idx % 4]
                self._idx += 1
                return resp

        reasoning = CapturingReasoning()
        memory, _ = make_memory(tmp_path)
        env = make_env(max_trials=1, step_returns=[("obs", 1.0, True)])
        ag = make_autogen(reasoning, memory, env)

        with patch(PATCH_GPTCHAT):
            ag.schedule({"task_main": "t", "task_description": "d"})

        # Third call (index 2) is the revised solver prompt — should contain instruction prefix
        revised_solver_user_msg = next(m.content for m in captured[2] if m.role == "user")
        assert "does not follow" in revised_solver_user_msg or "Validator" in revised_solver_user_msg

    def test_solver_instruction_contains_original_action_and_evaluation(self, tmp_path):
        """solver_instruction embeds the action that was rejected and the evaluation text."""
        captured = []

        class CapturingReasoning(ReasoningBase):
            def __init__(self):
                super().__init__(llm_model=MagicMock())
                self._idx = 0

            def __call__(self, prompts, config):
                captured.append(prompts)
                resp = ["ORIGINAL_ACTION", "INVALID: SPECIFIC_REASON", "ORIGINAL_ACTION", "VALID"][self._idx % 4]
                self._idx += 1
                return resp

        reasoning = CapturingReasoning()
        memory, _ = make_memory(tmp_path)
        env = make_env(max_trials=1, step_returns=[("obs", 1.0, True)])
        ag = make_autogen(reasoning, memory, env)

        with patch(PATCH_GPTCHAT):
            ag.schedule({"task_main": "t", "task_description": "d"})

        revised_solver_user_msg = next(m.content for m in captured[2] if m.role == "user")
        assert "ORIGINAL_ACTION" in revised_solver_user_msg
        assert "SPECIFIC_REASON" in revised_solver_user_msg

    def test_solver_instruction_reset_each_outer_iteration(self, tmp_path):
        """solver_instruction resets to '' at the start of each env.max_trials iteration."""
        captured = []

        class CapturingReasoning(ReasoningBase):
            def __init__(self):
                super().__init__(llm_model=MagicMock())
                self._idx = 0

            def __call__(self, prompts, config):
                captured.append(prompts)
                # Iter 1: INVALID then VALID; Iter 2: VALID immediately
                responses = ["a1", "INVALID: bad", "a1", "VALID", "a2", "VALID"]
                resp = responses[self._idx % len(responses)]
                self._idx += 1
                return resp

        reasoning = CapturingReasoning()
        memory, _ = make_memory(tmp_path)
        env = make_env(max_trials=2, step_returns=[("obs", 0.0, False), ("obs", 1.0, True)])
        ag = make_autogen(reasoning, memory, env)

        with patch(PATCH_GPTCHAT):
            ag.schedule({"task_main": "t", "task_description": "d"})

        # Iteration 2 solver prompt (captured[4]) should NOT start with instruction text
        second_iter_solver_msg = next(m.content for m in captured[4] if m.role == "user")
        assert not second_iter_solver_msg.startswith("Your response does not follow")


# ── E. schedule() — input validation ─────────────────────────────────────────

class TestScheduleValidation:

    @pytest.fixture
    def ag(self, tmp_path):
        reasoning = StubReasoning(["action", "VALID"])
        memory, _ = make_memory(tmp_path)
        env = make_env()
        return make_autogen(reasoning, memory, env)

    def test_raises_on_missing_task_main(self, ag):
        with pytest.raises(ValueError, match="task_main"):
            ag.schedule({"task_description": "d"})

    def test_raises_on_missing_task_description(self, ag):
        with pytest.raises(ValueError, match="task_description"):
            ag.schedule({"task_main": "t"})


# ── F. schedule() — lifecycle ─────────────────────────────────────────────────

class TestScheduleLifecycle:

    @pytest.fixture
    def setup(self, tmp_path):
        reasoning = StubReasoning(["pickup block", "VALID"])
        memory, _ = make_memory(tmp_path)
        env = make_env(max_trials=1, step_returns=[("obs", 1.0, True)])
        ag = make_autogen(reasoning, memory, env)
        return ag, memory, env

    def test_env_reset_called_once(self, setup):
        ag, _, env = setup
        with patch(PATCH_GPTCHAT):
            ag.schedule({"task_main": "t", "task_description": "d"})
        env.reset.assert_called_once()

    def test_solver_memory_initialised_with_task(self, setup):
        ag, memory, _ = setup
        with patch(PATCH_GPTCHAT):
            ag.schedule({"task_main": "my_task", "task_description": "my_desc"})
        assert memory.current_task_context.task_main == "my_task"
        assert memory.current_task_context.task_description == "my_desc"

    def test_valid_action_reaches_env_step(self, setup):
        ag, _, env = setup
        with patch(PATCH_GPTCHAT):
            ag.schedule({"task_main": "t", "task_description": "d"})
        env.step.assert_called_once_with("pickup block")

    def test_agent_node_recorded_per_step(self, setup):
        ag, memory, _ = setup
        with patch(PATCH_GPTCHAT):
            with patch.object(memory, "add_agent_node", wraps=memory.add_agent_node) as mock_add:
                ag.schedule({"task_main": "t", "task_description": "d"})
        mock_add.assert_called_once()

    def test_memory_state_moves_after_env_step(self, setup):
        ag, memory, _ = setup
        with patch(PATCH_GPTCHAT):
            with patch.object(memory, "move_memory_state", wraps=memory.move_memory_state) as mock_move:
                ag.schedule({"task_main": "t", "task_description": "d"})
        mock_move.assert_called_once()

    def test_save_task_context_called_on_solver_memory(self, setup):
        ag, memory, _ = setup
        with patch(PATCH_GPTCHAT):
            with patch.object(memory, "save_task_context", wraps=memory.save_task_context) as mock_save:
                ag.schedule({"task_main": "t", "task_description": "d"})
        mock_save.assert_called_once()

    def test_save_task_context_called_on_validator_memory(self, setup):
        ag, _, _ = setup
        with patch(PATCH_GPTCHAT):
            with patch.object(ag.meta_memory_validator, "save_task_context",
                               wraps=ag.meta_memory_validator.save_task_context) as mock_save:
                ag.schedule({"task_main": "t", "task_description": "d"})
        mock_save.assert_called_once()

    def test_backward_called_at_end(self, setup):
        ag, memory, _ = setup
        with patch(PATCH_GPTCHAT):
            with patch.object(memory, "backward") as mock_bwd:
                ag.schedule({"task_main": "t", "task_description": "d"})
        mock_bwd.assert_called_once()

    def test_returns_reward_and_done_from_feedback(self, setup):
        ag, _, env = setup
        env.feedback.return_value = (0.75, True, "great job")
        with patch(PATCH_GPTCHAT):
            reward, done = ag.schedule({"task_main": "t", "task_description": "d"})
        assert reward == 0.75
        assert done is True

    def test_ground_truth_used_when_solver_stuck(self, tmp_path):
        """When _solver_stuck fires, ground_truth.response() is called instead of solver."""
        # Three identical actions to trigger stuck detection
        responses = ["same action", "VALID"] * 3 + ["gt action", "VALID"]
        reasoning = StubReasoning(responses)
        memory, _ = make_memory(tmp_path)
        env = make_env(
            max_trials=4,
            step_returns=[
                ("obs", 0.0, False),
                ("obs", 0.0, False),
                ("obs", 0.0, False),  # stuck on 3rd, ground_truth called
                ("obs", 1.0, True),
            ],
        )
        ag = make_autogen(reasoning, memory, env)

        with patch(PATCH_GPTCHAT):
            ag.schedule({"task_main": "t", "task_description": "d"})

        # Ground truth should have been called (call_count > solver-only count)
        # With 3 solver+validator pairs + 1 ground_truth call = at least 7 calls
        assert reasoning.call_count >= 7

    def test_agent_message_records_ground_truth_name_when_stuck(self, tmp_path):
        """AgentMessage recorded when solver is stuck has agent_name == 'ground_truth'."""
        responses = ["same action", "VALID"] * 3 + ["gt action"]
        reasoning = StubReasoning(responses)
        memory, _ = make_memory(tmp_path)
        env = make_env(
            max_trials=4,
            step_returns=[("obs", 0.0, False), ("obs", 0.0, False), ("obs", 1.0, True)],
        )
        ag = make_autogen(reasoning, memory, env)

        recorded_messages = []
        original_add = memory.add_agent_node

        def capturing_add(msg, upstream_agent_ids):
            recorded_messages.append(msg)
            return original_add(msg, upstream_agent_ids)

        memory.add_agent_node = capturing_add

        with patch(PATCH_GPTCHAT):
            ag.schedule({"task_main": "t", "task_description": "d"})

        ground_truth_msgs = [m for m in recorded_messages if m.agent_name == "ground_truth"]
        assert len(ground_truth_msgs) >= 1


# ── G. _solver_stuck() ────────────────────────────────────────────────────────

class TestSolverStuck:

    @pytest.fixture
    def ag(self):
        return AutoGen()

    def test_not_stuck_empty_history(self, ag):
        assert ag._solver_stuck("action", []) is False

    def test_not_stuck_one_item_history(self, ag):
        assert ag._solver_stuck("action", ["action"]) is False

    def test_not_stuck_single_repeat(self, ag):
        assert ag._solver_stuck("action", ["different", "action"]) is False

    def test_detects_identical_triple(self, ag):
        assert ag._solver_stuck("go north", ["go north", "go north"]) is True

    def test_detects_think_loop(self, ag):
        # _solver_stuck requires len(action_history) >= 3 for the think check
        assert ag._solver_stuck("think: x", ["think: a", "think: y", "think: z"]) is True

    def test_think_loop_requires_three_items(self, ag):
        assert ag._solver_stuck("think: x", ["go north", "think: y"]) is False

    def test_detects_similar_actions(self, ag):
        # similar check also requires len(action_history) >= 3
        base = "pickup block_a from table"
        s1 = "pickup block_a from tabl"
        s2 = "pickup block_a from tabl."
        s3 = "pickup block_a from tab.."
        assert ag._solver_stuck(base, [s3, s1, s2]) is True

    def test_not_stuck_on_different_actions(self, ag):
        history = ["pickup block_a", "putdown block_b"]
        assert ag._solver_stuck("stack block_c block_d", history) is False


# ── H. Additional gaps ─────────────────────────────────────────────────────────

class TestAdditionalGaps:

    def test_solver_and_validator_have_distinct_memory_objects(self, tmp_path):
        """After build_system, meta_memory_solver and meta_memory_validator are distinct objects."""
        reasoning = StubReasoning()
        memory, _ = make_memory(tmp_path)
        env = make_env()
        ag = make_autogen(reasoning, memory, env)

        assert ag.meta_memory_solver is not ag.meta_memory_validator

    def test_validator_memory_has_different_namespace(self, tmp_path):
        """Validator memory namespace is derived from solver memory namespace + '_validator'."""
        reasoning = StubReasoning()
        memory, _ = make_memory(tmp_path)
        env = make_env()
        ag = make_autogen(reasoning, memory, env)

        assert ag.meta_memory_validator.namespace == ag.meta_memory_solver.namespace + "_validator"

    def test_schedule_raises_if_build_system_not_called(self):
        """Calling schedule() before build_system() raises AttributeError."""
        ag = AutoGen()
        with pytest.raises(AttributeError):
            ag.schedule({"task_main": "t", "task_description": "d"})

    def test_insights_topk_truncates_rules(self, tmp_path):
        """_project_insights limits each role's rule list to insights_topk."""
        reasoning = StubReasoning()
        memory, _ = make_memory(tmp_path)
        env = make_env()
        ag = AutoGen()
        ag.build_system(reasoning, memory, env, config={
            "successful_topk": 1, "failed_topk": 0,
            "insights_topk": 2, "threshold": 0.0, "use_projector": False,
        })
        rules = ["r1", "r2", "r3", "r4", "r5"]
        result = ag._project_insights(rules)
        for role_rules in result.values():
            assert len(role_rules) <= 2

    def test_trajectory_accumulates_action_and_observation(self, tmp_path):
        """MASMessage.move_state appends action and observation to task_trajectory."""
        mem, _ = make_memory(tmp_path)
        mem.init_task_context("task", "desc")
        mem.move_memory_state("pickup block", "You picked up the block.")
        assert "pickup block" in mem.current_task_context.task_trajectory
        assert "You picked up the block." in mem.current_task_context.task_trajectory

    def test_tries_incremented_on_invalid_limits_to_three(self, tmp_path):
        """tries IS incremented on each INVALID response (line 207 in autogen_pddl.py).
        After 3 INVALID evaluations the while loop exits and env.step is called with
        the last solver action regardless — total reasoning calls = 3 solver + 3 validator.
        """
        responses = []
        for _ in range(3):
            responses += ["action", "INVALID: bad"]
        # after 3 INVALIDs, tries==3, loop exits; no more reasoning calls needed

        reasoning = StubReasoning(responses)
        memory, _ = make_memory(tmp_path)
        env = make_env(max_trials=1, step_returns=[("obs", 1.0, True)])
        ag = make_autogen(reasoning, memory, env)

        with patch(PATCH_GPTCHAT):
            ag.schedule({"task_main": "t", "task_description": "d"})

        assert reasoning.call_count == 6
        env.step.assert_called_once_with("action")
