"""
Integration tests for autogen_mas.py that require a live LLM endpoint.

What is tested
--------------
1. Validator correctly classifies valid vs malformed PDDL-style actions.
2. IntrinsicMASMemoryNoTemplate produces a sensible, non-trivial memory summary
   from a real trajectory when the LLM is called without mocking.

Why a separate file
-------------------
These tests make real network calls and are non-deterministic.  They are kept
separate from the unit tests so the unit suite can run offline, in CI, and
without credentials.

Credential requirements
-----------------------
Both OPENAI_API_BASE and OPENAI_API_KEY must be exported as SHELL environment
variables before running pytest.  Setting them only in a .env file is NOT
sufficient: conftest.py's setdefault() runs before load_dotenv() and will lock
in the placeholder values first.

    export OPENAI_API_BASE="https://your-endpoint"
    export OPENAI_API_KEY="your-key"
    export TEST_LLM_MODEL="Llama-4-Maverick-17B-128E-Instruct-FP8"   # optional

Run:
    python3 -m pytest tasks/tests/test_autogen_pddl_integration.py -v -s
"""

import os
import sys
import pytest
from unittest.mock import MagicMock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── skip guard ────────────────────────────────────────────────────────────────
# Both vars must be real (non-placeholder) values set in the shell.
_PLACEHOLDER_URL = "http://localhost:9999"
_PLACEHOLDER_KEY = "test-key"

_has_real_credentials = (
    os.getenv("OPENAI_API_BASE") not in (None, _PLACEHOLDER_URL)
    and os.getenv("OPENAI_API_KEY") not in (None, _PLACEHOLDER_KEY)
)

pytestmark = pytest.mark.skipif(
    not _has_real_credentials,
    reason=(
        "Integration tests require real LLM credentials exported as shell env vars: "
        "OPENAI_API_BASE and OPENAI_API_KEY"
    ),
)

# ── imports (only reached when credentials are present) ───────────────────────
from mas.llm import GPTChat
from mas.reasoning import ReasoningIO, ReasoningConfig
from mas.memory.mas_memory.intrinsicmemory_notemplate import IntrinsicMASMemoryNoTemplate
from mas.agents import Agent
from tasks.mas_workflow.autogen.autogen_prompt import AUTOGEN_PROMPT

_DEFAULT_MODEL = "Llama-4-Maverick-17B-128E-Instruct-FP8"
TEST_MODEL = os.getenv("TEST_LLM_MODEL", _DEFAULT_MODEL)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def llm():
    """Real GPTChat instance shared across tests in this module."""
    return GPTChat(model_name=TEST_MODEL)


@pytest.fixture(scope="module")
def reasoning(llm):
    """Real ReasoningIO instance backed by the live LLM."""
    return ReasoningIO(llm_model=llm)


def _make_validator(reasoning) -> Agent:
    return Agent(
        name="validator",
        role="validator",
        system_instruction=AUTOGEN_PROMPT.validator_system_prompt,
        reasoning_module=reasoning,
    )


def _build_validator_prompt(
    solver_action: str,
    task_description: str,
    few_shots: list[str],
) -> str:
    """Replicates the validator_prompt built inside autogen_mas.schedule()."""
    return (
        f"You are evaluating whether a solver agent's response follows the expected format.\n\n"
        f"Solver's latest response:\n{solver_action}\n\n"
        f"Task description:\n{task_description}\n\n"
        f"Format that responses must follow (one short command per line):\n"
        + "\n".join(few_shots)
    )


# ── 1. Validator: valid vs malformed PDDL actions ─────────────────────────────

class TestValidatorIntegration:
    """
    The validator system prompt asks the LLM to respond 'VALID' or 'INVALID'
    based on whether the solver's output matches the reference format (few_shots).
    These tests use unambiguous cases so the outcome is stable at temperature=0.
    """

    _TASK = (
        "Blockworld: your goal is block_a on block_b and block_b on block_c. "
        "Output one action per step as a short command."
    )
    _FEW_SHOTS = [
        "pickup block_a",
        "putdown block_a",
        "stack block_a on block_b",
        "unstack block_b from block_c",
    ]
    _CONFIG = ReasoningConfig(temperature=0)

    def test_single_command_is_valid(self, reasoning):
        """A correctly formatted single-line action must be classified VALID."""
        validator = _make_validator(reasoning)
        prompt = _build_validator_prompt(
            solver_action="pickup block_a",
            task_description=self._TASK,
            few_shots=self._FEW_SHOTS,
        )
        response = validator.response(prompt, self._CONFIG)
        assert "VALID" in response, f"Expected VALID, got: {response!r}"
        assert "INVALID" not in response

    def test_correct_format_with_unseen_objects_is_valid(self, reasoning):
        """An action that follows the format but uses objects not in the few-shots
        must still be classified VALID.  This confirms the validator is checking
        FORMAT (verb + object pattern) rather than doing a verbatim string match
        against the reference examples.
        """
        validator = _make_validator(reasoning)
        prompt = _build_validator_prompt(
            solver_action="stack block_d on block_e",   # block_d / block_e not in few-shots
            task_description=self._TASK,
            few_shots=self._FEW_SHOTS,
        )
        response = validator.response(prompt, self._CONFIG)
        assert "VALID" in response, f"Expected VALID, got: {response!r}"
        assert "INVALID" not in response

    def test_invalid_response_includes_explanation(self, reasoning):
        """When the validator classifies an action as INVALID it must also
        provide a non-empty explanation, not just the bare keyword.

        The validator system prompt instructs: 'respond with "INVALID" and
        provide a brief explanation of why it is incorrect.'  This test
        verifies the LLM honours that instruction.
        """
        validator = _make_validator(reasoning)
        prompt = _build_validator_prompt(
            solver_action="TASK COMPLETE. The goal has been achieved.",
            task_description=self._TASK,
            few_shots=self._FEW_SHOTS,
        )
        response = validator.response(prompt, self._CONFIG)

        assert "INVALID" in response, f"Expected INVALID, got: {response!r}"

        # Strip the keyword and any surrounding punctuation/whitespace;
        # what remains must be a non-trivial explanation.
        explanation = response.replace("INVALID", "").strip(" :.–-\n")
        assert len(explanation) > 10, (
            f"Validator returned INVALID but no meaningful explanation.\n"
            f"Full response: {response!r}"
        )

    def test_reasoning_paragraph_is_invalid(self, reasoning):
        """A multi-sentence reasoning paragraph must be classified INVALID.
        The format requires a single short command, not an explanation.
        """
        validator = _make_validator(reasoning)
        prompt = _build_validator_prompt(
            solver_action=(
                "I have carefully analysed the current state of the blocks world. "
                "After considering all available moves, I believe the best strategy "
                "is to first clear the top of block_b, then stack block_a on block_b, "
                "and finally move block_b onto block_c."
            ),
            task_description=self._TASK,
            few_shots=self._FEW_SHOTS,
        )
        response = validator.response(prompt, self._CONFIG)
        assert "INVALID" in response, f"Expected INVALID, got: {response!r}"

    def test_task_completion_claim_is_invalid(self, reasoning):
        """A meta-comment claiming task completion must be classified INVALID.
        The format requires an executable action command, not a status update.
        """
        validator = _make_validator(reasoning)
        prompt = _build_validator_prompt(
            solver_action="TASK COMPLETE. The goal has been achieved.",
            task_description=self._TASK,
            few_shots=self._FEW_SHOTS,
        )
        response = validator.response(prompt, self._CONFIG)
        assert "INVALID" in response, f"Expected INVALID, got: {response!r}"


# ── 2. Memory summary: sensible output from real trajectory ───────────────────

class TestMemorySummaryIntegration:
    """
    IntrinsicMASMemoryNoTemplate.summarize() calls the LLM to compress the
    current trajectory into agent_intrinsic_memory.  These tests verify that
    the real LLM produces non-trivial, domain-relevant output.
    """

    def _make_memory(self, llm, tmp_path, namespace: str) -> IntrinsicMASMemoryNoTemplate:
        return IntrinsicMASMemoryNoTemplate(
            namespace=namespace,
            global_config={"working_dir": str(tmp_path)},
            llm_model=llm,
            embedding_func=MagicMock(),
        )

    def test_summary_is_nonempty_after_trajectory(self, llm, tmp_path):
        """LLM must produce a non-empty memory string from a real trajectory."""
        mem = self._make_memory(llm, tmp_path, "integ_nonempty")
        mem.init_task_context(
            task_main="blockworld",
            task_description="Stack block_a on block_b. block_c is on the table.",
        )
        mem.move_memory_state(
            "pickup block_a",
            "You are now holding block_a. block_a is no longer on the table.",
        )
        mem.move_memory_state(
            "stack block_a on block_b",
            "block_a is now on block_b. Your hand is empty.",
        )

        mem.summarize(solver_message="")

        assert mem.agent_intrinsic_memory.strip() != "", "LLM returned empty memory"
        assert len(mem.agent_intrinsic_memory.strip()) > 20, (
            f"Memory too short to be meaningful: {mem.agent_intrinsic_memory!r}"
        )

    def test_summary_reflects_trajectory_content(self, llm, tmp_path):
        """Memory must reference objects or actions that appeared in the trajectory."""
        mem = self._make_memory(llm, tmp_path, "integ_content")
        mem.init_task_context(
            task_main="blockworld",
            task_description="Goal: block_a on block_b.",
        )
        mem.move_memory_state(
            "unstack block_c from block_a",
            "block_c is now on the table. block_a is clear.",
        )
        mem.move_memory_state(
            "pickup block_a",
            "You are holding block_a.",
        )
        mem.move_memory_state(
            "stack block_a on block_b",
            "block_a is on block_b. Goal satisfied.",
        )

        mem.summarize(solver_message="")

        memory_lower = mem.agent_intrinsic_memory.lower()
        domain_terms = [
            "block", "pickup", "stack", "unstack",
            "hold", "table", "clear", "goal",
        ]
        assert any(term in memory_lower for term in domain_terms), (
            f"Memory summary does not mention any domain terms.\n"
            f"Got: {mem.agent_intrinsic_memory!r}"
        )

    def test_summary_does_not_contain_raw_prompt_boilerplate(self, llm, tmp_path):
        """The LLM should output only the updated memory, not the prompt template."""
        mem = self._make_memory(llm, tmp_path, "integ_boilerplate")
        mem.init_task_context(
            task_main="blockworld",
            task_description="Goal: block_a on block_b.",
        )
        mem.move_memory_state("pickup block_a", "Holding block_a.")
        mem.move_memory_state("stack block_a on block_b", "block_a on block_b.")

        mem.summarize(solver_message="")

        # The memory update prompt instructs: "OUTPUT ONLY THE UPDATED MEMORY. NOTHING MORE."
        # A well-behaved LLM should not echo the prompt headers back.
        assert "## Task Description" not in mem.agent_intrinsic_memory, (
            "LLM echoed prompt template headers into the memory output"
        )
        assert "## New Memory" not in mem.agent_intrinsic_memory, (
            "LLM echoed prompt template headers into the memory output"
        )

    def test_memory_is_wiped_between_tasks(self, llm, tmp_path):
        """Memory from task 1 must not survive into task 2 after save_task_context."""
        mem = self._make_memory(llm, tmp_path, "integ_wipe")
        mem.init_task_context("blockworld", "Task 1: block_a on block_b.")
        mem.move_memory_state("pickup block_a", "Holding block_a.")
        mem.summarize(solver_message="")
        assert mem.agent_intrinsic_memory.strip() != "", "Precondition: task 1 memory should exist"

        mem.save_task_context(label=True)

        assert mem.agent_intrinsic_memory == "", (
            "Memory was not wiped after save_task_context"
        )
