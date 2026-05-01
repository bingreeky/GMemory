"""
Compatibility tests for autogen_pddl across all --task and --mas_memory
combinations supported by run.py.

Two parametrized test suites:
  1. build_system_compatible  — one test per mas_memory type (12 total)
     Verifies that AutoGen.build_system() accepts every memory module without error.

  2. schedule_compatible       — one test per (task × mas_memory) pair (48 total)
     Verifies that AutoGen.schedule() completes a one-step episode for every
     combination of task type and memory module.

All tests are fully mocked (no LLM calls, no real envs).  The heavy optional
dependencies (langchain_chroma, finch, …) are already stubbed by conftest.py.
GPTChat is patched globally in this module because every intrinsic memory class
re-instantiates it inside save_task_context().

Derived from run.py's argparse choices:
  --task       alfworld | fever | pddl | sciworld
  --mas_memory empty | voyager | memorybank | chatdev | generative | metagpt |
               g-memory | intrinsicmemory-pddl | intrinsicmemory-fever |
               intrinsicmemory-alfworld | intrinsicmemory-llm-structured-template |
               intrinsicmemory-notemplate
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mas.module_map import module_map
from mas.reasoning import ReasoningBase, ReasoningConfig
from tasks.mas_workflow.autogen.autogen_pddl import AutoGen

# ── constants ─────────────────────────────────────────────────────────────────

ALL_MAS_MEMORIES = [
    "empty",
    "voyager",
    "memorybank",
    "chatdev",
    "generative",
    "metagpt",
    "g-memory",
    "intrinsicmemory-pddl",
    "intrinsicmemory-fever",
    "intrinsicmemory-alfworld",
    "intrinsicmemory-llm-structured-template",
    "intrinsicmemory-notemplate",
]

# Representative task_configs matching the field shape each task type produces
# in run.py after env.set_env() and get_task_few_shots() have run.
TASK_CONFIGS = {
    "pddl": {
        "task_main": "The goal is: block_a on block_b",
        "task_description": "Initial state: block_a on table. Goal: block_a on block_b.",
        "few_shots": ["pickup block_a", "stack block_a on block_b"],
        "game_name": "blockworld",
        "problem_index": 0,
        "difficulty": "easy",
    },
    "alfworld": {
        "task_main": "put a mug on the table",
        "task_description": "pick up the mug from the shelf and put it on the table",
        "few_shots": ["go to shelf 1", "take mug 1 from shelf 1", "go to table 1", "put mug 1 on table 1"],
        "task_type": "pick_and_place",
        "env_name": "pick_and_place",
    },
    "fever": {
        "task_main": "The Great Wall of China is visible from space",
        "task_description": "Claim: The Great Wall of China is visible from space",
        "few_shots": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
        "answer": "REFUTES",
        "env_name": "fever",
    },
    "sciworld": {
        "task_main": "boil water",
        "task_description": "Use the stove to heat the water to 100 degrees Celsius.",
        "few_shots": ["look around", "go to stove", "turn on stove"],
        "task_name": "boil-water",
        "var": 1,
        "difficulty": "easy",
    },
}

PATCH_GPTCHAT = "mas.memory.mas_memory.intrinsicmemory.GPTChat"

# ── stubs ─────────────────────────────────────────────────────────────────────

class StubReasoning(ReasoningBase):
    """Minimal concrete ReasoningBase that passes build_system's isinstance check."""

    def __init__(self):
        super().__init__(llm_model=MagicMock())

    def __call__(self, prompts, config):
        # Returns "VALID" so the inner while loop always breaks cleanly.
        return "VALID"


def _make_memory(memory_key: str, tmp_path):
    """Instantiate the memory class for *memory_key* with a mock LLM."""
    _, MemoryClass = module_map("io", memory_key)
    llm = MagicMock()
    llm.return_value = "mock memory content"
    llm.model_name = "test-model"
    return MemoryClass(
        namespace=f"test_{memory_key.replace('-', '_')}",
        global_config={"working_dir": str(tmp_path)},
        llm_model=llm,
        embedding_func=MagicMock(),
    )


def _make_env():
    env = MagicMock()
    env.max_trials = 1
    env.process_action.side_effect = lambda a: a.strip()
    env.step.side_effect = [("obs", 1.0, True)]
    env.feedback.return_value = (1.0, True, "done")
    return env


# ── 1. build_system compatibility ─────────────────────────────────────────────

@pytest.mark.parametrize("memory_key", ALL_MAS_MEMORIES)
def test_build_system_compatible_with_memory_type(memory_key, tmp_path):
    """build_system() must not raise for every supported --mas_memory value.

    Verifies that AutoGen can be initialised with each memory module and that
    the validator memory is created as a distinct object with a derived namespace.
    """
    memory = _make_memory(memory_key, tmp_path)
    env = _make_env()
    ag = AutoGen()

    with patch(PATCH_GPTCHAT):
        ag.build_system(
            StubReasoning(),
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

    # Solver and validator memories must be distinct objects.
    assert ag.meta_memory_solver is not ag.meta_memory_validator, (
        f"[{memory_key}] meta_memory_solver and meta_memory_validator are the same object"
    )
    # Validator namespace must be derived from the solver namespace.
    assert ag.meta_memory_validator.namespace == ag.meta_memory_solver.namespace + "_validator", (
        f"[{memory_key}] Unexpected validator namespace: {ag.meta_memory_validator.namespace!r}"
    )
    # All four agents must be hired.
    assert set(ag.agents_team.keys()) == {"solver", "validator", "ground_truth"}, (
        f"[{memory_key}] Unexpected agents: {set(ag.agents_team.keys())}"
    )


# ── 2. schedule compatibility ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "task,memory_key",
    [(t, m) for t in TASK_CONFIGS for m in ALL_MAS_MEMORIES],
    ids=lambda x: x if isinstance(x, str) else x,
)
def test_schedule_compatible_with_task_and_memory(task, memory_key, tmp_path):
    """schedule() must complete a one-step episode for every (task × memory) pair.

    The env is fully mocked; this test verifies no unhandled exception is raised
    and that the return type is (float, bool) regardless of which task type or
    memory module is configured.
    """
    memory = _make_memory(memory_key, tmp_path)
    env = _make_env()
    ag = AutoGen()

    with patch(PATCH_GPTCHAT):
        ag.build_system(
            StubReasoning(),
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

        result = ag.schedule(TASK_CONFIGS[task])

    assert isinstance(result, tuple) and len(result) == 2, (
        f"[{task}/{memory_key}] schedule() returned {result!r}, expected (float, bool)"
    )
    reward, done = result
    assert isinstance(reward, (int, float)), (
        f"[{task}/{memory_key}] reward is {type(reward).__name__}, expected numeric"
    )
    assert isinstance(done, bool), (
        f"[{task}/{memory_key}] done is {type(done).__name__}, expected bool"
    )
