# Plan: Fix distinct agent memories + test suite for AutoGen PDDL MAS

## Step 1 — Code fix: give each agent its own memory object

**File:** `tasks/mas_workflow/autogen/autogen_pddl.py`, in `build_system()`

**Bug:** Both `meta_memory_solver` and `meta_memory_validator` are assigned the same object,
so updates to one silently pollute the other.

**Fix:** Reconstruct a second `IntrinsicMASMemoryNoTemplate` (or whatever subclass is passed)
using the same config but a distinct namespace. `MASMemoryBase` is a plain dataclass with exactly
four fields, so the class constructor can be called directly on `mas_memory.__class__`:

```python
# current (wrong)
self.meta_memory_solver = mas_memory
self.meta_memory_validator = mas_memory

# fixed
self.meta_memory_solver = mas_memory
self.meta_memory_validator = mas_memory.__class__(
    namespace=mas_memory.namespace + "_validator",
    global_config=mas_memory.global_config,
    llm_model=mas_memory.llm_model,
    embedding_func=mas_memory.embedding_func,
)
```

No signature changes to `build_system`, so `run.py` is unaffected.

---

## Context
The user wants unit tests for `tasks/mas_workflow/autogen/autogen_pddl.py`.
The three explicit concerns are:
1. Memory update operations work correctly for **both** solver and validator agents (only new memory output, not accumulated)
2. Validator provides correct VALID/INVALID evaluation
3. Solver instruction is correctly prepended to `user_prompt` inside the `while tries < 3` loop

`autogen_pddl.py` (current state) extends `autogen.py` with a third agent (validator) and two memory objects
(`meta_memory_solver`, `meta_memory_validator`). The validator sits **inside** the inner `while` loop —
it checks each solver action before allowing it to proceed to `env.step()`.

---

## Critical files

| File | Role |
|---|---|
| `tasks/mas_workflow/autogen/autogen_pddl.py` | Primary SUT |
| `tasks/mas_workflow/autogen/autogen_prompt.py` | Prompt strings (solver/validator/ground_truth) |
| `mas/memory/mas_memory/intrinsicmemory_notemplate.py` | `IntrinsicMASMemoryNoTemplate` — target memory class |
| `mas/memory/mas_memory/intrinsicmemory.py` | Parent: `IntrinsicMASMemory.summarize()` / `save_task_context()` |
| `mas/memory/mas_memory/prompt.py` | `INTRINSICMEMORY_NOTEMPLATE` prompt strings |
| `mas/memory/mas_memory/memory_base.py` | `MASMemoryBase` base class |
| `mas/memory/common.py` | `MASMessage`, `AgentMessage` |
| `mas/agents/base.py` | `Agent` |
| `mas/reasoning/reasoning_modules.py` | `ReasoningBase`, `ReasoningConfig` |
| `mas/llm.py` | `Message(role, content)` |

**Output file:** `tasks/tests/test_autogen_pddl.py`

---

## `IntrinsicMASMemoryNoTemplate` vs `IntrinsicMASMemory`

`NoTemplate` only overrides `memory_system_prompt` in `__post_init__`:

| | `IntrinsicMASMemory` | `IntrinsicMASMemoryNoTemplate` |
|---|---|---|
| `memory_update_prompt` | `DEFAULT_MEMORY_UPDATE_PROMPT` | same (inherited) |
| `memory_system_prompt` | `""` (empty) | `DEFAULT_MEMORY_SYSTEM_PROMPT_NOTEMPLATE` |
| `summarize()` | defined here | **inherited** |
| `save_task_context()` | defined here | **inherited** |

All behavioural tests use `IntrinsicMASMemoryNoTemplate` directly. Two additional tests verify the `NoTemplate`-specific properties.

---

## Architecture summary (for test design)

```
for i in range(env.max_trials):
    solver_instruction = ""           # ← reset each outer iteration
    while tries < 3:
        action = solver.response(solver_instruction + user_prompt)
        evaluation = validator.response(validator_prompt)
        meta_memory_validator.summarize(solver_message=evaluation)
        if "INVALID" in evaluation:
            solver_instruction = "...feedback + action + evaluation..."
            continue                  # ← tries NOT incremented here (potential infinite loop bug)
        action = env.process_action(action)
        break
    tries += 1                        # ← only on exception
    if _solver_stuck(): use ground_truth
    env.step(action)
    meta_memory_solver.move_memory_state(action, obs)

env.feedback()
meta_memory_solver.save_task_context()   # note: validator memory NOT saved
meta_memory_solver.backward()
```

Key facts:
- `meta_memory_solver` and `meta_memory_validator` both point to the **same** `mas_memory` object passed to `build_system`
- `meta_memory_validator.save_task_context()` is **never called** — validator memory accumulates across tasks (gap to note in tests)
- `tries` is only incremented on exception, not on INVALID → potential infinite loop if validator always returns INVALID

---

## Stubs needed

```python
class StubReasoning(ReasoningBase):
    """Concrete subclass — needed because build_system does isinstance(reasoning, ReasoningBase)."""
    def __init__(self, response="pickup block"):
        super().__init__(llm_model=MagicMock())
        self._response = response
    def __call__(self, prompts, config):
        return self._response

def make_memory(tmp_path, llm_response="updated memory"):
    llm = MagicMock()
    llm.return_value = llm_response
    llm.model_name = "test-model"
    return IntrinsicMASMemoryNoTemplate(   # ← NoTemplate, not IntrinsicMASMemory
        namespace="test",
        global_config={"working_dir": str(tmp_path)},
        llm_model=llm,
        embedding_func=MagicMock()
    ), llm

# Patch GPTChat when testing save_task_context
with patch("mas.memory.mas_memory.intrinsicmemory.GPTChat"):
    memory.save_task_context(label=True)
```

---

## Test categories and methods

### A. Memory update — `IntrinsicMASMemory.summarize()` (applies to both solver & validator)

| Test | What it checks |
|---|---|
| `test_llm_not_called_short_trajectory` | LLM not invoked when trajectory ≤ 5 chars (gating at `> 5`) |
| `test_llm_called_long_trajectory` | LLM invoked once after trajectory grows past threshold |
| `test_only_new_memory_stored` | `agent_intrinsic_memory` is **replaced** (not appended); old content absent |
| `test_summary_contains_agent_memory_header` | Return value contains `"### Agent Memory"` |
| `test_summary_contains_task_description` | Return value contains the original task description |
| `test_summary_contains_action_injection` | Return value contains `"You can only perform one action"` |
| `test_solver_message_in_llm_prompt` | The `solver_message` arg appears in the user `Message` sent to LLM (checks `custom_message` mapping) |
| `test_notemplate_memory_system_prompt_nonempty` | `memory_system_prompt` is not `""` — contrasts with base `IntrinsicMASMemory` |
| `test_notemplate_system_prompt_sent_to_llm` | The non-empty system prompt is the first `Message` in the LLM call (role `"system"`) |
| `test_notemplate_update_prompt_format_vars` | `memory_update_prompt` contains `{custom_message}`, `{task_description}`, `{task_trajectory}`, `{current_memory}` |

### B. Memory wipe — `save_task_context()`

| Test | What it checks |
|---|---|
| `test_wipes_agent_intrinsic_memory` | `agent_intrinsic_memory == ""` after call |
| `test_resets_counter` | `counter == 0` |
| `test_reinitialises_llm` | New LLM instance created (patch GPTChat) |
| `test_no_cross_task_memory_leak` | Memory from task 1 absent in task 2's summarize output |
| `test_validator_memory_not_saved_at_end` | `meta_memory_validator.save_task_context()` is never called in `schedule()` — documents gap |

### C. Validator agent — VALID/INVALID evaluation

| Test | What it checks |
|---|---|
| `test_validator_prompt_contains_valid_keyword` | `AUTOGEN_PROMPT.validator_system_prompt` contains `"VALID"` |
| `test_validator_prompt_contains_invalid_keyword` | Contains `"INVALID"` |
| `test_validator_response_valid_passes_through` | When validator returns `"VALID"`, solver action proceeds to `env.process_action()` |
| `test_validator_response_invalid_blocks_action` | When validator returns `"INVALID: reason"`, `env.process_action()` is NOT called on that attempt |
| `test_validator_prompt_includes_solver_action` | Validator prompt string contains the solver's latest action |
| `test_validator_prompt_includes_task_description` | Validator prompt contains `task_config["task_description"]` |
| `test_validator_prompt_includes_few_shots` | Validator prompt contains the few_shots strings |
| `test_validator_evaluation_stored_in_memory` | `meta_memory_validator.summarize()` called with evaluation text after each validator call |

### D. Solver instruction — prepended on INVALID

| Test | What it checks |
|---|---|
| `test_solver_instruction_empty_on_first_call` | First solver call uses `""` prefix (no instruction) |
| `test_solver_instruction_set_after_invalid` | After INVALID, `solver_instruction` contains the original action and evaluation text |
| `test_solver_instruction_prepended_to_prompt` | Second solver call uses `solver_instruction + user_prompt` (not just `user_prompt`) |
| `test_solver_instruction_reset_each_outer_iteration` | `solver_instruction` is `""` at the start of each `for i in range(env.max_trials)` iteration |
| `test_valid_after_invalid_clears_instruction_path` | A VALID response after an INVALID one causes action to be accepted and loop to `break` |

### E. `schedule()` — input validation

| Test | What it checks |
|---|---|
| `test_raises_missing_task_main` | `ValueError` when `task_main` absent |
| `test_raises_missing_task_description` | `ValueError` when `task_description` absent |

### F. `schedule()` — lifecycle & agent dispatch

| Test | What it checks |
|---|---|
| `test_env_reset_called_once` | `env.reset()` called exactly once |
| `test_solver_memory_initialised` | `meta_memory_solver.init_task_context()` called with correct task_main, task_description |
| `test_valid_action_reaches_env_step` | After VALID evaluation, `env.step()` called with processed action |
| `test_agent_node_recorded_per_env_step` | `add_agent_node` called at least once per step |
| `test_memory_state_moves_per_env_step` | `meta_memory_solver.move_memory_state()` called once per env step |
| `test_save_task_context_called_at_end` | Called exactly once on `meta_memory_solver` |
| `test_backward_called_at_end` | Called exactly once |
| `test_returns_reward_and_done` | Return value `(reward, done)` matches `env.feedback()` output |
| `test_ground_truth_used_when_stuck` | When `_solver_stuck` fires, `ground_truth.response()` is called |
| `test_agent_message_records_ground_truth_name` | `AgentMessage.agent_name == "ground_truth"` when stuck |

### G. `_solver_stuck()` — loop detection (same logic as autogen.py)

| Test | What it checks |
|---|---|
| `test_not_stuck_empty_history` | False with `[]` |
| `test_not_stuck_one_item` | False with 1-element history |
| `test_not_stuck_single_repeat` | False when only `hist[-1]` matches |
| `test_detects_identical_triple` | True: current == hist[-1] == hist[-2] |
| `test_detects_think_loop` | True: 3 consecutive "think: …" actions |
| `test_think_loop_requires_three` | False with only 2 "think" actions |
| `test_detects_similar_actions` | True for near-identical strings (pairwise > 0.8) |
| `test_not_stuck_different_actions` | False for genuinely different actions |

### H. Additional gaps the user may have missed

| Test | What it checks |
|---|---|
| `test_tries_increments_on_exception_only` | `tries` increments only in the `except` block, not on INVALID — documents the potential infinite loop if validator always returns INVALID |
| `test_solver_and_validator_have_distinct_memory_objects` | `meta_memory_solver is not meta_memory_validator` after `build_system` — currently **fails** because both are assigned the same `mas_memory` arg; this test documents the bug |
| `test_meta_memory_not_set_raises_on_schedule` | If `build_system` not called first, `schedule()` fails with AttributeError (no `meta_memory_solver`) |
| `test_insights_topk_truncates_rules` | `_project_insights` trims to `insights_topk` length |
| `test_trajectory_accumulates_action_and_obs` | After `move_state("a","b")`, `task_trajectory` contains both `"a"` and `"b"` |
| `test_format_prompt_embeds_all_sections` | `format_task_prompt_with_insights` output contains few_shots, memory_few_shots, insights, task_description sections |

---

## Verification

```bash
cd /path/to/GMemory
python -m pytest tasks/tests/test_autogen_pddl.py -v --tb=short
```

All tests pass without network access (no real LLM calls, GPTChat patched).
