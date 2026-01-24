Plan: Integrate TravelPlanner dataset into GMemory (sole-planning only)

1) Data & deps
- Download HF dataset osunlp/TravelPlanner to `data/travelplanner/{split}.jsonl`.
- Download official `database/` dir into `data/travelplanner/database/`.
- Ensure TravelPlanner evaluation code is available locally (copy from repo or submodule). Tool APIs are not needed for sole-planning.
- Add deps: pandas, func_timeout (TravelPlanner uses langchain==0.1.4; prefer using existing langchain version and shim only if needed).

2) Task config
- Update `tasks/configs.yaml`: add `travelplanner` with `env_config_path`, `max_steps` (e.g., 6), `few_shots_num` (e.g., 1-2).
- Add `tasks/env_configs/travelplanner_config.yaml` containing `set_type` (train/validation/test), `data_path`, `database_dir`, `evaluation_root` (path to evaluation scripts), `mode: sole-planning`.
- Extend CLI choices in `tasks/run.py` (`--task travelplanner`) and ensure WORKING_DIR path handles the new task.

3) Task loading
- In `tasks/envs/__init__.py` add `TASKS_PATH['travelplanner'] = data/travelplanner/<set_type>.jsonl`.
- Build `travelplanner_tasks`: each item needs `query`, `days`, `visiting_city_number`, `org`, `dest`, `budget`, `people_number`, `local_constraint`, `level`, `background` (list of Description/Content strings).
- Register `ENVS/RECORDERS/TASK_DATA` entries for `travelplanner`.

4) Environment (sole-planning)
- New file `tasks/envs/travelplanner_env.py`:
  - `TravelPlannerEnv(BaseEnv)`: load evaluation modules from `evaluation_root` (commonsense_constraint/hard_constraint), keep `max_trials`.
  - `set_env`: stash current sample; `task_main=query`; `task_description` = stitched background + constraints + required output format (list of day dicts with fields: current_city, transportation, breakfast, lunch, dinner, attraction, accommodation; use '-' when not needed).
  - `process_action`: extract JSON/list from model output; validate keys; return normalized structure or error string.
  - `step`: on first valid submission, run commonsense/hard eval to get scores; observation summarizes format errors or pass/fail details; set `done` when a valid plan is evaluated (or after `max_trials`).
  - `feedback`: return delivery flag, success booleans, summary message.
- `TravelPlannerRecorder(BaseRecorder)`: track delivery/commonsense/hard/final pass counts; log per-task scores; optionally dump each plan as line JSON for offline eval.

5) Prompts & few-shots
- Add `tasks/prompts/travelplanner_prompt.py` with solver system prompt: must only use provided background, output JSON list in the fixed schema, use '-' for unnecessary fields, stop after return city; clarify no tool calls (sole-planning).
- Add 1-2 few-shots from `finetuning_data/std_travelplanner.jsonl` (trim for brevity).
- Register in `tasks/prompts/__init__.py` and extend `get_dataset_system_prompt/get_task_few_shots`.

6) Run example
- `python tasks/run.py --task travelplanner --mas_type autogen --mas_memory g-memory --reasoning io --model <model> --max_trials 6 --successful_topk 0 --failed_topk 0 --insights_topk 0`
- Recorder can save predictions to a file ready for `evaluation/eval.py --set_type <set_type> --evaluation_file_path <preds>`.

7) Notes
- Keep actions = single full-plan output per episode; no tool calls (sole-planning only).
- If langchain version mismatches, isolate TravelPlanner imports or lightly patch paths instead of downgrading global deps.
