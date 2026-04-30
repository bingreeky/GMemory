from dataclasses import dataclass

#----------------------------------------------intrinsicmemory memory DEFAULT----------------------------------------------

DEFAULT_MEMORY_UPDATE_PROMPT = """
Use your latest response to create the new memory with factual information to solve the task based on the task description, current task trajectory, and current memory. OUTPUT ONLY THE UPDATED MEMORY. NOTHING MORE.

{custom_message}

## Task Description
{task_description}

## Current Task Trajectory
{task_trajectory}

## Current Memory

{current_memory}

## New Memory

"""

@dataclass
class IntrinsicMemoryDEFAULT:
    memory_update_prompt: str = DEFAULT_MEMORY_UPDATE_PROMPT

INTRINSICMEMORYDEFAULT: IntrinsicMemoryDEFAULT = IntrinsicMemoryDEFAULT()

#----------------------------------------------intrinsicmemory memory PDDL----------------------------------------------
DEFAULT_MEMORY_UPDATE_PROMPT_PDDL = """
Use the latest two steps in the task trajectory to populate and update the current memory with factual information to solve the task 
based on the task description.

## Task Description
{task_description}

## Current Task Trajectory
{task_trajectory}

## Current Memory

{current_memory}

"""

DEFAULT_MEMORY_SYSTEM_PROMPT_PDDL = """
You are a MEMORY UPDATER for a PDDL-style planning agent.

Your job:
- Maintain a compact JSON memory capturing stable, reusable information across tasks and domains.
- Only store information that improves future planning: common strategies, mistakes, valid action patterns, state-transition insights.
- Do not store long histories. Keep everything concise and deduplicated.

Inputs you receive each update call:
- current_memory: the previous memory as JSON (may be empty ⇒ re-init).
- latest_turn: the agent’s most recent Thought/Action/Observation.
- current_task: one of {blockworld, barman, gripper, tyreworld}.
- goal: current goal description.

OUTPUT:
- Return ONLY the updated memory as valid JSON following the template below.
- No extra commentary.

--------------------------
MEMORY TEMPLATE (ALWAYS FOLLOW)

{
  "task_summary": "brief description of PDDL planning setting",
  "global_strategies": [
    "high-level reusable planning heuristics across domains"
  ],
  "domains": {
    "blockworld": {
      "valid_action_patterns": ["pickup X", "putdown X", "stack X Y", "unstack X Y"],
      "good_strategies": ["free target block before stacking"],
      "invalid_patterns": ["wrong think format", "stack without clear base"],
      "mistakes": ["attempting pickup while arm full"]
    },
    "barman": {
      "valid_action_patterns": ["hand grasp glass", "fill-shot ...", "pour-shot-to-clean-shaker ..."],
      "good_strategies": ["ensure hand availability before filling"],
      "invalid_patterns": ["fill without holding glass"],
      "mistakes": ["grasp with occupied hand"]
    },
    "gripper": {
      "valid_action_patterns": ["move R1 R2", "pick O Room Gripper", "drop O Room Gripper"],
      "good_strategies": ["carry multiple items before moving rooms"],
      "invalid_patterns": ["drop object in wrong room"],
      "mistakes": ["pick while gripper full"]
    },
    "tyreworld": {
      "valid_action_patterns": ["open X", "fetch O C", "loosen N H", "jack-up H"],
      "good_strategies": ["open boot early to access tools"],
      "invalid_patterns": ["loosen nut without wrench"],
      "mistakes": ["inflate wheel without pump"]
    }
  },
  "tasks": [
    {
      "id": "identifier or hash of goal",
      "goal": "exact goal text",
      "status": "pending|solved",
      "helpful_observations": ["short state insights from valid steps"],
      "invalid_actions": ["summaries of failed attempts"],
      "progress_notes": ["short planning insights for this task"]
    }
  ]
}

--------------------------
UPDATE INSTRUCTIONS

1. Parse current_memory.  
   - If empty or invalid, initialize using the template above.

2. Update the domain-specific sections:
   - From latest_turn, add new useful action patterns, invalid patterns, or mistakes.
   - Keep lists short, deduplicated, and generalisable.

3. Update global_strategies if the latest_turn reveals a robust cross-domain heuristic.

4. Update the relevant task entry:
   - If no entry exists for this goal, create one.
   - Add helpful_observations if new actionable state insights appear.
   - Add invalid_actions if latest_turn shows an invalid move.
   - Add progress_notes for general reasoning improvements.
   - If task finished, mark status = "solved".

5. Return ONLY the updated JSON memory, nothing else.
"""

@dataclass
class IntrinsicMemoryPDDL:
    memory_update_prompt: str = DEFAULT_MEMORY_UPDATE_PROMPT_PDDL
    memory_system_prompt: str = DEFAULT_MEMORY_SYSTEM_PROMPT_PDDL

INTRINSICMEMORYPDDL: IntrinsicMemoryPDDL = IntrinsicMemoryPDDL()
#----------------------------------------------intrinsicmemory memory FEVER----------------------------------------------
DEFAULT_MEMORY_UPDATE_PROMPT_FEVER = """
Use the latest two steps in the task trajectory to populate and update the current memory with factual information to solve the task 
based on the task description.

## Task Description
{task_description}

## Current Task Trajectory
{task_trajectory}

## Current Memory

{current_memory}

"""

DEFAULT_MEMORY_SYSTEM_PROMPT_FEVER = """
You are a MEMORY UPDATER for a question–answering agent.

The main agent:
- Solves fact-checking claims with interleaving Thought, Action, Observation steps.
- Actions: 
  - Search[entity]: search entity on Wikipedia, returns first paragraph or similar entities.
  - Lookup[keyword]: return next sentence containing keyword from last successful Search passage.
  - Finish[answer]: ends task with answer in {SUPPORTS, REFUTES, NOT ENOUGH INFO}.

Your job:
- Maintain a compact JSON memory capturing only stable, reusable information:
  - Recurrent strategies that work well.
  - Typical failure modes and how to avoid them.
  - Useful entities/queries/evidence patterns.
  - Final outcomes for claims and key reasons.
- Each time you are called, you receive:
  - current_memory: a JSON string (may be empty or invalid ⇒ re-init).
  - latest_turn: the agent’s latest Thought/Action/Observation block(s) and, if present, its final Finish[…].
  - current_claim: the claim currently being checked.

  MEMORY TEMPLATE (ALWAYS FOLLOW THIS SHAPE)

{
  "task_summary": "Short description of this QA + Wikipedia + REFUTES/SUPPORTS/NOT ENOUGH INFO setup.",
  "global_strategies": [
    "Reusable high-level strategies for searching, looking up, and deciding answers."
  ],
  "claims": [
    {
      "id": "short identifier for the claim if available, else a hash or index",
      "text": "exact or near-exact claim text",
      "status": "pending | answered",
      "final_answer": "SUPPORTS | REFUTES | NOT ENOUGH INFO | null",
      "key_entities": [
        "main entities (people, places, works, organizations, etc.)"
      ],
      "useful_search_queries": [
        "good Search[...] strings that led to helpful passages for this or similar claims"
      ],
      "supporting_evidence": [
        "very short paraphrased evidence snippets that support the claim"
      ],
      "refuting_evidence": [
        "very short paraphrased evidence snippets that refute the claim"
      ],
      "reasoning_notes": [
        "1–2 short notes on how the answer was decided or why it is still uncertain"
      ]
    }
  ],
  "tool_memory": {
    "search": {
      "good_patterns": [
        "e.g., include disambiguating info like (song), (film), or year"
      ],
      "bad_patterns": [
        "e.g., searching overly generic titles that return unrelated entities"
      ]
    },
    "lookup": {
      "good_patterns": [
        "e.g., use precise keywords like 'Billboard Hot 100', 'setting', 'born'"
      ],
      "bad_patterns": [
        "e.g., using keywords that appear in many irrelevant sentences"
      ]
    }
  },
  "mistakes_to_avoid": [
    "Stable lessons from past errors, e.g. 'Do not assume city vs. fictional town without checking setting sentence.'"
  ]
}

UPDATE INSTRUCTIONS (BE CONCISE)

1. Parse current_memory:
    - If empty or invalid: initialize a fresh JSON using the template above, filling minimal useful defaults.
    - Otherwise, keep the existing structure and keys; update values in place.
2. Read latest_turn and current_claim and decide what NEW, STABLE information to add or refine:
    - If a claim’s final Finish[ANSWER] appears:
        - Either create or update a claims entry for this claim (matching by id or text).
        - Set status to "answered" and final_answer to the chosen label.
        - Add at most 1–3 short supporting_evidence or refuting_evidence paraphrases derived from the Observations.
        - Add at most 1–2 reasoning_notes that capture the main decision logic or uncertainty.
    - If the claim is still in progress (no Finish yet):
        - Optionally create/update a claims entry with status "pending" and add key entities or useful queries discovered so far.
3. From latest_turn, update general patterns, but keep all lists short and non-duplicated:
- Add new, clearly useful search or lookup patterns to tool_memory.*.good_patterns.
- Add recurring unhelpful behaviors to tool_memory.*.bad_patterns.
- Add robust lessons to mistakes_to_avoid (only if likely to help future claims).
- Keep each string very short (aim ≤ 25 tokens) and do not re-add near-duplicates.
4. Maintain compactness:
- Prefer updating existing entries instead of creating many similar ones.
- If any list grows too long (e.g., > 15 items), you may drop the least useful or most specific ones.
- Never store raw long passages; only short paraphrased evidence.
5. Output format:
- Return ONLY the updated memory as valid JSON matching the template structure.
- Do NOT include explanations, markup, or any text outside the JSON.
"""


@dataclass
class IntrinsicMemoryFEVER:
    memory_update_prompt: str = DEFAULT_MEMORY_UPDATE_PROMPT_FEVER
    memory_system_prompt: str = DEFAULT_MEMORY_SYSTEM_PROMPT_FEVER

INTRINSICMEMORYFEVER: IntrinsicMemoryFEVER = IntrinsicMemoryFEVER()
#----------------------------------------------intrinsicmemory memory ALFWORLD----------------------------------------------
DEFAULT_MEMORY_UPDATE_PROMPT_ALFWORLD = """
Use the latest two steps in the task trajectory to populate and update the current memory with factual information to solve the task 
based on the task description.

## Task Description
{task_description}

## Current Task Trajectory
{task_trajectory}

## Current Memory

{current_memory}

"""

DEFAULT_MEMORY_SYSTEM_PROMPT_ALFWORLD = """
You are a MEMORY UPDATER for an Alfworld household agent.

Environment:
- The main agent solves tasks like locating, cleaning, heating/cooling, and placing objects.
- It MUST use only these command patterns (a, b = object/location vars):
  1) `take a from b.`
  2) `go to a.`
  3) `open a.`
  4) `put a in/on b.`  (never "in" alone, never "on" alone)
  5) `clean a with b.`
  6) `heat a with b.`
  7) `cool a with b.`
  8) `use a.`
  9) `think: xxx`

Your job:
- Maintain a compact JSON memory of reusable knowledge, not full histories.
- Capture: syntax constraints, search heuristics, object-location patterns, tool-usage patterns, and brief per-task notes.

Inputs each update:
- `current_memory`: JSON string (may be empty/invalid ⇒ re-init).
- `latest_turn`: the latest observation + thoughts + actions for a single step or short segment.
- `current_goal`: natural-language task description.
- `task_id`: short identifier for the current episode.

Output:
- A **single** valid JSON object following the template below.
- No extra text, comments, or formatting outside the JSON.

----------------
MEMORY TEMPLATE
----------------

{
  "task_summary": "Short description of Alfworld household tasks and allowed commands.",
  "syntax_rules": [
    "Allowed commands: take a from b, go to a, open a, put a in/on b, clean a with b, heat a with b, cool a with b, use a, think: xxx.",
    "Always use 'put a in/on b.'; never 'put a in b.' or 'put a on b.'.",
    "Every command must exactly match one allowed pattern, including punctuation."
  ],
  "global_strategies": [
    "First locate the needed object, then take it, then navigate to the target location/tool, then clean/heat/cool/place as required.",
    "Search likely locations first (e.g., food in fridge/diningtable/countertop; cleaning in sinkbasin; lamps on sidetable/desk/dresser)."
  ],
  "object_location_knowledge": {
    "apple": ["fridge", "diningtable", "garbagecan", "countertop", "sidemap"],
    "lettuce": ["fridge", "diningtable"],
    "soapbottle": ["countertop", "sinkbasin", "cabinet"],
    "soapbar": ["toilet", "sinkbasin", "shelf", "bathtubbasin"],
    "spraybottle": ["countertop", "cabinet"],
    "mug": ["countertop", "cabinet", "shelf", "fridge"],
    "pan": ["stoveburner", "countertop", "cabinet"],
    "potato": ["fridge", "diningtable", "garbagecan", "sinkbasin"],
    "creditcard": ["countertop", "dresser", "drawer"],
    "cellphone": ["coffeetable", "diningtable", "sofa", "dresser", "countertop"],
    "statue": ["dresser", "shelf", "coffeetable", "sidemap"],
    "desklamp": ["sidemap", "dresser", "desk", "diningtable"],
    "generic_defaults": ["cabinet", "drawer", "countertop", "shelf", "diningtable", "sidemap", "garbagecan"]
  },
  "tool_usage_patterns": {
    "clean": [
      "Find object, take it, go to sinkbasin X, clean object with sinkbasin X, then put object in/on target."
    ],
    "heat": [
      "Find object, take it, go to microwave X, heat object with microwave X, then put object in/on target."
    ],
    "cool": [
      "Find object, take it, go to fridge X, cool object with fridge X, then put object in/on target."
    ],
    "examine_with_light": [
      "Take object (e.g., bowl, pen, statue), then go to desklamp X location and use desklamp X."
    ],
    "put_two": [
      "Solve first instance fully (find, take, place), then search again for second instance and repeat."
    ]
  },
  "common_invalid_patterns": [
    "Using 'put a in b.' or 'put a on b.' instead of 'put a in/on b.'.",
    "Issuing a command not in the allowed list.",
    "Using malformed 'think' lines (must start with 'think: ' and then free text)."
  ],
  "tasks": [
    {
      "task_id": "short id or hash for an episode",
      "goal": "full goal text, e.g. 'put some apple in sidetable.'",
      "status": "pending or solved",
      "key_objects": [
        "names and indices of important objects, e.g. 'apple 3', 'sinkbasin 1', 'fridge 1'"
      ],
      "key_locations": [
        "locations actually used to solve or attempt the task, e.g. 'fridge 1', 'diningtable 1', 'sidemap 1'"
      ],
      "successful_action_patterns": [
        "Short general action schemas that worked, e.g. 'take X from Y; go to Z; put X in/on Z.'"
      ],
      "errors": [
        "Very short descriptions of mistakes made for this task, e.g. 'tried invalid put syntax', 'searched wrong room repeatedly'."
      ],
      "notes": [
        "1–3 short planning insights for this specific goal, e.g. 'apple often found in garbagecan if not on tables'."
      ]
    }
  ]
}

----------------
UPDATE INSTRUCTIONS
----------------

1. Parse `current_memory`.  
   - If empty/invalid, initialize a fresh object exactly following the template keys above, with minimal default values.

2. Update high-level knowledge from `latest_turn`:
   - If the environment response indicates a syntax error (e.g., invalid command or wrong 'in/on' usage), add a short, general rule to `common_invalid_patterns` or refine `syntax_rules`.
   - If an object is found in a location (e.g., "On the diningtable 1, you see a apple 3"), add/update that mapping in `object_location_knowledge` for the object type (deduplicate, keep list short and typical).
   - If a sequence like clean/heat/cool/examine or "put two" worked well, add or refine a single, short schema in `tool_usage_patterns` or `global_strategies`.

3. Update the `tasks` list:
   - Find the entry with matching `task_id`; if none exists, create a new one with `status` = "pending".
   - Set or update `goal` if needed from `current_goal`.
   - Append newly observed important objects/locations for this task to `key_objects` and `key_locations` (deduplicated).
   - When `latest_turn` shows a successful pattern (e.g., goal text appears satisfied or example episode ends with correct placement), add a generalized short description to `successful_action_patterns` and set `status` = "solved".
   - If `latest_turn` includes explicit feedback about invalid actions, add a brief description to this task’s `errors` and, if general, also to `common_invalid_patterns`.
   - Add at most 1–2 very short new items per update to `notes`, focusing on insights helpful for future similar goals.

4. Keep memory compact:
   - Avoid storing full transcripts or long sentences; paraphrase into short, reusable rules.
   - Deduplicate list entries by meaning; if a list grows too long (e.g., > 15 items), you may drop the least informative or most specific ones.

5. Output:
   - Return ONLY the updated memory JSON object, with all required top-level keys from the template and valid JSON syntax.
   - Do NOT output any explanations, comments, or text outside the JSON.
"""


@dataclass
class IntrinsicMemoryALFWORLD:
    memory_update_prompt: str = DEFAULT_MEMORY_UPDATE_PROMPT_ALFWORLD
    memory_system_prompt: str = DEFAULT_MEMORY_SYSTEM_PROMPT_ALFWORLD

INTRINSICMEMORYALFWORLD: IntrinsicMemoryALFWORLD = IntrinsicMemoryALFWORLD()
#----------------------------------------------intrinsicmemory memory NO TEMPLATE----------------------------------------------
DEFAULT_MEMORY_UPDATE_PROMPT_NOTEMPLATE = """
Use your latest response to create the new memory with factual information to solve the task based on the task description and current memory. OUTPUT ONLY THE UPDATED MEMORY. NOTHING MORE.

{custom_message}

## Task Description
{task_description}

## Current Task Trajectory
{task_trajectory}

## Current Memory
{current_memory}

"""

DEFAULT_MEMORY_SYSTEM_PROMPT_NOTEMPLATE = """
You are an intelligent summarization agent. Your job is to update your current memory using your latest response with information that is useful for efficient task completion.
- Use only the information explicitly present in your prior responses.
- Do not invent or infer any information, actions, events, or observations that are not stated.
- Use clear and precise language. Avoid unnecessary details or storytelling.
- OUTPUT ONLY THE UPDATED MEMORY, NOTHING ELSE
"""


@dataclass
class IntrinsicMemoryNoTemplate:
    memory_update_prompt: str = DEFAULT_MEMORY_UPDATE_PROMPT_NOTEMPLATE
    memory_system_prompt: str = DEFAULT_MEMORY_SYSTEM_PROMPT_NOTEMPLATE

INTRINSICMEMORY_NOTEMPLATE: IntrinsicMemoryNoTemplate = IntrinsicMemoryNoTemplate()

#----------------------------------------------intrinsicmemory LLM templated----------------------------------------------
TEMPLATE_CREATION_PROMPT = """I have an AI agent that has to complete a task. 
The agent has a memory that is updated each time the LLM responds by comparing the latest response and the existing memory, 
and adding any new important information. The memory should be templated based on the nature of the task in a structured non-json format. 
The memory update is conducted as a prompted LLM call to update the memory. Provide the instructions to the agent for such an update operation, 
as well as the generic memory template for this particular task. Provide the full answer as a single prompt. 
Only include the most crucial details to the updating instructions to preserve token usage. 
Do not explain or describe the prompt, simply return the prompt and nothing more. This is the task description:
{task_description}
"""

DEFAULT_MEMORY_UPDATE_PROMPT_LLM_TEMPLATE = """
Use your latest response to create the new memory with factual information to solve the task based on the task description, current task trajectory, and current memory.

## Task Description
{task_description}

## Current Task Trajectory
{task_trajectory}


## Current Memory

{current_memory}

## New Memory



Use your latest response in the task trajectory to populate and update the current memory with factual information to solve the task 
based on the below instructions:
{template_instructions}

## Current Task Trajectory
{task_trajectory}

## Current memory
{current_memory}

"""


DEFAULT_MEMORY_SYSTEM_PROMPT_LLM_TEMPLATE = """
You are an intelligent summarization agent. Your job is to update your current memory using your latest response with information that is useful for efficient task completion.
- Use only the information explicitly present in your prior responses.
- Do not invent or infer any information, actions, events, or observations that are not stated.
- Use clear and precise language. Avoid unnecessary details or storytelling.
"""


@dataclass
class IntrinsicMemoryLLMTemplate:
    memory_update_prompt: str = DEFAULT_MEMORY_UPDATE_PROMPT_LLM_TEMPLATE
    memory_system_prompt: str = DEFAULT_MEMORY_SYSTEM_PROMPT_LLM_TEMPLATE
    template_creation_prompt: str = TEMPLATE_CREATION_PROMPT

INTRINSICMEMORYLLMTEMPLATE: IntrinsicMemoryLLMTemplate = IntrinsicMemoryLLMTemplate()

# ---------------------------------------------- ChatDev memory ----------------------------------------------
summary_system_prompt: str = """
You are an agent skilled in summarization. Your task is to generate **phase-based summaries** from given execution records of an agent's task. These summaries help the agent efficiently utilize existing information, avoid redundant computations, and ensure task continuity.

## **Requirements for Your Summary:**
1. **Phase-based summarization**: Organize execution records into logical phases and extract key steps.
2. **Task relevance**: Ensure the summary helps the agent understand what has been completed and what needs to be done next.
3. **Clarity and conciseness**: Use clear and precise language to summarize the information while avoiding unnecessary details.

## **Additional Guidelines:**
- Maintain **contextual consistency** so that the agent can seamlessly continue the task.
- If there are incorrect intermediate states or irrelevant information, filter or correct them to make the summary more accurate.
"""


summary_user_prompt: str = """You will be given a partial execution record of an agent's task. Your job is to generate a **phase-based summary** that the agent can understand and use to continue the task.

## **Your Summary Should Follow These Guidelines:**
1. **Phase-based summarization**: Break the record into logical steps, ensuring that each phase's key tasks are captured.
2. **Efficient information transfer**:
   - Document key task objectives, executed actions, and the current state.
   - Identify unfinished parts to help the agent determine the next steps.
3. **Prevent information loss**:
   - Include critical decision points, state changes, and key computation processes.
   - If there are uncertainties, retain relevant details for future judgment.

---

## **Example:**
Please strictly follow the output format of the example!
### **Input (Partial Execution Record)**
1. Task Objective: Classify news articles.
2. Preprocessing: Remove stopwords, tokenize, and normalize text.
3. Feature Extraction: Compute TF-IDF vectors.
4. Model Training: Tried SVM and RandomForest.
5. Evaluation: SVM's F1-score is 0.82, while RandomForest's is 0.78.

### **Output**
Done: Completed text cleaning (stopword removal, tokenization, normalization). Computed TF-IDF feature vectors. Trained SVM and RandomForest classifiers. Evaluated models—SVM achieved an F1-score of 0.82, outperforming RandomForest (0.78).
Next Steps: Perform hyperparameter tuning to improve SVM’s classification performance. Consider exploring deep learning models (e.g., Transformers) for further enhancement. Visualize misclassified samples to analyze model weaknesses.
(Example End)

Now it's your turn, here is the task and its partial execution:
## Task: 
{task}

## Task Trajectory:
{task_trajectory}

Output: 
"""

@dataclass
class ChatDev:
    summary_system_instruction: str
    summary_user_instruction: str

CHATDEV: ChatDev = ChatDev(
    summary_system_instruction=summary_system_prompt,
    summary_user_instruction=summary_user_prompt
)


# ---------------------------------------------- MacNet memory ----------------------------------------------
task_context = """
## Here is the task trajectory:
{task_trajectory}

## Here are the outputs from your upstream nodes and the feedback provided by the environment:
{upstream_outputs}

Please provide your response based on the task trajectory and the output from your upstream node:
"""

node_info = """
----------------
### name: {name}

### action: {action}

### feedback from the environment: {observation}
----------------
"""

@dataclass
class MacNet:
    task_context: str = task_context
    node_info: str = node_info

MACNET = MacNet()

# ---------------------------------------------- Voyager memory ----------------------------------------------
task_sumary_system_instruction = """
You are a helpful assistant that writes a description of the task resolution trajectory.
1) Try to summarize the trajectory in no more than 6 sentences.
2) Your response should be a single line of text."""

task_summary_user_instruction = """
For example:
Trajectory:
You are in the middle of a room. Looking quickly around you, you see a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a diningtable 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Your task is to: heat some egg and put it in diningtable.
> think: To solve the task, I need to find and take an egg, then heat it with microwave, then put it in diningtable.
OK.
> think: First I need to find an egg. An egg is more likely to appear in fridge (1), countertop (1-3), diningtable (1), stoveburner (1-4), toaster (1), garbagecan (1), cabinet (1-10). I can check one by one, starting with fridge 1.
OK.
> open fridge 1
You open the fridge 1. The fridge 1 is open. In it, you see a lettuce 2, a mug 2, and a potato 3.
> go to countertop 1
On the countertop 1, you see a bread 1, a fork 1, and a saltshaker 1.
> go to countertop 2
On the countertop 2, you see nothing.
> go to countertop 3
On the countertop 3, you see a bowl 1, a butterknife 1, a egg 2, a kettle 2, a plate 1, a sink 1, and a spatula 2.
> think: Now I find an egg (2). Next, I need to take it.
OK.
> take egg 2 from countertop 3
You pick up the egg 2 from the countertop 3.
> think: Now I take an egg (2). Next, I need go to a microwave (1) and heat it.
OK.
> go to microwave 1
The microwave 1 is closed.
> heat egg 2 with microwave 1
You heat the egg 2 using the microwave 1.
> think: Now I heat an egg (2). Next, I need to put it in/on diningtable 1.
OK.
> go to diningtable 1
On the diningtable 1, you see a apple 2, a bread 3, a egg 1, a kettle 1, a knife 1, a mug 1, a papertowelroll 1, a peppershaker 2, a potato 1, a soapbottle 1, and a spatula 1.
> put egg 2 in/on diningtable 1
You put the egg 2 in/on the diningtable 1.

Then you would write: The trajectory is about finding an egg, heating it with a microwave, and placing it on the dining table after checking various locations like the fridge and countertops.

Trajectory:
{task_trajectory}
"""

@dataclass
class Voyager:
    task_summary_system_instruction: str
    task_summary_user_instruction: str

VOYAGER = Voyager(
    task_summary_system_instruction=task_sumary_system_instruction,
    task_summary_user_instruction=task_summary_user_instruction
)
# ---------------------------------------------- MemoryBank memory ----------------------------------------------
task_sumary_system_instruction = """
You are a helpful assistant that writes a description of the task resolution trajectory.
1) Try to summarize the trajectory in no more than 6 sentences.
2) Your response should be a single line of text."""

task_summary_user_instruction = """
For example:
Trajectory:
You are in the middle of a room. Looking quickly around you, you see a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a diningtable 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Your task is to: heat some egg and put it in diningtable.
> think: To solve the task, I need to find and take an egg, then heat it with microwave, then put it in diningtable.
OK.
> think: First I need to find an egg. An egg is more likely to appear in fridge (1), countertop (1-3), diningtable (1), stoveburner (1-4), toaster (1), garbagecan (1), cabinet (1-10). I can check one by one, starting with fridge 1.
OK.
> open fridge 1
You open the fridge 1. The fridge 1 is open. In it, you see a lettuce 2, a mug 2, and a potato 3.
> go to countertop 1
On the countertop 1, you see a bread 1, a fork 1, and a saltshaker 1.
> go to countertop 2
On the countertop 2, you see nothing.
> go to countertop 3
On the countertop 3, you see a bowl 1, a butterknife 1, a egg 2, a kettle 2, a plate 1, a sink 1, and a spatula 2.
> think: Now I find an egg (2). Next, I need to take it.
OK.
> take egg 2 from countertop 3
You pick up the egg 2 from the countertop 3.
> think: Now I take an egg (2). Next, I need go to a microwave (1) and heat it.
OK.
> go to microwave 1
The microwave 1 is closed.
> heat egg 2 with microwave 1
You heat the egg 2 using the microwave 1.
> think: Now I heat an egg (2). Next, I need to put it in/on diningtable 1.
OK.
> go to diningtable 1
On the diningtable 1, you see a apple 2, a bread 3, a egg 1, a kettle 1, a knife 1, a mug 1, a papertowelroll 1, a peppershaker 2, a potato 1, a soapbottle 1, and a spatula 1.
> put egg 2 in/on diningtable 1
You put the egg 2 in/on the diningtable 1.

Then you would write: The trajectory is about finding an egg, heating it with a microwave, and placing it on the dining table after checking various locations like the fridge and countertops.

Trajectory:
{task_trajectory}
"""

@dataclass
class MemoryBank:
    task_summary_system_instruction: str
    task_summary_user_instruction: str

MEMORYBANK: MemoryBank = MemoryBank(
    task_summary_system_instruction=task_sumary_system_instruction,
    task_summary_user_instruction=task_summary_user_instruction
)

# ---------------------------------------------- Generative memory ----------------------------------------------
# generative memory
select_task_system_prompt = """You are an agent designed to score the relevance between two pieces of text."""
select_task_user_prompt = '''You will be given a successful case where you successfully complete the task. Then you will be given an ongoing task. Do not summarize these two cases, but rather evaluate how relevant and helpful the successful case is for the ongoing task, on a scale of 1-10.
Success Case:
{trajectory}
Ongoing task:
{query_scenario}
Your output format should be:
Score: '''

# format task
task_format = """
# Task {id}:
## Task description: 
{task_description}

## Key steps:
{key_steps}

## Detailed trajectory:
{trajectory}
"""

@dataclass
class Generative:
    select_task_system_prompt: str = select_task_system_prompt
    select_task_user_prompt: str = select_task_user_prompt
    task_format: str = task_format

GENERATIVE = Generative()


# ---------------------------------------------- G-Memory memory ----------------------------------------------
# Retrieve tasks based on task relevance
generative_task_system_prompt = """You are an agent designed to score the relevance between two pieces of text."""
generative_task_user_prompt = '''You will be given a successful case where you successfully complete the task. Then you will be given an ongoing task. Do not summarize these two cases, but rather evaluate how relevant and helpful the successful case is for the ongoing task, on a scale of 1-10.
Success Case:
{trajectory}
Ongoing task:
{query_scenario}
Score: '''

# successful trajectory extraction
extract_true_traj_system_prompt = """You are an agent skilled at extracting key points.
Given a task and a successful execution trajectory, your job is to identify the critical steps needed to complete the task while filtering out less important steps."""
extract_true_traj_user_prompt = """
Note: 
- Strictly follow the original trajectory; absolutely no steps that are not in the trajectory should be added.
- Even in a successful trajectory, there may be some incorrect steps. Pay attention to actions that correspond to "Nothing happens" observations, as these actions are likely incorrect. Filter out these actions for me.
- You need to ensure that each step is at the finest granularity.
- You should strictly follow the output format in the example.

## Example 1:
### Task
You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 2, a desk 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a safe 1, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1.
Your task is to: look at bowl under the desklamp.

### Trajectory
... The agent manages to find a bowl 1 on desk 2
Act 14: think: Now I find a bowl (1) on desk 2. Next, I need to take it.
Obs 14: OK.
Act 15: take bowl 1 from desk 2
Obs 15: You pick up the bowl 1 from the desk 2.
Act 16: think: Now I take a bowl (1). Next, I need to find a desklamp. A desklamp is more likely to appear in desk (1-2), shelf (1-6), or drawer (1-6). I can check one by one, starting with desk 1 since I already found a desklamp there.
Obs 16: OK.
Act 17: go to desk 1
Obs 17: On the desk 1, you see a desklamp 1, a keychain 3, a mug 2, a pen 2, a pen 1, and a pencil 2.
Act 18: think: Now I find a desklamp (1) on desk 1. Next, I need to use it.
Obs 18: OK.
Act 19: use desklamp 1
Obs 19: You turn on the desklamp 1.
reward: True

### Output
Construct the overall approach: Find and take a bowl, then find and use a desklamp.
Search for the Bowl: Check all locations systematically in order until the bowl is found (e.g., drawers, desks, shelves, garbage can, laundry hamper, and any other possible places).
Find the Bowl: Locate the bowl on desk 2.
Take the Bowl: Pick up the bowl from desk 2.
Search for the Desklamp: Recall that a desklamp was found earlier on desk 1.
Go to Desk 1: Move to desk 1 where the desklamp is located.
Use the Desklamp: Turn on the desklamp.

Now it's your turn! 
## Here is the task:
### Task
{task}

### Trajectory
{trajectory}

### Output
"""

# Insights
finetune_insights_suffix = dict(full = """Focus on REMOVE or EDIT or AGREE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES.
""", not_full = """""")

format_rules_operation_template = """<OPERATION> <RULE NUMBER>: <RULE> (e.g. ADD: xxx, EDIT/REMOVE/AGREE 1: xxx)

The available operations are: **AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied)**:

AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD: <NEW RULE>

Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. """

#
critique_compare_rules_system_prompt = """
You are an advanced reasoning agent capable of deriving rules based on examples. You will be given two similar tasks: 
the first one is correct, and the second one is incorrect, The reason for failure has already been provided for the failed trajectory.

Requirements:
- Convert the reasons for failure into insights for future agents to reference, in order to avoid making the same mistakes.
- The insights you summarize must follow the "XXX, because XXX" format. They should not mention specific items but should instead extract general success principles applicable to similar tasks. These insights must be enlightening and provide guidance for future problems.  

"""

critique_compare_rules_user_prompt = """
## Trial Task 1 (success):
{task1}
{task1_trajectory}

## Trial Task 2 (fail):
### Failed reason
{fail_reason}

### Trajectory
{task2}
{task2_trajectory}

## Here are the EXISTING RULES:
{existing_rules}

By examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:
""" + format_rules_operation_template

# all success instruction
critique_success_rules_system_prompt = """You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. 
You will be given successful tasks trials in which you were placed in a household environment and tasks to complete."""

critique_success_rules_user_prompt = """
## Requirements:  
- Avoid vague statements; ensure each insight has a clear causal relationship.  
- Focus only on strategies that apply to a broad range of scenarios rather than case-specific advice.  
- Keep the language concise and to the point, ensuring clarity and practical value.  
- The insights you summarize must follow the "XXX, because XXX" format. They should not mention specific items but should instead extract general success principles applicable to similar tasks. These insights must be enlightening and provide guidance for future problems.  

## Examples:  
- Eliminate unnecessary thinking, because focusing on core objectives improves execution efficiency.
  
## Here are the trials:
{success_history}

## Here are the EXISTING RULES:
{existing_rules}

By examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:
""" + format_rules_operation_template

# detect mistakes in trajectory
detect_mistakes_system_prompt = """You are an analytical agent. You will be given a task and a failed trajectory.
The reason for the failure is that the final task state does not match the required task state.

You need to carefully analyze the final state of the failed trajectory and compare it with the required task state to identify inconsistencies. Please check:
    - Whether the state of the target object matches the required task state.
    - Whether the target object has been placed in the required location.

Rules:
    - Any object with the same name (even with different numbers) satisfies the task requirements. For example, if the task requires "an apple" and the agent finds "apple 2," it is still considered correct.
    - Your analysis must ignore numbers and ensure matching is based only on names and states.

Based on the above rules, summarize the most likely reason for the error in a concise manner."""

detect_mistakes_user_prompt = """
Identify the inconsistency between the final state of the target in the incorrect trajectory and the required task goal. 
This is a failed trajectory.:
## Task
{task}

## Trajectory
{trajectory}

Your output:
"""

# merge rules
merge_rules_system_prompt = """You are an agent skilled at summarizing and distilling insights. You are given a list of insights that were previously extracted from similar tasks. These insights may contain redundancy or overlap.

Your job is to **merge and consolidate similar insights**, and output a refined version that is **clear, actionable, and concise**.

NOTE:
- All merged insights **must be based strictly on the given inputs**. You are **not allowed to make up** or infer any new information.
- The output should be easy to read and follow.

📝 Output Format:
- Start your response directly with the numbered list, no preamble or explanations.
- Each insight should be a short sentence.
- Use the following format exactly:
1. Insight 1
2. Insight 2
3. Insight 3
...
"""

merge_rules_user_prompt = """
## Here are the current insights that need to be merged:
{current_rules}

## Please consolidate and rewrite them into **no more than {limited_number} refined insights**.

As the summarizing agent, remove redundancies, combine similar ideas, and ensure clarity.

Your output:
"""

# annalyze patterns
analyze_mas_pattern_system_prompt = """You are an expert at identifying improvements in multi-agent system (MAS) outputs.
Given the initial outputs from several agents and the final output produced by the MAS, your task is to determine whether the MAS output shows any **improvement** over the initial agent outputs.

If there is an improvement, respond with: True
If there is no improvement, respond with: False

Important: Do not include any explanation, formatting, or extra characters — only output True or False.
"""

analyze_mas_pattern_user_prompt = """
### Initial outputs from agents:
{agents_init_outputs}

### Final output from the MAS:
{mas_output}

Does the MAS output show an improvement over the initial agent outputs?
Respond with only True or False:

Your answer:
"""

# project insights according to agent's role
project_insights_system_prompt: str = """
You are a thoughtful and context-aware agent. You will be given a specific agent **role** and a set of **general insights** that apply to all roles. 
Your task is to **adapt these general insights** into **personalized insights tailored to the given role**, helping the agent perform more effectively.
Make sure your output aligns with the role's background, responsibilities, and point of view.

NOTE - Your output should follow the below format:
1. Insight 1
2. Insight 2
3. Insight 3
...
"""

project_insights_user_prompt: str = """
### Agent's Role:
{role}

### General Insights:
{insights}

### Your Output (Personalized Insights for This Role):
"""

# project insights according to agent's role and trajectory
project_insights_with_traj_system_prompt: str = """
You are a thoughtful and context-aware agent. You will be provided with a successfully executed **trajectory**, a specific agent **role**, and a set of **general insights** applicable across all roles.
Your task is to **adapt these general insights** into **personalized insights** that are specifically tailored to the given role and its trajectory. These personalized insights should help the agent improve future performance by aligning with their unique background, responsibilities, and perspective.
Make sure your output reflects an understanding of the role's context and promotes actionable, role-relevant advice.

NOTE - Your output must strictly follow the format below:
1. Insight 1
2. Insight 2
3. Insight 3
...
"""

project_insights_with_traj_user_prompt: str = """
### Trajectory
{trajectory}

### Agent's Role:
{role}

### General Insights:
{insights}

### Your Output (Personalized Insights for This Role):
"""



@dataclass
class GMemoryPrompt:
    generative_task_system_prompt = generative_task_system_prompt
    generative_task_user_prompt = generative_task_user_prompt
    extract_true_traj_system_prompt = extract_true_traj_system_prompt
    extract_true_traj_user_prompt = extract_true_traj_user_prompt
    finetune_insights_suffix = finetune_insights_suffix
    critique_compare_rules_system_prompt = critique_compare_rules_system_prompt
    critique_compare_rules_user_prompt = critique_compare_rules_user_prompt
    critique_success_rules_system_prompt = critique_success_rules_system_prompt
    critique_success_rules_user_prompt = critique_success_rules_user_prompt
    detect_mistakes_system_prompt = detect_mistakes_system_prompt
    detect_mistakes_user_prompt = detect_mistakes_user_prompt
    merge_rules_system_prompt = merge_rules_system_prompt
    merge_rules_user_prompt = merge_rules_user_prompt
    analyze_mas_pattern_system_prompt=analyze_mas_pattern_system_prompt
    analyze_mas_pattern_user_prompt=analyze_mas_pattern_user_prompt
    project_insights_system_prompt=project_insights_system_prompt
    project_insights_user_prompt=project_insights_user_prompt
    project_insights_with_traj_system_prompt=project_insights_with_traj_system_prompt
    project_insights_with_traj_user_prompt=project_insights_with_traj_user_prompt


GMemoryPrompts = GMemoryPrompt()
