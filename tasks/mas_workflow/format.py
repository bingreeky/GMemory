task_solve_with_insights = """
## Successful Examples (Reference Cases)
Below are some examples of similar tasks that were successfully completed.  
Please use these as references to guide your thinking and approach to the current task:

{few_shots}
---

## Your Own Past Successes (Execution Patterns)
Here are examples of successful execution processes you've previously used on similar tasks.  
Pay special attention to the step-by-step procedures and strategies, especially when encountering obstacles:

{memory_few_shots}
---

## Key Insights from Related Tasks
The following are insights gathered during the execution of similar tasks. You may refer to them during your task execution to improve problem-solving accuracy.

{insights}
---

## Your Turn: Take Action!
Use the above examples and insights as a foundation, and now work on the following task:
{task_description}
"""

task_format: str = """
### Task description:   
{task_description}    

### Key steps:
{key_steps}

### Detailed trajectory:
{trajectory}
"""

temp = """NOTE: You must use the command `think: <your thoughts here>` if you want to think!!!
    - Right output: think: To solve the task, ...
    - Wrong output: To solve the task, ... """

def format_task_prompt_with_insights(
    few_shots: list[str],
    memory_few_shots: list[str],
    insights: list[str],
    task_description: str 
) -> str: 
    
    existing_rules_text: str = '\n'.join([f'{i}. {r}' for i, r in enumerate(insights, 1)])
    memory_few_shots: str = '\n\n'.join([f"Task {i+1}:\n{shot}" for i, shot in enumerate(memory_few_shots)])
    user_prompt: str = task_solve_with_insights.format(
        few_shots='\n'.join(few_shots),
        memory_few_shots=memory_few_shots,
        task_description=task_description,   
        insights=existing_rules_text,
    )

    return user_prompt

def format_task_context(task_description: str, task_traj: str, key_steps: str = None) -> str:

    return task_format.format(
        task_description=task_description,
        key_steps=key_steps,
        trajectory=task_traj
    )