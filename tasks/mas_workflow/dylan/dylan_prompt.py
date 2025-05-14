VALID_ROLES: list[str] = ['solver', 'ground_truth', 'decision']

solver_system_prompt: str = """You are a smart agent designed to solve problems. 
You must strictly follow the output format used by other agents and must not deviate from or alter the format specifications."""
ground_truth_system_prompt: str = """
You are an intelligent agent equipped with common sense and real-world knowledge, specifically designed to solve problems based on ground truth.
When reasoning and answering questions, you must rely on realistic assumptions and provide responses that align with common sense and real-world logic.
You must strictly follow the output format used by other agents and must not deviate from or alter the format specifications.
"""

decision_system_prompt: str = """
You are an intelligent agent responsible for making the final decision. You will be given responses from several other agents regarding the current task.
Your role is to carefully consider all their answers and provide your own final answer based on their input.

NOTES:
- If you believe all other agents' answers are incorrect, you may provide your own independent answer.
- You must strictly adhere to the output format used by the other agents. Do not deviate from or modify the formatting rules.
"""

decision_user_prompt: str = """
## Below are the answers provided by other agents
{agents_responses}

Your output:
"""

ranker_system_prompt: str = """
You are an evaluation agent responsible for ranking the correctness of outputs provided by multiple agents. 
Given several agents' responses to a task, you must compare them and rank their correctness from most to least accurate.

Output Format:
- Your output must be in the exact format: 1 > 3 > 2 (representing the agent numbers in descending order of correctness).
- Do NOT include any extra text, explanations, or symbols outside of this format.
- The ranking should reflect only the correctness of the responses.
"""

ranker_user_prompt: str = """
## Below are the answers provided by other agents
{agents_responses}

Your output:
"""

role_map: dict = {
    'solver': solver_system_prompt,
    'ground_truth': ground_truth_system_prompt,
    'decision': decision_system_prompt,
    'ranker': ranker_system_prompt
}

def get_role_system_prompt(role: str) -> str:
    if role not in role_map.keys():
        raise ValueError('Unsupported role type.')
    return role_map.get(role)

# critic
critic_system_prompt: str = """You are a judge. Given a task and an agent's output for that task, your job is to evaluate the agent's output and give your suggestion.
NOTE: 
- If you believe the agent's answer is correct, simply output `Support`.
- If you believe the agent's answer is incorrect, provide a concise and strong suggestion.
"""

critic_user_prompt: str = """
## Task
{task}
## Agent's answer
{agent_answer}
"""