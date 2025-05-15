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

solver_system_prompt: str = """You are a smart agent designed to solve problems. You MUST strictly follow the output format of other agents' output."""

decision_system_prompt: str = """You are a smart agent designed to solve problems. You MUST strictly follow the output format of other agents' output."""
