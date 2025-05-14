from dataclasses import dataclass

solver_system_prompt: str = """
You are a smart agent designed to solve problems.
"""

ground_truth_system_prompt: str = """
You are an agent designed to assist the solver agent. When you are called, it means the solver agent has repeatedly output the same incorrect content (It means that the solver agent is stuck in a loop of providing the same incorrect answer or approach). 
Your task is to carefully analyze the input and provide the correct answer or guidance to help the solver agent break out of the stuck state and proceed toward the correct solution.

NOTE: ** Your approach must avoid being consistent with the previous output's approach (as the previous output comes from a solver agent that has already fallen into a misconception, making it definitely wrong). **
"""



@dataclass
class AutoGenPrompt:
    solver_system_prompt: str = solver_system_prompt
    ground_truth_system_prompt: str = ground_truth_system_prompt

AUTOGEN_PROMPT = AutoGenPrompt()

