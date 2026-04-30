from dataclasses import dataclass
import os
import sys

from .memory_base import MASMemoryBase
from .prompt import INTRINSICMEMORYDEFAULT
from ..common import MASMessage,AgentMessage # a MASMessage, which is a specific type of message used in MAS
from mas.llm import Message, GPTChat # a "normal" message, not a MASMessage?

@dataclass
class IntrinsicMASMemory(MASMemoryBase):
    """
    IntrinsicMASMemory keeps agent-specific memory. 
    
    Task context is stored persistently in mas_message, which includes the task 
    description (mas_message.task_description) and trajectory (mas_message.task_trajectory) (i.e., the sequence of actions taken to complete the task).)

    A separate memory is used to store the agent's memory, which is updated with the latest information from the agent's messages.
    
    """
    def __post_init__(self):
        super().__post_init__()
        os.makedirs(self.persist_dir, exist_ok=True)
        self.counter: int = 0
        self.agent_intrinsic_memory: str = ""

        self.memory_update_prompt = INTRINSICMEMORYDEFAULT.memory_update_prompt
        self.memory_system_prompt = ""

    def summarize(self, solver_message="", template_instructions="") -> str:

        """UPDATE AGENT MEMORY STEP"""

        # Update agent's memory using existing agent memory, latest output from the agent, and the memory update prompt
        mas_message: MASMessage = self.current_task_context
        if self.current_task_context is None:
            raise RuntimeError('The current task memory is empty.')

        # Construct the user prompt for memory update with memory update prompt, task description, and latest agent output
        memory_update_prompt = self.memory_update_prompt.format(
                custom_message=solver_message, # I have modified this so that we can pass custom instructions and information to each agent. It now appears in the intrinsicmemory prompt for baseline and notemplate (not implemented for the other intrinsicmemory prompt variants)
                template_instructions=template_instructions, # We never pass this to summarize, so it's also empty??
                task_description=mas_message.task_description,
                task_trajectory=mas_message.task_trajectory,
                current_memory=self.agent_intrinsic_memory,
        )

        messages = [Message("system", self.memory_system_prompt), Message("user", memory_update_prompt)]

        print(f"==== MEMORY UPDATE PROMPT ==== \n{memory_update_prompt}\n==== END MEMORY UPDATE PROMPT ====\n", file=sys.stderr)

        # only summarise after some history has been built up
        if len(mas_message.task_trajectory) > 5:
            self.agent_intrinsic_memory = self.llm_model(messages)

        injection = "You can only perform one action. Output in a single line your next action"

        # Funny that summary_message becomes the new task description, which itself contains the task description. What is the differnce between the two??
        summary_message = f"""{mas_message.task_description} \n\n### Agent Memory\n 
        {self.agent_intrinsic_memory} \n\n {mas_message.task_trajectory} \n\n {injection}"""

        return summary_message


    def save_task_context(self, label: bool, feedback: str = None) -> MASMessage:
        # Task is finished so wipe current memory
        self.counter = 0
        self.agent_intrinsic_memory = ""

        # reset self.llm_model
        llm_model_name = self.llm_model.model_name
        self.llm_model = GPTChat(model_name=llm_model_name)

    # chat history
    # context
    # agent memory
    # agent memory update
    # agent memory initialisation
