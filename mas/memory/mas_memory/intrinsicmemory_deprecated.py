from dataclasses import dataclass
import os

from .memory_base import MASMemoryBase
from .prompt import INTRINSICMEMORY
from ..common import MASMessage,AgentMessage # a MASMessage, which is a specific type of message used in MAS
from mas.llm import Message # a "normal" message, not a MASMessage?

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

    def summarize(self, **kargs) -> str:
        
        """UPDATE AGENT MEMORY STEP"""

        # Update agent's memory using existing agent memory, latest output from the agent, and the memory update prompt
        mas_message: MASMessage = self.current_task_context
        if self.current_task_context is None:
            raise RuntimeError('The current task memory is empty.')
        
        # Construct the user prompt for memory update with memory update prompt, task description, and latest agent output
        MEMORY_UPDATE_PROMPT = INTRINSICMEMORY.memory_update_prompt
        TASK_DESCRIPTION = f'"TASK DESCRIPTION: {mas_message.task_description}'
        AGENT_MEMORY = f'CURRENT AGENT MEMORY: {self.agent_intrinsic_memory}'
        LATEST_AGENT_OUTPUT = f'LATEST AGENT OUTPUT: {mas_message.task_trajectory}' ###This is not quite exactly an intrinsic memory update (although all the actions are from the same agent anyway, except in the cases where the other agent may have had to be called)
        #print(f'\n---------TASK TRAJECTORY-----------\n{mas_message.task_trajectory}\n------------END OF TASK TRAJECTORY ------------\n')
        memory_update_user_prompt: list[Message] = [Message('system', MEMORY_UPDATE_PROMPT), Message('user', TASK_DESCRIPTION), Message('user', AGENT_MEMORY), Message('user', LATEST_AGENT_OUTPUT)]

        # Update the agent's intrinsic memory with the latest information
        self.agent_intrinsic_memory = self.llm_model(memory_update_user_prompt)
        intrinsic_memory = self.agent_intrinsic_memory
        #print(f"\n---------AGENT MEMORY-----------\n{intrinsic_memory}\n------------END OF AGENT MEMORY ------------\n")

        """CONTEXT CONSTRUCTION STEP"""
        DEFAULT_CONTEXT_PROMPT = INTRINSICMEMORY.memory_reply_prompt
        CHAT_HISTORY = f'\nCHAT HISTORY: {mas_message.task_trajectory}\n'
        AGENT_MEMORY = f'AGENT MEMORY: {self.agent_intrinsic_memory}\n'

        context_creation_prompt: list[Message] = [Message('system', DEFAULT_CONTEXT_PROMPT), Message('user', CHAT_HISTORY), Message('user', AGENT_MEMORY)]

        return self.llm_model(context_creation_prompt)
        # get the output from the previous task/initialiste if doesn't exist
        #  response: str = self.llm_model(messages) where messages is the previous mmemory + chat history + memory update prompt
        # context construction step: take the updated memory and and return it to the autogen loop
        """    
    def save_task_context(self, label: bool, feedback: str = None) -> MASMessage:
        self.counter = 0
        return super().save_task_context(label, feedback=feedback)
    """
    # chat history
    # context
    # agent memory
    # agent memory update
    # agent memory initialisation