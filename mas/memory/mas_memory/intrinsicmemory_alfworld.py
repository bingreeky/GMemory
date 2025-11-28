from dataclasses import dataclass
import os
import sys

from .memory_base import MASMemoryBase
from .prompt import INTRINSICMEMORYALFWORLD
from .intrinsicmemory import IntrinsicMASMemory
from ..common import MASMessage,AgentMessage # a MASMessage, which is a specific type of message used in MAS
from mas.llm import Message, GPTChat # a "normal" message, not a MASMessage?

@dataclass
class IntrinsicMASMemoryALFWORLD(IntrinsicMASMemory):
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

        self.memory_system_prompt = INTRINSICMEMORYALFWORLD.memory_system_prompt
