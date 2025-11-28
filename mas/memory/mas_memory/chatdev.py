from dataclasses import dataclass
import os

from .memory_base import MASMemoryBase
from .prompt import CHATDEV
from ..common import MASMessage
from mas.llm import Message

@dataclass
class ChatDevMASMemory(MASMemoryBase):
    def __post_init__(self):
        super().__post_init__()
        os.makedirs(self.persist_dir, exist_ok=True)
        self.counter: int = 0

    def summarize(self, **kargs) -> str:   
        self.counter += 1
        if self.counter % 10 != 0:
            return super().summarize()
        
        mas_message: MASMessage = self.current_task_context

        if self.current_task_context is None:
            raise RuntimeError('The current task memory is empty.')
        
        user_prompt: str = CHATDEV.summary_user_instruction.format(
            task=mas_message.task_description,
            task_trajectory=mas_message.task_trajectory
        )
        messages: list[Message] = [Message('system', CHATDEV.summary_system_instruction), Message('user', user_prompt)]

        response: str = self.llm_model(messages)
        return self.current_task_context.task_description + '\n' + response
    
    def save_task_context(self, label: bool, feedback: str = None) -> MASMessage:
        self.counter = 0
        return super().save_task_context(label, feedback=feedback)
    
