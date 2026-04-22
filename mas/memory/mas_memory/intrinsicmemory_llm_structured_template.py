from dataclasses import dataclass
import os
import sys

from .memory_base import MASMemoryBase
from .prompt import INTRINSICMEMORYLLMTEMPLATE
from .intrinsicmemory import IntrinsicMASMemory
from ..common import MASMessage,AgentMessage # a MASMessage, which is a specific type of message used in MAS
from mas.llm import Message, GPTChat # a "normal" message, not a MASMessage?

@dataclass
class IntrinsicMASMemoryLLMTemplate(IntrinsicMASMemory):
    """
    IntrinsicMASMemoryLLMTemplate keeps agent-specific memory using automatically generated structured memory templates with LLMs.
    
    Task context is stored persistently in mas_message, which includes the task 
    description (mas_message.task_description) and trajectory (mas_message.task_trajectory) (i.e., the sequence of actions taken to complete the task).)

    A separate memory is used to store the agent's memory, which is updated with the latest information from the agent's messages.
    
    """
    def __post_init__(self):
        super().__post_init__()
        os.makedirs(self.persist_dir, exist_ok=True)
        self.counter: int = 0
        self.agent_intrinsic_memory: str = ""
        self.memory_template: str = ""
        self.memory_template_flag: bool = False

        #self.memory_update_prompt = INTRINSICMEMORYLLMTEMPLATE.memory_update_prompt
        self.memory_system_prompt = INTRINSICMEMORYLLMTEMPLATE.memory_system_prompt


    def summarize(self, solver_message="") -> str:
        mas_message: MASMessage = self.current_task_context
        if not self.memory_template_flag:
            # Generate initial memory template if this is the first time running summarize
            print("\n==== Generating initial memory template... ====\n", file=sys.stderr)

            template_creation_message: str = INTRINSICMEMORYLLMTEMPLATE.template_creation_prompt.format(
                task_description = mas_message.task_description
            )

            print(f"\n==== GENERATE MEMORY TEMPLATE ====\n{template_creation_message}\n==== END GENERATE MEMORY TEMPLATE ====\n")
            msg =[Message("system", self.memory_system_prompt), Message("user", template_creation_message)]

            result = self.llm_model(msg)
            self.agent_intrinsic_memory = result
            
            print(f"[DEBUG] memory_template is empty: {not bool(self.memory_template)}", file=sys.stderr)
            self.memory_template_flag = True

        return super().summarize(solver_message, template_instructions=self.memory_template)

