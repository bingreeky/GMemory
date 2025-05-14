import os
from dataclasses import dataclass
from abc import ABC

from ...utils import EmbeddingFunc
from ..common import (
    AgentMessage,
    MASMessage,
    StorageNameSpace
)
from mas.llm import LLMCallable

@dataclass
class MASMemoryBase(StorageNameSpace, ABC):

    llm_model: LLMCallable
    embedding_func: EmbeddingFunc
    
    def __post_init__(self):
        self.persist_dir: str = os.path.join(self.global_config['working_dir'], self.namespace)
        os.makedirs(self.persist_dir, exist_ok=True)
        
    # ---------------------------------- inside-trial memory ----------------------------------
    def init_task_context( 
        self, 
        task_main: str,    
        task_description: str = None,
    ) -> MASMessage:

        self.current_task_context = MASMessage(
            task_main=task_main,
            task_description=task_description
        )
        return self.current_task_context
    
    def add_agent_node(
        self, 
        agent_message: AgentMessage,
        upstream_agent_ids: list[str]
    ) -> str:
        node_id: str = self.current_task_context.add_message_to_current_state(agent_message, upstream_agent_ids)

        return node_id
    
    def move_memory_state(self, action: str, observation: str, **args) -> None:
        self.current_task_context.move_state(action, observation, **args)
    
    def save_task_context(self, label: bool, feedback: str = None) -> MASMessage:
        if self.current_task_context == None:
            raise RuntimeError('The current inside-trial memory is empty.')
        
        self.current_task_context.label = label
        if feedback is not None:
            self.current_task_context.task_description += f'\n- Environment feedback\n{feedback}\n'
        self.add_memory(self.current_task_context)

        return self.current_task_context

    def summarize(self, **kargs) -> str:
        return self.current_task_context.task_description + self.current_task_context.task_trajectory
    

    # ---------------------------------- cross-trials memory ----------------------------------
    def add_memory(self, mas_message: MASMessage):
        pass
    
    def retrieve_memory(self, **kargs) -> tuple[list, list, list]:
        return [], [], []
    
    def update_memory(self, query: str, **args) -> None:
        pass
    
    def backward(self, reward, **kwargs) -> None:
        pass
