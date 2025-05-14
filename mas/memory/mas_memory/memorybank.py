import math
import copy
from dataclasses import dataclass, field
from langchain_chroma import Chroma
from langchain.docstore.document import Document

from .memory_base import MASMemoryBase
from .prompt import MEMORYBANK
from ..common import MASMessage, AgentMessage
from mas.llm import Message


@dataclass
class MemoryForgetter:
    trajectory_time_pairs: list[tuple[AgentMessage, int]] = field(default_factory=list)
    threshold: float = 0.3

    def get_current_time(self) -> int:
        return self.current_time
    
    def add_traj_time_pair(self, agent_message: AgentMessage, time_stemp: int) -> None:
        self.trajectory_time_pairs.append((agent_message, time_stemp))
    
    def manage_memory(self) -> list[tuple[AgentMessage, int]]:
        if len(self.trajectory_time_pairs) == 0:
            return []
        
        max_time_stemp: int = self.trajectory_time_pairs[-1][1]
        self.trajectory_time_pairs = [pair for pair in self.trajectory_time_pairs 
                                      if self._forgetting_function(max_time_stemp-pair[1]) >= self.threshold]
        
        return copy.deepcopy(self.trajectory_time_pairs)
    
    def _forgetting_function(self, time_interval: float, scale: float = 1) -> float:
        return math.exp(-time_interval / 5 * scale)

    def clear(self) -> None:
        self.current_time = 0
        self.trajectory_time_pairs = []
    
    @staticmethod
    def format_task_trajectory(agent_steps: list[AgentMessage]) -> str:
        task_trajectory: str = '\n>'
        for agent_step in agent_steps:
            task_trajectory += f' {agent_step.message}\n{agent_step.get_extra_field('observation')}\n>'
        
        return task_trajectory


@dataclass
class MemoryBankMASMemory(MASMemoryBase):
    def __post_init__(self):
        super().__post_init__()

        self.main_memory = Chroma(          
            embedding_function=self.embedding_func,      # caculate the vector(embedding)
            persist_directory=self.persist_dir           # store directory
        )

        self.current_time_stemp: int = 0
        self.memory_forgetter: MemoryForgetter = MemoryForgetter()
    
    def move_memory_state(self, action: str, observation: str, **args) -> None:
        self.current_task_context.move_state(action, observation, **args)
        
        agent_message = AgentMessage(message=action)
        agent_message.add_extra_field('observation', observation)
        self.memory_forgetter.add_traj_time_pair(agent_message, self.current_time_stemp)
        self.current_time_stemp += 1
    
    def summarize(self, **kargs) -> str:
        trajectory_time_pairs: list[tuple[AgentMessage, int]] = self.memory_forgetter.manage_memory()
        agent_messages: list[AgentMessage] = [pair[0] for pair in trajectory_time_pairs]
        return self.current_task_context.task_description + MemoryForgetter.format_task_trajectory(agent_messages)

    def save_task_context(self, label: bool, feedback: str = None) -> MASMessage:
        self.current_time_stemp = 0
        self.memory_forgetter.clear()
        return super().save_task_context(label, feedback=feedback)

    def add_memory(self, mas_message: MASMessage) -> None:
        
        prompt: str = MEMORYBANK.task_summary_user_instruction.format(
            task_trajectory=mas_message.task_description+mas_message.task_trajectory
        )
        messages: list[Message] = [Message('system', MEMORYBANK.task_summary_system_instruction), Message('user', prompt)]
        response: str = self.llm_model(messages, temperature=0.1)
        mas_message.task_main = response

        meta_data: dict = MASMessage.to_dict(mas_message)
        memory_doc = Document(
            page_content=mas_message.task_main,   # 以task_main作为区分
            metadata=meta_data
        )
        if mas_message.label == True or mas_message.label == False:
            self.main_memory.add_documents([memory_doc])
        else:
            raise ValueError('The mas_message must have label!')
        
        self._index_done()
    

    def retrieve_memory(
        self, 
        query_task: str, 
        successful_topk: int = 1, 
        failed_topk: int = 1, 
        **args
    ) -> tuple[list, list, list]:

        true_tasks_doc: list[tuple[Document, float]] = []
        false_tasks_doc: list[tuple[Document, float]] = []

        if successful_topk != 0:
            true_tasks_doc = self.main_memory.similarity_search_with_score(
                query=query_task, k=successful_topk, filter={'label': True}
            )
        if failed_topk != 0:
            false_tasks_doc = self.main_memory.similarity_search_with_score(
                query=query_task, k=failed_topk, filter={'label': False}
            )
        sorted(true_tasks_doc, key=lambda x: x[1]) 
        sorted(false_tasks_doc, key=lambda x: x[1]) 

        true_task_messages: list[MASMessage] = []
        false_task_messages: list[MASMessage] = []
        for doc in true_tasks_doc:
            meta_data: dict = doc[0].metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            true_task_messages.append(mas_message)
        
        for doc in false_tasks_doc:
            meta_data: dict = doc[0].metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            false_task_messages.append(mas_message)
        
        return true_task_messages, false_task_messages, []



    


