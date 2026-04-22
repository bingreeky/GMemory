from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import re

from .memory_base import MASMemoryBase
from .prompt import GENERATIVE
from ..common import MASMessage
from mas.llm import Message

@dataclass
class GenerativeMASMemory(MASMemoryBase):

    def __post_init__(self):
        super().__post_init__()

        self.main_memory = Chroma(          
            embedding_function=self.embedding_func,      # caculate the vector(embedding)
            persist_directory=self.persist_dir           # store directory
        )
    
    def add_memory(self, mas_message: MASMessage) -> None:
        """
        Add a MAS message to the main memory if it has a valid label.

        Args:
            mas_message (MASMessage): The message representing a completed task to be stored.

        Raises:
            ValueError: If the MAS message does not have a valid label (True/False).
        """
        meta_data: dict = MASMessage.to_dict(mas_message)
        memory_doc = Document(
            page_content=mas_message.task_main,   
            metadata=meta_data
        )
        if mas_message.label == True or mas_message.label == False:
            self.main_memory.add_documents([memory_doc])
        else:
            raise ValueError('The mas_message must have label!')
        
        self._index_done()
    

    def _retrieve_memory_raw(
        self, 
        query_task: str, 
        successful_topk: int = 1, 
        failed_topk: int = 1, 
    ) -> tuple[list[MASMessage], list[MASMessage]]:
        
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
        
        return true_task_messages, false_task_messages
    

    def retrieve_memory(
        self, 
        query_task: str,         
        successful_topk: int = 1, 
        failed_topk: int = 1,
        **kargs
    ) -> tuple[list, list, list]:
        """
        Retrieve and rank relevant memory trajectories for a given task.

        This method fetches raw successful and failed task trajectories similar to the query task.
        For successful trajectories, it computes an importance score using an LLM by prompting it
        to assess task relevance. The top-K most important successful tasks and the top-K failed tasks
        are then returned.

        Args:
            query_task (str): The task to be used as a query.
            successful_topk (int, optional): Number of top successful trajectories to return. Defaults to 2.
            failed_topk (int, optional): Number of top failed trajectories to return. Defaults to 1.
            **kargs: Additional arguments (currently unused).

        Returns:
            tuple[list, list, list]: A tuple containing:
                - Top successful task trajectories
                - Top failed task trajectories
                - An empty list placeholder for future insights
        """
        successful_task_trajectories, failed_task_trajectories = self._retrieve_memory_raw(
            query_task, 2 * successful_topk, 2 * failed_topk)
        
        importance_score: list[float] = []
        for success_task in successful_task_trajectories:
            prompt: str = GENERATIVE.select_task_user_prompt.format(
                trajectory=success_task.task_description + '\n' + success_task.task_trajectory,
                query_scenario=query_task
            )
            response: str = self.llm_model(messages=[Message('system', GENERATIVE.select_task_system_prompt), Message('user', prompt)])
            score = int(re.search(r'\d+', response).group()) if re.search(r'\d+', response) else 0
            importance_score.append(score)
        
        sorted_success_tasks = [task for _, task in sorted(zip(importance_score, successful_task_trajectories), 
                                                           key=lambda x: x[0], reverse=True)]

        top_success_task_trajectories = sorted_success_tasks[:successful_topk]
        
        top_fail_task_trajectories = failed_task_trajectories[:failed_topk]

        return top_success_task_trajectories, top_fail_task_trajectories, [] 
