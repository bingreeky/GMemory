from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain.docstore.document import Document

from ..common import MASMessage
from .memory_base import MASMemoryBase

@dataclass
class MetaGPTMASMemory(MASMemoryBase):
    def __post_init__(self):
        super().__post_init__()

        self.main_memory = Chroma(          
            embedding_function=self.embedding_func,  
            persist_directory=self.persist_dir       
        )
    
    def add_memory(self, mas_message: MASMessage) -> None:

        # basic write
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
    
    def retrieve_memory(
        self, 
        query_task: str,  
        successful_topk: int = 1, 
        failed_topk: int = 1, 
        **kargs
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
    