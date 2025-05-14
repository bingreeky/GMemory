from dataclasses import dataclass, field
import os
import logging
import time
from abc import ABC, abstractmethod

from mas.agents import Env

class BaseEnv(Env, ABC):
     
    def __init__(self, env_config: dict, max_trials: int):
        pass
    
    @abstractmethod
    def set_env(self, task_config: dict) -> tuple[str, str]:
        pass
    
    @abstractmethod
    def step(self, action: str) -> tuple[str, float, bool]:
        pass

    @classmethod
    @abstractmethod
    def process_action(cls, action: str) -> str:
        pass
    
    @abstractmethod
    def feedback(self) -> tuple[float, bool, str]:
        pass


@dataclass
class BaseRecorder:

    working_dir: str = None
    namespace: str = None
    task: str = None

    def __post_init__(self):

        self.file_path = os.path.join(self.working_dir, self.namespace + '.log')
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True) 
        
        self.logger = logging.getLogger(self.namespace)
        self.logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(self.file_path)
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(console_handler)

        self.current_task_id: int = None
        self.current_task_config: dict = field(default_factory=dict)

    def task_begin(self, task_id: int, task_config: dict) -> None:
        
        self.current_task_id = task_id
        self.current_task_config = task_config

    def task_end(self, reward: float, done: bool) -> None:
        
        if self.current_task_id is None or self.current_task_config is None:
            raise RuntimeError('The task id or the task config should not be None.')
        
    def dataset_begin(self) -> None:
        
        self.start_time = time.time()

        message: str = f"=============== Task Begin ==============="
        self.log(message)

    def dataset_end(self) -> None:
        message: str = f"=============== Task End ==============="

        end_time = time.time()  
        total_time = end_time - self.start_time
        time_message: str = f"Total execution time: {total_time:.2f} seconds"
        self.log(message)
        self.log(time_message)
    
    def log(self, message: str) -> None:    

        if hasattr(self, "logger") and self.logger:
            self.logger.info(message)
        else:
            print("Logger is not initialized.")