from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Optional, Dict, Any

from .agents import Agent, Env
from .reasoning import ReasoningBase
from .memory import MASMemoryBase
from mas.llm import LLMCallable

@dataclass
class MetaMAS:
    """对MAS系统中的成员进行统一管理, 实现mas的整体调度
    """
    agents_team: Dict[str, Agent] = field(default_factory=dict)  
    env: Optional[Env] = None  
    meta_memory: Optional[MASMemoryBase] = None  
    
    def hire(self, agents: Iterable[Agent]) -> None:
        for agent in agents:
            if agent.name not in self.agents_team:
                self.agents_team[agent.name] = agent  
            else:
                print(f"Agent {agent.name} is already in the team.")  

    def set_env(self, env: Env) -> None:
        self.env = env
    
    def get_agent(self, agent_name: str) -> Optional[Agent]:
        return self.agents_team.get(agent_name)
    
    @abstractmethod
    def build_system(self, reasoning: ReasoningBase, mas_memory: MASMemoryBase) -> Any:
        pass
    
    @abstractmethod
    def schedule(self, **args) -> Any:
        pass