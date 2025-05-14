from attr import dataclass
from mas.llm import LLMCallable, Message

@dataclass
class ReasoningConfig:
    temperature: float = None
    max_tokens: int = None
    stop_strs: list[str] = None
    num_comps: int = None

class ReasoningBase:
    def __init__(self, llm_model: LLMCallable):
        self.llm_model: LLMCallable = llm_model
    
    def __call__(self, prompts: list[Message], config: ReasoningConfig) -> str:
        raise NotImplementedError("This method should be implemented by subclasses.")



class ReasoningIO(ReasoningBase):

    def __call__(self, prompts: list[Message], config: ReasoningConfig) -> str:
        
        reasoning_result = self.llm_model(
            prompts,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stop_strs=config.stop_strs,
            num_comps=config.num_comps
        )
        return reasoning_result