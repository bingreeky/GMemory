import numpy as np

from mas.agents import Agent
from mas.llm import Message
from mas.reasoning import ReasoningConfig
from mas.utils import EmbeddingFunc

class LLMNeuron:
    def __init__(self, agent: Agent):
        self._id = agent.name
        self._agent = agent
        self.importance = 1

        self.in_edges: set[LLMEdge] = set()
        self.out_edges: set[LLMEdge] = set()

        self._cached_answer: str = None
        self._is_activate: bool = True

        self.reasoning_config = ReasoningConfig(temperature=0, stop_strs=['\n'])
        self.embedding_func = EmbeddingFunc()

    def execute(self, user_prompt: str, use_critic: bool) -> str:

        if not self.is_active():
            return None
        
        user_prompt = self._process_inputs(user_prompt, use_critic=use_critic)
        answer = self._agent.response(user_prompt, reason_config=self.reasoning_config)
        self._cached_answer = answer

        self._update_edge_weights(answer)

        return answer
    
    def is_active(self) -> bool:
        return self._is_activate

    def activate(self) -> None:
        self._is_activate = True

    def deactivate(self) -> None:
        self._cached_answer = None
        self._is_activate = False

    def clear_cached_answer(self):
        self._cached_answer = None

    @property
    def id(self) -> str:
        return self._id
    
    @property
    def role(self) -> str:
        return self._agent.profile
    
    @property
    def cached_answer(self) -> str:
        return self._cached_answer

    def _update_edge_weights(self, answer: str) -> None:

        formers = [(edge, edge.from_neuron) for edge in self.in_edges if edge.from_neuron.is_active()]
        if not formers:
            return
        
        current_vec = self.embedding_func.embed_query(answer)
        sims = []
        for edge, from_neuron in formers:
            from_vec = self.embedding_func.embed_query(from_neuron.cached_answer)
            sim = np.dot(current_vec, from_vec) / (np.linalg.norm(current_vec) * np.linalg.norm(from_vec) + 1e-8)
            sims.append(sim)

        exp_sims = np.exp(sims - np.max(sims))  
        weights = exp_sims / np.sum(exp_sims)
        for (edge, _), weight in zip(formers, weights):
            edge.weight = weight

    def _process_inputs(self, user_prompt: str, use_critic: bool) -> str:
        
        upstream_message: str = ""
        for edge in self.in_edges:
            upstream_neuron: LLMNeuron = edge.from_neuron
            if not upstream_neuron.is_active():
                continue
            upstream_message += f"## Agent {upstream_neuron.id}, role is {upstream_neuron.role}, importance score is {upstream_neuron.importance}, output is:\n\n {upstream_neuron.cached_answer}\n"
            if use_critic:
                critic_message: str = self._critic_upstream_agent(user_prompt, upstream_neuron.cached_answer)
                upstream_message += f"## Critic's Suggestions for Improvement: {critic_message}\n\n"
        
        final_prompt = ''
        if upstream_message != "":
            final_prompt += f"\n# Outputs from other agents in the current round. Carefully consider their opinions and extract valuable insights, but maintain independent thinking and avoid being misled.\n{upstream_message}\n"
            final_prompt += "-" * 20
            final_prompt += '\n'

        return final_prompt + user_prompt
    
    def _critic_upstream_agent(self, task: str, agent_response: str) -> str:
        
        from .dylan_prompt import critic_system_prompt, critic_user_prompt

        user_prompt: str = critic_user_prompt.format(task=task, agent_answer=agent_response)
        messages: list[Message] = [Message('system', critic_system_prompt),
                                   Message('user', user_prompt)]
        
        return self._agent.reasoning(messages, self.reasoning_config)  


class LLMEdge:
    def __init__(self, src: LLMNeuron, dst: LLMNeuron):
        self.weight = 0
        self.from_neuron = src
        self.to_neuron = dst
        self.from_neuron.out_edges.add(self)
        self.to_neuron.in_edges.add(self)

    def set_weight(self, weight: float):
        self.weight = weight