from __future__ import annotations
from typing import List, Tuple, Dict

from mas.agents import Agent
from mas.llm import Message
from mas.reasoning import ReasoningConfig

class Node:

    def __init__(self, agent: Agent):
        assert agent is not None, "Node's agent cannot be None."
        
        self._id: str = agent.name
        self._agent: Agent = agent
        
        self._spatial_predecessors: List[Node] = []
        self._spatial_successors: List[Node] = []
        self._temporal_predecessors: List[Node] = []
        self._temporal_successors: List[Node] = []

        self._input: list[str] = []  
        self._output: list[str] = []

        self._memory: Dict[str, List[str]] = {'inputs':[], 'outputs':[]}  

        self.reasoning_config = ReasoningConfig(temperature=0, stop_strs=['\n'])
    @property
    def id(self) -> str:
        return self._id
    @property
    def role(self) -> str:

        return self._agent.profile
    
    @property
    def spatial_successors(self) -> Tuple[Node]:

        return tuple(self._spatial_successors)
    
    @property
    def spatial_predecessors(self) -> Tuple[Node]:

        return tuple(self._spatial_predecessors)

    @property
    def temporal_successors(self) -> Tuple[Node]:

        return tuple(self._temporal_successors)
        
    @property
    def temporal_predecessors(self) -> Tuple[Node]:

        return tuple(self._temporal_predecessors)
    
    @property
    def current_output(self) -> Tuple[str]:
        return tuple(self._output)

    @property
    def memory(self) -> Tuple[list, list, list]:

        return (self._memory['inputs'].copy(),   # defensive copy
                self._memory['outputs'].copy())

    @staticmethod
    def add_spatial_edge(source_node: Node, target_node: Node) -> None:

        source_node._spatial_successors.append(target_node)
        target_node._spatial_predecessors.append(source_node)

    @staticmethod
    def add_temporal_edge(source_node: Node, target_node: Node) -> None:

        source_node._temporal_successors.append(target_node)
        target_node._temporal_predecessors.append(source_node)

    @staticmethod
    def remove_spatial_edge(source_node: Node, target_node: Node) -> None:

        source_node._spatial_successors.remove(target_node)
        target_node._spatial_predecessors.remove(source_node)
        
    @staticmethod
    def remove_temporal_edge(source_node: Node, target_node: Node) -> None:

        if target_node in source_node._temporal_successors: 
            source_node._temporal_successors.remove(target_node)
            target_node._temporal_predecessors.remove(source_node)
    
    def clear_spatial_connections(self) -> None:

        for successor in self._spatial_successors:
            Node.remove_spatial_edge(self, successor)
        
        for predecessor in self._spatial_predecessors:
            Node.remove_spatial_edge(predecessor, self)
    
    def clear_temporal_connections(self) -> None:

        for successor in self._temporal_successors:
            Node.remove_temporal_edge(self, successor)
        
        for predecessor in self._temporal_predecessors:
            Node.remove_temporal_edge(predecessor, self)

    def update_memory(self):

        self._memory['inputs'].append(self._input[0])
        self._memory['outputs'].append(self._output[0])

        self._check_rep()
    
    def clear_state(self):

        self._memory['inputs'] = []
        self._memory['outputs'] = []
        self._output = []
        self._input = []

    def get_spatial_upstream_info(self) -> Dict[str, Dict[str, str]]:
        """
        Retrieve the outputs and roles of spatially connected upstream nodes.

        Raises:
            RuntimeError: If the upstream node's output is not a list.
            RuntimeError: If the upstream node has no output to retrieve.

        Returns:
            Dict[str, Dict[str, str]]: A dictionary mapping each upstream node's ID 
            to its role and first output, in the form:
                {
                    node_id: {
                        "role": str,
                        "output": str
                    }
                }
        """
        predecessors: Tuple[Node] = self.spatial_predecessors

        upstream_info = {}

        if predecessors is not None:
            for predecessor in predecessors:
                 
                if not isinstance(predecessor._output, list):
                    raise RuntimeError("The node's output must be type of list.")
                if len(predecessor._output) == 0:
                    raise RuntimeError("Unable to retrieve output from the upstream node.")
                
                predecessor_output = predecessor._output[0]  

                upstream_info[predecessor._id] = {"role": predecessor.role, "output": predecessor_output}

        return upstream_info

    def get_temporal_upstream_info(self) -> Dict[str, Dict[str, str]]:
        """
        Retrieve the outputs and roles of temporally connected upstream nodes.

        Raises:
            RuntimeError: If the upstream node's output is not a list.
            RuntimeError: If the upstream node has no output to retrieve.

        Returns:
            Dict[str, Dict[str, str]]: A dictionary mapping each upstream node's ID 
            to its role and first output, in the form:
                {
                    node_id: {
                        "role": str,
                        "output": str
                    }
                }
        """
        predecessors: Tuple[Node] = self.temporal_predecessors

        upstream_info = {}

        if predecessors is not None:
            for predecessor in predecessors:
                 
                history_outputs = predecessor._memory['outputs']

                if len(history_outputs) == 0:
                    continue

                predecessor_output = history_outputs[-1]  

                upstream_info[predecessor._id] = {"role": predecessor.role, "output": predecessor_output}

        return upstream_info
    
    def execute(self, user_message: Message, use_critic: bool) -> str:
        """
        Generate the node's response by integrating answers from its upstream nodes.

        Args:
            user_message (Message): The user's input message to the current node.
            use_critic (bool): Whether to apply a critic mechanism during reasoning.

        Returns:
            str: The generated response from the node.
        """
        self._output, self._input = [], [] 

        spatial_info: Dict[str, Dict] = self.get_spatial_upstream_info()
        temporal_info: Dict[str, Dict] = self.get_temporal_upstream_info()
            
        user_prompt: str = self._process_inputs(user_message, spatial_info, temporal_info, use_critic)
        answer: str = self._agent.response(user_prompt, self.reasoning_config)
        
        self._input = [user_message.content]
        self._output = [answer]

        self._check_rep()
        return answer
    
    def _process_inputs(self, user_message: Message, spatial_info: dict, temporal_info: dict, use_critic: bool) -> str:
        """Get input prompts for the agent

        Args:
            user_message (Message): The message object containing the user's question or input
            spatial_info (dict): Information about other agents' outputs in the current round, keyed by their UUIDs
            temporal_info (dict): Information about other agents' outputs in the previous round, keyed by their UUIDs
            use_critic (bool): Flag to determine whether to use critic's feedback on other agents' outputs

        Returns:
            str: The final prompt string compiled from the user's message and other agents' outputs
        """
        user_prompt = user_message.content

        spatial_upstream_message: str = ""
        temporal_upstream_message: str = ""

        for uuid, info in spatial_info.items():
            spatial_upstream_message += f"## Agent {uuid}, role is {info['role']}, output is:\n\n {info['output']}\n"
            if use_critic:
                critic_message: str = self._critic_upstream_agent(user_prompt, info['output'])
                spatial_upstream_message += f"## Critic's Suggestions for Improvement:{critic_message}\n\n"
        
        for uuid, info in temporal_info.items():
            temporal_upstream_message += f"## Agent {uuid}, role is {info['role']}, output is:\n\n {info['output']}\n\n"
            if use_critic:
                critic_message: str = self._critic_upstream_agent(user_prompt, info['output'])
                spatial_upstream_message += f"## Critic's Suggestions for Improvement:{critic_message}\n\n"
        
        final_prompt = ''
        if spatial_upstream_message != "":
            final_prompt += f"\n# Outputs from other agents in the current round:\n{spatial_upstream_message}\n"
            final_prompt += "-" * 20
            final_prompt += '\n'

        if temporal_upstream_message != "":
            final_prompt += f"\n# Outputs from other agents in the previous round:\n{temporal_upstream_message}\n"
            final_prompt += "-" * 20
            final_prompt += '\n'
            
        return final_prompt + user_prompt 
    
    def _critic_upstream_agent(self, task: str, agent_response: str) -> str:
        """get critic agent's response to the agent's response

        Args:
            task (str): current task
            agent_response (str): agent's response

        Returns:
            str: critic agent's response to the agent's response
        """
        from .graph_prompt import critic_system_prompt, critic_user_prompt

        user_prompt: str = critic_user_prompt.format(task=task, agent_answer=agent_response)
        messages: list[Message] = [Message('system', critic_system_prompt),
                                   Message('user', user_prompt)]
        
        return self._agent.reasoning(messages, self.reasoning_config)  

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False

        return self._id == other._id
    
    def __str__(self):        
        return f"node: {self._id}"

    def _check_rep(self) -> bool:

        if len(self._memory['inputs']) != len(self._memory['outputs']):
            raise RuntimeError("The history of the node must be synchronized.")