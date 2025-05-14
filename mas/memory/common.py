from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Iterator
import networkx as nx
import json
from networkx.readwrite import json_graph

@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    def _index_done(self):
        pass

    def _query_done(self):
        pass


@dataclass
class AgentMessage:
    agent_name: Optional[str] = None
    system_instruction: Optional[str] = None
    user_instruction: Optional[str]  = None
    message: Optional[str] = None
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def add_extra_field(self, key: str, value: Any):
        self.extra_fields[key] = value

    def get_extra_field(self, key: str) -> Optional[Any]:
        return self.extra_fields.get(key, None) 

@dataclass
class StateChain:
    def __post_init__(self):
        initial_state = nx.DiGraph()
        initial_state.graph["name_counter"] = {}
        self._chain_of_states = [initial_state]

    def __iter__(self) -> Iterator[nx.DiGraph]:
        return iter(self.chain_of_states)
    
    def __len__(self) -> int:
        return len(self.chain_of_states)

    def add_message(self, agent_message: AgentMessage, upstream_agent_ids: list[str]) -> str:
        
        current_state: nx.DiGraph = self._get_current_state()

        agent_message_dict: dict = asdict(agent_message)
        node_id = self._generate_node_id(agent_message.agent_name)
        current_state.add_node(node_id, **agent_message_dict)
        
        for up_node_id in upstream_agent_ids:
            if not current_state.has_node(up_node_id):
                raise ValueError("Upstream node does not exist.")
            current_state.add_edge(up_node_id, node_id, edge_type='spatial')
        return node_id

    def move_state(self, action: str, observation: str, **args) -> None:

        current_state: nx.DiGraph = self._get_current_state()
        current_state.graph.update({"action": action, "observation": observation, **args})

        initial_state = nx.DiGraph()
        initial_state.graph["name_counter"] = {}
        self._chain_of_states.append(initial_state)
    
    def get_state(self, idx: int) -> nx.Graph:
        if idx >= len(self.chain_of_states) or idx < -len(self.chain_of_states):
            raise ValueError('Index out of range.')
        return self.chain_of_states[idx]

    def pop_state(self, idx: int) -> nx.Graph:
        if idx >= len(self.chain_of_states) or idx < -len(self.chain_of_states):
            raise ValueError('Index out of range.')
        return self.chain_of_states.pop(idx)
    
    @property
    def chain_of_states(self) -> list[nx.Graph]:
        return self._chain_of_states[:-1]
    
    def _generate_node_id(self, agent_name: str) -> str:
        current_state: nx.DiGraph = self._get_current_state()
        name_counter = current_state.graph["name_counter"]

        if agent_name not in name_counter:
            name_counter[agent_name] = 0
        else:
            name_counter[agent_name] += 1
        return f"{agent_name}-{name_counter[agent_name]}"

    def _get_current_state(self) -> nx.DiGraph:
        return self._chain_of_states[-1]
    
    @staticmethod
    def to_str(state_chain: "StateChain") -> str:
        return json.dumps([json_graph.node_link_data(state) for state in state_chain])

    @staticmethod
    def from_str(state_chain_str: str) -> "StateChain":
        state_chain = StateChain()
        state_chain._chain_of_states = [json_graph.node_link_graph(state_data) for state_data in json.loads(state_chain_str)]
        return state_chain


@dataclass
class MASMessage:
    task_main: str
    task_description: Optional[str] = None
    task_trajectory: Optional[str] = '\n\n>'
    label: Optional[bool] = None
    chain_of_states: StateChain = field(default_factory=StateChain, repr=False)
    extra_fields: dict[str, Any] = field(default_factory=dict, repr=False)
    
    def add_message_to_current_state(self, agent_message: AgentMessage, upstream_agent_ids: list[str]) -> str:
        return self.chain_of_states.add_message(agent_message, upstream_agent_ids)
    
    def move_state(self, action: str, observation: str, **args) -> None:
        self.task_trajectory += f'{action}\n{observation}\n>'
        self.chain_of_states.move_state(action, observation, **args)

    def add_extra_field(self, key: str, value: Any):
        self.extra_fields[key] = value

    def get_extra_field(self, key: str) -> Optional[Any]:
        return self.extra_fields.get(key, None)
    
    @staticmethod
    def to_dict(mas_message: "MASMessage") -> dict[str, str]:
        return {
            "task_main": mas_message.task_main,
            "task_description": mas_message.task_description,
            "task_trajectory": mas_message.task_trajectory,
            "label": mas_message.label,
            "extra_fields": json.dumps(mas_message.extra_fields),
            "state_chain": StateChain.to_str(mas_message.chain_of_states)
        }
    
    @staticmethod
    def from_dict(message_dict: dict) -> "MASMessage":
        return MASMessage(
            task_main=message_dict.get("task_main"),
            task_description=message_dict.get("task_description"),
            task_trajectory=message_dict.get("task_trajectory"),
            label=message_dict.get("label"),
            extra_fields=json.loads(message_dict.get("extra_fields", "{}")),
            chain_of_states=StateChain.from_str(message_dict.get('state_chain'))
        )