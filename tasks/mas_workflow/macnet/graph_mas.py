from dataclasses import dataclass
import numpy as np
from collections import deque

from mas.agents import Agent
from mas.memory.common import MASMessage, AgentMessage
from mas.mas import MetaMAS
from mas.reasoning import ReasoningBase, ReasoningConfig
from mas.memory import MASMemoryBase, GMemory
from mas.agents import Env
from mas.llm import Message

from .graph import GraphMaskInfo, gen_graph_mask_info
from .node import Node
from .graph_prompt import *
from ..format import format_task_context, format_task_prompt_with_insights

@dataclass
class MacNet(MetaMAS):
    
    def __post_init__(self):

        self.observers = []  
        self.reasoning_config = ReasoningConfig(temperature=0, stop_strs=['\n'])

        self._agent_nodes: dict[int, Node] = {}
        self._decision_node: Node = None

    def build_system(self, reasoning: ReasoningBase, mas_memory: MASMemoryBase, env: Env, config: dict):
        
        # parse configs
        graph_type: str = config.get('graph_type', 'Random')
        node_num: int = config.get('node_num', 3)
        self._use_critic: bool = config.get('use_critic', True)
        self._successful_topk: int = config.get('successful_topk', 1)
        self._failed_topk: int = config.get('failed_topk', 1)
        self._insights_topk: int = config.get('insights_topk', 3)
        self._threshold: float = config.get('threshold', 0)
        self._use_projector: bool = config.get('use_projector', False)

        self.notify_observers(f"Configuration Loaded:")
        self.notify_observers(f"Node Number       : {node_num}")
        self.notify_observers(f"Graph Type        : {graph_type}")
        self.notify_observers(f"Use Critic        : {self._use_critic}")
        self.notify_observers(f"Successful Topk   : {self._successful_topk}")
        self.notify_observers(f"Failed Topk       : {self._failed_topk}")
        self.notify_observers(f"Insights Topk     : {self._insights_topk}")
        self.notify_observers(f"Retrieve Threshold: {self._threshold}")
        self.notify_observers(f"Use Role Projector: {self._use_projector}")

        # build macnet mas
        self.compute_graph: GraphMaskInfo = gen_graph_mask_info(mode=graph_type, N=node_num)
        self._size: int = len(self.compute_graph.fixed_spatial_masks)

        self._agent_nodes, self._decision_node = self._init_nodes(reasoning)
        
        self._spatial_matrix: np.ndarray[bool] = self._construct_spatial_connection()
        self._temporal_matrix: np.ndarray[bool] = self._construct_temporal_connection()

        for node in self._agent_nodes.values():
            self.hire([node._agent])
        self.set_env(env)
        self.meta_memory = mas_memory
    
    def schedule(self, task_config: dict) -> tuple[float, bool]:
        """
        Schedules and executes a task based on the given task configuration.

        Args:
            task_config (dict): A dictionary containing the task configuration, including the main task, task description, and optional few-shot examples.

        Returns:
            final_reward: The final reward obtained upon task completion.
            final_done: A boolean indicating whether the task is completed.
        """
        def get_state_graph_upstream_node_ids(node: Node, upstream_node_ids: dict[str, str]) -> list[str]:
            upstream_ids: list[str] = []
            for node in node.spatial_predecessors:
                if node.id not in upstream_node_ids.keys():
                    raise ValueError('Upstream node should be in the `upstream_node_ids` dict.')
                upstream_ids.append(upstream_node_ids.get(node.id))
            
            return upstream_ids
        
        # parse args
        if task_config.get('task_main') is None:
            raise ValueError("Missing required keys `task_main` in task_config")
        if task_config.get('task_description') is None:
            raise ValueError("Missing required keys `task_description` in task_config")
        
        task_main: str = task_config.get('task_main')
        task_description: str = task_config.get('task_description')
        few_shots: list[str] =  task_config.get("few_shots", [])
        
        env = self.env
        env.reset()

        self.meta_memory.init_task_context(task_main, task_description)

        # retrieve
        successful_trajectories: list[MASMessage]
        insights: list[dict]
        
        successful_trajectories, _, insights = self.meta_memory.retrieve_memory(
            query_task=task_main,
            successful_topk=self._successful_topk,
            failed_topk=self._failed_topk,
            insight_windows=self._insights_topk,
            threshold=self._threshold
        )
        successful_shots: list[str] = [format_task_context(
            traj.task_description, traj.task_trajectory, traj.get_extra_field('key_steps')
        ) for traj in successful_trajectories]
        raw_rules: list[str] = [insight for insight in insights]
        roles_rules: dict[str, list[str]] = self._project_insights(raw_rules)
        
        user_prompt: str = format_task_prompt_with_insights(
            few_shots=few_shots, 
            memory_few_shots=successful_shots,
            insights=raw_rules,
            task_description=self.meta_memory.summarize(upstream_agent_ids=None)
        )
        self.notify_observers(user_prompt)
        
        # Main loop for task execution
        for i in range(env.max_trials):

            upstream_node_ids: dict[str, str] = {}   

            in_degree = {node.id: len(node.spatial_predecessors) for node in self._agent_nodes.values()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0] 
            
            # topo sorting
            while zero_in_degree_queue:  
                current_node_id = zero_in_degree_queue.pop(0) 
                curr_node: Node = self._find_agent_node_by_uuid(current_node_id)
                user_prompt: str = format_task_prompt_with_insights(
                    few_shots=few_shots, 
                    memory_few_shots=successful_shots,
                    insights=roles_rules.get(curr_node._agent.profile, raw_rules),
                    task_description=self.meta_memory.summarize(upstream_agent_ids=None)
                )
                user_message: Message = Message('user', user_prompt)

                tries = 0
                while tries < 3:
                    try:
                        action: str = curr_node.execute(user_message, use_critic=self._use_critic)     
                        if action == '':
                            continue  
                        action = self.env.process_action(action)
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                
                agent_message: AgentMessage = AgentMessage(
                    agent_name=curr_node._agent.name,
                    system_instruction=curr_node._agent.system_instruction,
                    user_instruction=user_prompt,
                    message=action
                )
                current_id: str = self.meta_memory.add_agent_node(  
                    agent_message, upstream_agent_ids=get_state_graph_upstream_node_ids(curr_node, upstream_node_ids)
                )
                upstream_node_ids[curr_node.id] = current_id

                for successor in curr_node.spatial_successors:
                    
                    if successor not in self._agent_nodes.values():
                        raise RuntimeError(f"Contains a node that has not been processed by the system: {successor}")

                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            # Finish one step
            self._update_memory()
            self._connect_decision_node()
            action = self._decision_node.execute(user_message, use_critic=False)   
            self._disconnect_dicision_node()

            action = env.process_action(action)
            observation, reward, done = env.step(action)
            step_message: str = f'Act {i + 1}: {action}\nObs {i + 1}: {observation}'
            self.notify_observers(step_message)

            self.meta_memory.move_memory_state(action, observation, reward=reward) 

            if done:
                break

        # Final feedback and memory update
        final_reward, final_done, final_feedback = self.env.feedback()
        self.notify_observers(final_feedback)
        self.meta_memory.save_task_context(label=final_done, feedback=final_feedback) 
        self.meta_memory.backward(final_done)    

        return final_reward, final_done    
    def add_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, message: str):
        for observer in self.observers:
            observer.log(message)
    
    def _update_memory(self) -> None:
        for node in self._agent_nodes.values():
            node.update_memory()

    def _find_agent_node_by_index(self, id: int) -> Node:
        return self._agent_nodes[id]
    
    def _find_agent_node_by_uuid(self, uuid: str) -> Node:
        for node in self._agent_nodes.values():
            if node.id == uuid:
                return node
        return None

    def _init_nodes(self, reasoning_module: ReasoningBase) -> tuple[dict[int, Node], Node]:

        system_node: dict[int, Node] = {}

        for index in range(self._size):

            agent: Agent = Agent(
                name=f'solver_{index}', 
                role='solver', 
                system_instruction=solver_system_prompt, 
                reasoning_module=reasoning_module,
                memory_module=None
            )
            node: Node = Node(agent) 
            system_node[index] = node
        
        decision_agent: Agent = Agent(
            name='dicision', 
            role='solver', 
            system_instruction=decision_system_prompt, 
            reasoning_module=reasoning_module
        )
        decision_node = Node(decision_agent)

        return system_node, decision_node
    
    def _clear_spatial_connection(self) -> None:
        for node in self._agent_nodes.values():
            node.clear_spatial_connections()

        self._decision_node.clear_spatial_connections()
    
    def _clear_temporal_connection(self) -> None:
        for node in self._agent_nodes.values():
            node.clear_temporal_connections()

        self._decision_node.clear_temporal_connections()

    def _connect_decision_node(self) -> None:
        for node in self._agent_nodes.values():
            Node.add_spatial_edge(node, self._decision_node)
    
    def _disconnect_dicision_node(self) -> None:
        for node in self._agent_nodes.values():
            Node.remove_spatial_edge(node, self._decision_node)


    def _construct_spatial_connection(self) -> np.ndarray[bool]: 
        """
        Construct a spatial connection matrix based on fixed spatial masks, ensuring no cycles are introduced.

        Returns:
            np.ndarray[bool]: A 2D boolean matrix where matrix[i][j] is True if node j is spatially connected to node i.
        """

        self._clear_spatial_connection() 
        spatial_matrix = np.zeros((self._size, self._size), dtype=bool)
        
        for out_node_index, in_nodes in enumerate(self.compute_graph.fixed_spatial_masks):

            for in_node_index, connection in enumerate(in_nodes):

                if connection == 0: 
                    continue

                src_node: Node = self._find_agent_node_by_index(out_node_index)
                tar_node: Node = self._find_agent_node_by_index(in_node_index)
                
                Node.add_spatial_edge(src_node, tar_node)
                if self._check_system_cycle():
                    Node.remove_spatial_edge(src_node, tar_node)
                else:
                    spatial_matrix[out_node_index][in_node_index] = 1
        
        return spatial_matrix

    def _construct_temporal_connection(self) -> np.ndarray[bool]:  
        """
        Construct a temporal connection matrix based on fixed spatial masks, ensuring no cycles are introduced.

        Returns:
            np.ndarray[bool]: A 2D boolean matrix where matrix[i][j] is True if node j is spatially connected to node i.
        """
        self._clear_temporal_connection() 
        temporal_matrix = np.zeros((self._size, self._size), dtype=bool)
        
        for out_node_index, in_nodes in enumerate(self.compute_graph.fixed_temporal_masks):

            for in_node_index, connection in enumerate(in_nodes):

                if connection == 0:  
                    continue

                src_node: Node = self._find_agent_node_by_index(out_node_index)
                tar_node: Node = self._find_agent_node_by_index(in_node_index)

                temporal_matrix[out_node_index][in_node_index] = 1
                Node.add_temporal_edge(src_node, tar_node)
        
        return temporal_matrix
        
    
    def _check_system_cycle(self) -> bool:
        """
        Check whether there is a cycle in the multi-agent system connections.

        Returns:
            bool: True if a cycle is detected, otherwise False.
        """
        frontier = deque() 
        visited = set() 
        in_degrees = {} 

        for node in self._agent_nodes.values():
            in_degrees[node.id] = len(node.spatial_predecessors)

        for node_id, in_degree in in_degrees.items():
            if in_degree == 0:
                frontier.append(node_id)
                visited.add(node_id)  

        while frontier:
            node_id = frontier.popleft() 
            node = self._find_agent_node_by_uuid(node_id)

            for succ in node.spatial_successors:
                in_degrees[succ.id] -= 1

                if in_degrees[succ.id] == 0 and succ.id not in visited:
                    frontier.append(succ.id)
                    visited.add(succ.id)

        return len(visited) != self._size

    def _project_insights(self, insights: list[str]) -> dict[str, list[str]]:
        """
        Process insights to generate a dictionary matching roles to insights, based on whether a projector is used.

        Args:
            insights (list[str]): A list of insight strings.

        Returns:
            dict[str, list[str]]: A dictionary with roles as keys and lists of insights as values.
        """
        roles_rules: dict[str, list[str]] = {}
        roles = set([agent.profile for agent in self.agents_team.values()])

        if not self._use_projector or not isinstance(self.meta_memory, GMemory):
            for role in roles:
                roles_rules[role] = insights
        else:
            for role in roles:
                roles_rules[role] = self.meta_memory.project_insights(insights, role)
        
        # ensure every role have maximum insights of self._insights_topk
        for role, insights in roles_rules.items():
            roles_rules[role] = insights[:self._insights_topk]
        return roles_rules