from dataclasses import dataclass
from collections import Counter
import math

from mas.agents import Agent
from mas.memory.common import MASMessage, AgentMessage
from mas.mas import MetaMAS
from mas.reasoning import ReasoningBase, ReasoningConfig
from mas.memory import MASMemoryBase, GMemory
from mas.agents import Env

from .neuron import LLMNeuron, LLMEdge
from .dylan_prompt import (
    VALID_ROLES, 
    get_role_system_prompt,
    decision_user_prompt,
    ranker_user_prompt
)
from ..format import format_task_prompt_with_insights, format_task_context


@dataclass
class DyLAN(MetaMAS):
    class NeuronGrid:
        def __init__(self, width: int, height: int):
            self.width = width
            self.height = height
            self.grid = [
                [None for y in range(height)]
                for x in range(width)
            ]

        def __getitem__(self, x) -> list[LLMNeuron]:
            if not (0 <= x < self.width):
                raise IndexError("Column index out of range.")
            return self.grid[x]
        
        def add_neuron(self, neuron: LLMNeuron, col: int, row: int) -> None:
            if not (0 <= col < self.width and 0 <= row < self.height):
                raise IndexError("Grid index out of range.")
            self.grid[col][row] = neuron
            

        def get(self, col: int, row: int) -> LLMNeuron:
            if not (0 <= col < self.width and 0 <= row < self.height):
                raise IndexError("Grid index out of range.")
            return self.grid[col][row]
            
    
    def __post_init__(self):
        self.observers = []    
        self.reasoning_config = ReasoningConfig(temperature=0, stop_strs=['\n'])

        self._neurons: DyLAN.NeuronGrid = None
        self._decision_neuron: LLMNeuron = None
    
    def build_system(self, reasoning: ReasoningBase, mas_memory: MASMemoryBase, env: Env, config: dict):
        
        # parse args
        self._height: int = config.get('node_num', 3)
        self._width: int = config.get('round_num', 3)
        self._learning_rate: float = config.get('learning_rate', 0.01)
        self._use_critic: bool = config.get('use_critic', False)
        self._successful_topk: int = config.get('successful_topk', 1)
        self._failed_topk: int = config.get('failed_topk', 1)
        self._insights_topk: int = config.get('insights_topk', 3)
        self._threshold: float = config.get('threshold', 0)
        self._use_projector: bool = config.get('use_projector', False)

        self._roles: list[str] = config.get('roles', ['solver'])
        self._roles = [role for role in self._roles if role in VALID_ROLES] 
        if len(self._roles) == 0:
            self._roles.append('solver')
        
        self.notify_observers(f"Configuration Loaded:")
        self.notify_observers(f"Node Number       : {self._height}")
        self.notify_observers(f"Round Number      : {self._width}")
        self.notify_observers(f"Learning Rate     : {self._learning_rate}")
        self.notify_observers(f"Roles             : {self._roles}")
        self.notify_observers(f"Use Critic        : {self._use_critic}")
        self.notify_observers(f"Successful Topk   : {self._successful_topk}")
        self.notify_observers(f"Failed Topk       : {self._failed_topk}")
        self.notify_observers(f"Insights Topk     : {self._insights_topk}")
        self.notify_observers(f"Retrieve Threshold: {self._threshold}")
        self.notify_observers(f"Use Role Projector: {self._use_projector}")
        
        # Initialize the NeuronGrid
        self._neurons = DyLAN.NeuronGrid(self._width, self._height)
        
        for w in range(self._width):
            for h in range(self._height):
                role: str = self._roles[h % len(self._roles)]
                agent = Agent(
                    name=f'{role}_{w}_{h}',   
                    role=role,
                    system_instruction=get_role_system_prompt(role),
                    reasoning_module=reasoning,
                    memory_module=None
                )
                neuron = LLMNeuron(agent)
                self.hire([agent])
                self._neurons.add_neuron(neuron, w, h)
                self._neurons.get(w, h).importance /= self._height  
        
        for w in range(1, self._width):
            for h_to in range(self._height):
                to_neuron = self._neurons.get(w, h_to)
                for h_from in range(self._height):
                    from_neuron = self._neurons.get(w - 1, h_from)

                    edge = LLMEdge(from_neuron, to_neuron)  

                    from_neuron.out_edges.add(edge)
                    to_neuron.in_edges.add(edge)
          
        decision_agent = Agent(
            name='final_decison',
            role='decison',
            system_instruction=get_role_system_prompt('decision'),
            reasoning_module=reasoning,
            memory_module=None
        )
        self._decision_neuron = LLMNeuron(decision_agent)
 
        self._ranking_llm = Agent(
            name='ranker',
            role='decison',
            system_instruction=get_role_system_prompt('ranker'),
            reasoning_module=reasoning,
            memory_module=None
        )
        
        self.set_env(env)
        self.meta_memory = mas_memory

    def schedule(self, task_config: dict):
        """
        Schedules and executes a task based on the given task configuration.

        Args:
            task_config (dict): A dictionary containing the task configuration, including the main task and description.

        Raises:
            ValueError: If `task_main` or `task_description` is missing in the task_config.

        Returns:
            tuple: The final reward and completion status of the task.
        """
        def get_state_graph_upstream_neuron_ids(neuron: LLMNeuron, upstream_neuron_ids: dict[str, str]) -> list[str]:
            upstream_ids: list[str] = []
            for edge in neuron.in_edges:
                up_neuron = edge.from_neuron
                if not up_neuron.is_active():
                    continue
                if up_neuron.id not in upstream_neuron_ids.keys():
                    raise ValueError('Upstream node should be in the `upstream_node_ids` dict.')
                upstream_ids.append(upstream_neuron_ids.get(up_neuron.id))
            
            return upstream_ids
        
        # parse args
        if task_config.get('task_main') is None:
            raise ValueError("Missing required keys `task_main` in task_config")
        if task_config.get('task_description') is None:
            raise ValueError("Missing required keys `task_description` in task_config")
        
        task_main: str = task_config.get('task_main')
        task_description: str = task_config.get('task_description')
        few_shots: list[str] =  task_config.get("few_shots", [])
        max_trials: int = task_config.get('max_trials', 3)

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
            task_description=self.meta_memory.summarize()
        )
        self.notify_observers(user_prompt)
        
        # Main loop for task execution
        for i in range(env.max_trials):

            self._reset_state()
            upstream_neuron_ids: dict[str, str] = {}  
            consensus = False
            for w in range(self._width):
                for h in range(self._height):
                    curr_neuron: LLMNeuron = self._neurons.get(w, h)
                    tries = 0
                    user_prompt: str = format_task_prompt_with_insights(
                        few_shots=few_shots, 
                        memory_few_shots=successful_shots,
                        insights=roles_rules.get(curr_neuron._agent.profile, raw_rules),
                        task_description=self.meta_memory.summarize()
                    )
                    while tries < max_trials:
                        try: 
                            action: str = curr_neuron.execute(user_prompt, use_critic=self._use_critic)
                            if action == '':
                                continue
                            action = env.process_action(action)
                            break
                        except Exception as e:
                            print(f'Error during execution of node {curr_neuron.id}: {e}')
                        tries += 1

                    agent_message: AgentMessage = AgentMessage(
                        agent_name=curr_neuron._agent.name,
                        system_instruction=curr_neuron._agent.system_instruction,
                        user_instruction=user_prompt,
                        message=action
                    )
                    current_id: str = self.meta_memory.add_agent_node( 
                        agent_message, upstream_agent_ids=get_state_graph_upstream_neuron_ids(curr_neuron, upstream_neuron_ids)
                    )
                    upstream_neuron_ids[curr_neuron.id] = current_id
                    
                    if self._reach_consensus(w):
                        consensus = True
                        break           
                if consensus:
                    break

                if self._height > 2 and w == (self._width - 1) // 2:
                    self._rank_neurons(w)
            
            # Finish one step
            raw_final_action: str = self._summary_response(w)
            final_action = env.process_action(raw_final_action)
            observation, reward, done = env.step(final_action)
            step_message: str = f'Act {i + 1}: {final_action}\nObs {i + 1}: {observation}'
            self.notify_observers(step_message)
            self.meta_memory.move_memory_state(final_action, observation, reward=reward) 

            if done:
                break

        # Final feedback and memory update
        final_reward, final_done, final_feedback = self.env.feedback()
        self.notify_observers(final_feedback)
        self.meta_memory.save_task_context(label=final_done, feedback=final_feedback)  
        
        return final_reward, final_done    

    def add_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, message: str):
        for observer in self.observers:
            observer.log(message)

    def _reach_consensus(self, col: int) -> bool:
        """
        Determine whether a consensus has been reached among the active neurons in a specific column.

        Args:
            col (int): The index of the column in the NeuronGrid to evaluate.

        Returns:
            bool: True if consensus is reached, otherwise False.
        """
        neuron_layer: list[LLMNeuron] = self._neurons[col]
        
        answers = [
            neuron.cached_answer
            for neuron in neuron_layer
            if neuron.is_active() and neuron.cached_answer is not None
        ]

        if not answers or len(answers) < 2:  
            return False 
        elif len(answers) == 2:
            return answers[0] == answers[1]

        counter = Counter(answers)
        max_count = max(counter.values())

        return max_count >= math.floor(2/3 * len(neuron_layer))
    
    def _summary_response(self, col: int) -> str:
        """
        Summarizes the responses of active neurons in the specified column of the NeuronGrid.

        Args:
            col (int): Index of the column in the NeuronGrid to summarize.

        Returns:
            str: A single summarized response. If a majority of active neurons agree,
                the common response is returned directly. Otherwise, a final decision is
                generated by aggregating and analyzing all active responses.
        """
        neuron_layer: list[LLMNeuron] = self._neurons[col]
    
        answers = [
            neuron.cached_answer
            for neuron in neuron_layer
            if neuron.is_active() and neuron.cached_answer is not None 
        ]

        if not answers:
            return "No active response."
        
        # 如果达成一致, 则使用多数的答案
        if self._reach_consensus(col=col):
            counter = Counter(answers)
            most_common_answer, count = counter.most_common(1)[0]   
            return most_common_answer
        # 否则, 使用 summary agent 进行总结
        else:
            upstream_responses: str = ''
            for i, answer in enumerate(answers):
                upstream_responses += f"Agent{i+1}: {answer}\n"
            user_prompt: str = decision_user_prompt.format(agents_responses=upstream_responses)
            final_response = self._decision_neuron.execute(user_prompt, use_critic=False)  
            return final_response

    def _rank_neurons(self, col: int) -> None:
        """
        Ranks the active neurons in the specified column based on their responses,
        and deactivates the lowest-ranked neuron.

        Args:
            col (int): Index of the column in the NeuronGrid whose neurons will be ranked.
        """
        def parse_ranks(ranks: str) -> list[int]:
            import re
            matches = re.findall(r'Agent\s*(\d+)', ranks) 
            if not matches:
                matches = re.findall(r'\b(\d+)\b', ranks)  
            return [int(idx)-1 for idx in matches]


        neuron_layer: list[LLMNeuron] = self._neurons[col]
        answers = [
            (neuron.cached_answer, neuron)
            for neuron in neuron_layer
            if neuron.is_active()
        ]

        if not answers:
            return

        upstream_responses: str = ''
        for i, (ans, _) in enumerate(answers):
            upstream_responses += f"Agent{i+1}: {ans}\n"

        user_prompt: str = ranker_user_prompt.format(agents_responses=upstream_responses)
        
        rank_output: str = self._ranking_llm.response(user_prompt, self.reasoning_config)
        rank_indices = parse_ranks(rank_output)

        if len(rank_indices) != len(answers):
            return  

        loser_idx = rank_indices[-1]
        _, loser_neuron = answers[loser_idx]
        loser_neuron.deactivate()  

    def _reset_state(self) -> None:
        """
        Resets the internal state of all neurons in the NeuronGrid.

        For each column in the grid:
            - Recalculates and normalizes the importance of each neuron in the column.
            - Reactivates each neuron and clears its cached answer.
            - Resets the weights of all incoming and outgoing edges of each neuron to 0.
        """

        for col in range(self._width):
            sum_importance: float = sum([n.importance for n in self._neurons[col]])
            for row in range(self._height):
                neuron = self._neurons.get(col, row)
                neuron.activate()
                neuron.clear_cached_answer()
                neuron.importance /= sum_importance

                for in_edge in neuron.in_edges:
                    in_edge.set_weight(0)
                for out_edge in neuron.out_edges:
                    out_edge.set_weight(0)
            

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