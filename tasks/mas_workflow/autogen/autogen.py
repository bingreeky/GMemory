from dataclasses import dataclass

from mas.agents import Agent
from mas.memory.common import MASMessage, AgentMessage
from mas.mas import MetaMAS
from mas.reasoning import ReasoningBase, ReasoningConfig
from mas.memory import MASMemoryBase, GMemory
from mas.agents import Env

from .autogen_prompt import AUTOGEN_PROMPT 
from ..format import format_task_prompt_with_insights, format_task_context


@dataclass
class AutoGen(MetaMAS):   

    def __post_init__(self):

        self.solver_name: str = 'solver'
        self.ground_truth_name: str = 'ground_truth'
        self.observers = []   

        self.reasoning_config = ReasoningConfig(temperature=0, stop_strs=['\n'])

    def build_system(self, reasoning: ReasoningBase, mas_memory: MASMemoryBase, env: Env, config: dict):  

        self._successful_topk: int = config.get('successful_topk', 1)
        self._failed_topk: int = config.get('failed_topk', 1)
        self._insights_topk: int = config.get('insights_topk', 3)
        self._threshold: float = config.get('threshold', 0)
        self._use_projector: bool = config.get('use_projector', False)
        self.notify_observers(f"Successful Topk   : {self._successful_topk}")
        self.notify_observers(f"Failed Topk       : {self._failed_topk}")
        self.notify_observers(f"Insights Topk     : {self._insights_topk}")
        self.notify_observers(f"Retrieve Threshold: {self._threshold}")
        self.notify_observers(f"Use Role Projector: {self._use_projector}")

        if not isinstance(reasoning, ReasoningBase):
            raise TypeError("reasoning module must be an instance of ReasoningBase")
        if not isinstance(mas_memory, MASMemoryBase):
            raise TypeError("mas_memory module must be an instance of MASMemoryBase")
        
        solver_agent: Agent = Agent(
            name=self.solver_name, 
            role='solver', 
            system_instruction=AUTOGEN_PROMPT.solver_system_prompt,
            reasoning_module=reasoning,
            memory_module=None
        )

        ground_truth_agent: Agent = Agent(
            name=self.ground_truth_name,
            role="ground truth agent",
            system_instruction=AUTOGEN_PROMPT.ground_truth_system_prompt,
            reasoning_module=reasoning,
            memory_module=None           
        )

        env_executor = env
        
        self.hire([
            solver_agent,
            ground_truth_agent
        ])
        self.set_env(env_executor)
        self.meta_memory = mas_memory
        
    def add_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, message: str):
        for observer in self.observers:
            observer.log(message)
    
    def schedule(self, task_config: dict) -> tuple[float, bool]:
        """
        Schedules and executes a task according to the given task configuration.
        This function initializes the task context based on the configuration, retrieves relevant memories and insights,
        and then executes the task by interacting with the environment and agents. It also handles memory updates and feedback.
        
        Parameters:
        - task_config (dict): A dictionary containing the task configuration, including the main task and description.
        
        Returns:
        - tuple[float, bool]: Returns the final reward and whether the task was successfully completed.
        """
        if task_config.get('task_main') is None:
            raise ValueError("Missing required keys `task_main` in task_config")
        if task_config.get('task_description') is None:
            raise ValueError("Missing required keys `task_description` in task_config")
        
        task_main: str = task_config.get('task_main')
        task_description: str = task_config.get('task_description')
        few_shots: list[str] =  task_config.get("few_shots", [])
        
        # Initialize environment and agents
        env: Env = self.env
        solver: Agent = self.get_agent(self.solver_name)
        ground_truth: Agent = self.get_agent(self.ground_truth_name)
        env.reset()
        
        self.meta_memory.init_task_context(task_main, task_description) 
        
        # Retrieve successful trajectories and insights from memory
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
        
        # Generate initial user prompt with insights
        user_prompt: str = format_task_prompt_with_insights(
            few_shots=few_shots, 
            memory_few_shots=successful_shots,
            insights=raw_rules,
            task_description=self.meta_memory.summarize()
        )
        self.notify_observers(user_prompt)

        # Main loop for task execution
        action_history: list = [] 
        
        for i in range(env.max_trials):    
            
            user_prompt: str = format_task_prompt_with_insights(
                few_shots=few_shots, 
                memory_few_shots=successful_shots,
                insights=roles_rules.get(solver.profile, raw_rules),
                task_description=self.meta_memory.summarize()
            )
            tries = 0
            
            while tries < 3:
                try:  
                    action: str = solver.response(user_prompt, self.reasoning_config)
                    if action == '':
                        continue
                    action = env.process_action(action)
                    break
                except Exception as e:
                    print(f'Error during execution of solver agent: {e}')
                tries += 1

            name: str = solver.name
            system_instruction = solver.system_instruction
            
            if self._solver_stuck(action, action_history):
                user_prompt: str = format_task_prompt_with_insights(
                    few_shots=few_shots, 
                    memory_few_shots=successful_shots,
                    insights=roles_rules.get(ground_truth.profile, raw_rules),
                    task_description=self.meta_memory.summarize()
                )
                tries = 0
                while tries < 3:
                    try: 
                        action: str = ground_truth.response(user_prompt, self.reasoning_config)
                        if action == '':
                            continue
                        action = env.process_action(action)
                        break
                    except Exception as e:
                        print(f'Error during execution of ground truth agent: {e}')
                    tries += 1
                name: str = ground_truth.name
                system_instruction = ground_truth.system_instruction
            
            agent_message: AgentMessage = AgentMessage(
                agent_name=name,
                system_instruction=system_instruction,
                user_instruction=user_prompt,
                message=action,
            )
            self.meta_memory.add_agent_node(agent_message, upstream_agent_ids=[])

            observation, reward, done = env.step(action)
            action_history.append(action)
            
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
    
    def _solver_stuck(self, current_action: str, action_history: list[str]) -> bool:
        """
        Determines whether the agent is stuck by repeating the same action.

        If the current action is identical to the last two actions in the history,
        the agent is considered to be stuck in a loop.

        Args:
            current_action (str): The action currently being executed by the agent.
            action_history (list[str]): A chronological list of previously taken actions.

        Returns:
            bool: True if the last two actions are the same as the current action; False otherwise.
        """
        return len(action_history) >= 2 and current_action == action_history[-1] and current_action == action_history[-2]

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
        
        # Limit the number of insights per role to self._insights_topk
        for role, insights in roles_rules.items():
            roles_rules[role] = insights[:self._insights_topk]
        return roles_rules
        
            
