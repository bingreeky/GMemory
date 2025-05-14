from dataclasses import dataclass

from mas.agents import Agent
from mas.memory.common import MASMessage, AgentMessage
from mas.meta_mas import MetaMAS
from mas.reasoning import ReasoningBase, ReasoningConfig
from mas.memory import MASMemoryBase
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
        self.notify_observers(f"Successful Topk   : {self._successful_topk}")
        self.notify_observers(f"Failed Topk       : {self._failed_topk}")
        self.notify_observers(f"Insights Topk     : {self._insights_topk}")
        self.notify_observers(f"Retrieve Threshold: {self._threshold}")


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

        if task_config.get('task_main') is None:
            raise ValueError("Missing required keys `task_main` in task_config")
        if task_config.get('task_description') is None:
            raise ValueError("Missing required keys `task_description` in task_config")
        
        task_main: str = task_config.get('task_main')
        task_description: str = task_config.get('task_description')
        few_shots: list[str] =  task_config.get("few_shots", [])


        env: Env = self.env
        solver: Agent = self.get_agent(self.solver_name)
        ground_truth: Agent = self.get_agent(self.ground_truth_name)
        env.reset()

        self.meta_memory.init_task_context(task_main, task_description) 

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
        rules: list[str] = [insight for insight in insights]
        
        user_prompt: str = format_task_prompt_with_insights(
            few_shots=few_shots, 
            memory_few_shots=successful_shots,
            insights=rules,
            task_description=self.meta_memory.summarize()
        )
        self.notify_observers(user_prompt)


        action_history: list = [] 
        
        for i in range(env.max_trials):    
            
            user_prompt: str = format_task_prompt_with_insights(
                few_shots=few_shots, 
                memory_few_shots=successful_shots,
                insights=rules,
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
        
        final_reward, final_done, final_feedback = self.env.feedback()
        self.notify_observers(final_feedback)
        self.meta_memory.save_task_context(label=final_done, feedback=final_feedback)  
        self.meta_memory.backward(final_done)    

        return final_reward, final_done   
    
    def _solver_stuck(self, current_action: str, action_history: list[str]) -> bool:
        return len(action_history) >= 2 and current_action == action_history[-1] and current_action == action_history[-2]


            
