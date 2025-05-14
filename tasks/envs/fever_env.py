from typing import Any, Literal
import re
from dataclasses import dataclass

from .base_env import BaseEnv, BaseRecorder
from .utils import LangChainWiki, match_exactly

class FeverEnv(BaseEnv):
    def __init__(
        self, 
        env_config: dict[str, Any], 
        max_trials: int = 7
    ) -> None:
        
        self.env_config = env_config
        self.explorer = LangChainWiki()
        self.max_trials: int = max_trials
        
        self.reset()
        
    def set_env(self, configs: dict) -> None:
        if configs.get('answer') is None:
            raise ValueError('Please provide the answer for the question.')
        if configs.get('task') is None:
            raise ValueError('The configs dict should have the `task` attribute.')
        self.config = configs
        
        task: str = f'Claim: {self.config.get('task')}'
        return task, task
    
    def reset(self) -> None:
        self.current_task: str = None
        self.reward: float = 0
    
    def step(self, action: str) -> tuple[str, float, bool]:

        action: str = self.process_action(action)

        if self._parse_action_type(action) == 'thought':
            return 'OK.', 0, False
        
        action_type, argument = self._parse_action(action)

        if action_type == 'Finish':
            if self.success_fn(argument):
                observation = 'Answer is CORRECT'
                self.reward = 1
                return observation, 1, True

            else: 
                observation = f'Answer is INCORRECT'
                return observation, 0, True

        elif action_type == 'Search':
            while True:
                try:
                    observation = self.explorer.search(argument).strip('\n').strip()
                    self.summary = observation
                    break
                except Exception as e:
                    print(e)
                    observation = f'Cannot find corresponding pages.'
                    break
        elif action_type == 'Lookup':
            try:
                observation = self.explorer.lookup(argument).strip('\n').strip()
            except ValueError:
                observation = f'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'
        else:
            observation = 'Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'
        
        if 'Invalid Action' in observation:
            processed_reward = -1
        else:
            processed_reward = 0
        return observation, processed_reward, False
    
    @staticmethod
    def _parse_action_type(action: str) -> Literal['action', 'thought']:
        if 'thought' in action.lower():
            return 'thought'
        else:
            return 'action'
    
    @staticmethod
    def process_action(action: str) -> str:
        action = action.strip().replace('<', '').split('\n')[0]
        action = action.replace('>', '').replace('OK.', '').replace('OK', '').strip()
        
        if FeverEnv._parse_action_type(action) == 'thought':
            return action
        if ':' in action:
            action = action.split(':')[1].strip()

        return action

    @staticmethod
    def _parse_action(string: str) -> tuple[str, str]:

        pattern = r'^(\w+)\[(.+)\]$'
        match = re.match(pattern, string)
        
        if match:
            action_type = match.group(1)
            argument = match.group(2)
            return action_type, argument
        else:
            return None, None
        
    def success_fn(self, agent_ans: str) -> bool:
        return match_exactly(agent_ans, self.config.get('answer'))
    
    def feedback(self) -> tuple[float, bool, str]:
        
        feedback: str = 'You successfully finished this task.' if self.reward == 1 else 'You failed the task.'
        done = self.reward == 1

        return self.reward, done, feedback
    


@dataclass
class FeverRecorder(BaseRecorder):
    def __post_init__(self):
        super().__post_init__()
        self.task = 'hotpotqa'
        self.counts = 0
        self.dones = 0
        self.rewards = 0
    
    def task_begin(self, task_id: int, task_config: dict):
        super().task_begin(task_id, task_config)
        
        message: str = f'---------- Task: {task_id} ----------'
        self.log(message)
    
    def task_end(self, reward: float, done: bool):
        
        self.rewards += reward
        self.dones += done
        self.counts += 1

        message = f'reward: {reward}, ave reward: {self.rewards / self.counts}.\ndone: {done}, ave done: {self.dones / self.counts}'
        self.log(message)