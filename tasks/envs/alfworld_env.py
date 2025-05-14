from dataclasses import dataclass
from typing import Union, Any
import alfworld
import alfworld.agents.environment
import re

from .base_env import BaseEnv, BaseRecorder

prefixes = {  # tasks: task_type
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

def get_env_name_from_gamefile(gamefile: str) -> Union[str, None]:

    for k in prefixes.keys():
        if k in gamefile:
            return k
    return None


class AlfworldEnv(BaseEnv):
    def __init__(
        self, 
        env_config: dict[str, Any], 
        max_trials: int = 50
    ): 
        self.env_config = env_config
        self.main_env = getattr(alfworld.agents.environment, self.env_config['env']['type'])(self.env_config, train_eval=self.env_config['split'])
        
        self.max_trials: int = max_trials
        self.reset()
    
    def set_env(self, configs: dict) -> tuple[str, str]:  
        self.gamefile = configs['env_kwargs']['gamefile']
        self.env_name: str = configs['env_name']
        self.main_env.game_files = [self.gamefile]
        
        task = configs['task']
        
        self.reset()
        return self._parse_task_main(task), self._parse_task_description(task)

    def reset(self):

        self.done = False
        self.env = self.main_env.init_env(batch_size=1)
        self.env.reset()

    def step(self, action: str) -> tuple[str, float, bool]:

        action = self.process_action(action)
        observation, reward, done, info = self.env.step([action])
        def process_ob(ob):
            if ob.startswith('You arrive at loc '):
                ob = ob[ob.find('. ')+2:]    
            return ob
        
        observation = process_ob(observation[0])

        self.done = done[0]

        if 'think:' in action:
            observation = 'OK.' 
            processed_reward = -1
        elif observation == 'Nothing happens.':
            processed_reward = -1
        else:
            processed_reward = 0 if info['won'][0] == False else 1
        
        return observation, processed_reward, self.done
    
    def feedback(self) -> tuple[float, bool, str]:
        success = self.done
        reward = 1.0 if success else 0.0
        message = "You successfully finished this task!" if success else "You failed the task."
        
        return reward, success, message
    
    @staticmethod
    def process_action(action: str) -> str:
        action = action.strip().replace('<', '').split('\n')[0]
        action = action.replace('>', '').replace('OK.', '').replace('OK', '').strip()

        return action
    
    def _parse_task_main(self, task: str):
        return self.env_name + '-' + re.search(r'Your task is to:\s*(.+)', task, re.DOTALL).group(1).strip()
    @staticmethod
    def _parse_task_description(task: str) -> str:
        return task.split('___')[0]
            

@dataclass
class AlfworldRecorder(BaseRecorder):   
    
    def __post_init__(self):
        
        super().__post_init__()
        self.task = 'alfworld'
        self.counts = [0] * 6
        self.results = [0] * 6

    def task_begin(self, task_id, task_config):
        super().task_begin(task_id, task_config)
        
        message: str = f'---------- Task: {task_id} ----------'
        self.log(message)
    
    def task_end(self, reward: float, done: bool):
        gamefile: str = self.current_task_config['env_kwargs']['gamefile']
        env_name = get_env_name_from_gamefile(gamefile)
        if env_name is None:
            raise ValueError('Format of the task config is wrong.')

        for i, (k, v) in enumerate(prefixes.items()):
            if env_name == k:
                self.results[i] += done
                self.counts[i] += 1
                break
        
        message = f'done: {done}, ave done: {sum(self.results) / sum(self.counts)}'
        self.log(message)
        self.log("rs: " + str(self.results))
        self.log("cnts: " + str(self.counts))
    
    

