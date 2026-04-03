from scienceworld import ScienceWorldEnv
import re
from dataclasses import dataclass

from .base_env import BaseEnv, BaseRecorder

def build_simplification_str():
    simplifications = list()
    simplifications.append("selfWateringFlowerPots")
    simplifications.append("openContainers")
    simplifications.append("openDoors")
    simplifications.append("noElectricalAction")

    return ",".join(simplifications)


class SciworldEnv(BaseEnv):
    
    def __init__(self, env_config: dict, max_trials: int):
        server_path: str = env_config.get('server_path', None)
        self.env = ScienceWorldEnv("", server_path, envStepLimit=max_trials)
        self.max_trials = max_trials

    def set_env(self, configs: dict) -> tuple[str, str]:

        task_name: str = configs.get('task_name')
        var: int = configs.get('var')
        simplification_str = configs.get('simplification_str')
        subgoals: list[str] = configs.get('subgoals')
        modified_goal: str = configs.get('modified_goal')
        difficulty: int = configs.get('difficulty')

        if task_name is None or var is None or simplification_str is None or modified_goal is None or difficulty is None:
            raise ValueError('Wrong configs given to sciworld.')
        
        env = self.env.load(taskName=task_name, variationIdx=var, simplificationStr=simplification_str)
        self.cur_task: dict = configs
        self.selected_obs = subgoals    
        self.modified_goal = modified_goal
        self.difficulty = difficulty

        self.finished_sub_goal = [0 for i in range(len(self.selected_obs))]
        
        init_obs, _ = self.reset()
        task_main = self.env.get_task_description() + f"___{configs.get('id')}"
        task_description = f"""- Here is your start point(you should check it carefully before you start to action):\n{init_obs}\n- {modified_goal}* Hint: You should use `look around` command to get some clues. *"""
        
        self.think_counts = 0
        return task_main, task_description
    
    def reset(self) -> tuple[str, dict]:
        return self.env.reset()
    
    def inventory(self):
        return self.env.inventory()
    
    @staticmethod
    def process_action(action: str):
        action = action.strip().replace('<', '').split('\n')[0]
        action = action.replace('>', '').replace('OK.', '').replace('OK', '').strip()
        if action.startswith("ACTION:"):
            return action[7:].strip() 
        return action

    def step(self, action: str) -> tuple[str, float, bool]:

        if 'think' in action:  
            self.think_counts += 1
            think_feedback: str = 'OK.'
            reward = 0
            if self.think_counts >= 2:
                think_feedback += '* Warning: You are thinking too much without any actions!'
                reward = -1
            return think_feedback, reward, False
        
        self.think_counts = 0
        action = self.process_action(action)
        if action == "check valid actions":
            valid_actions = ", ".join(self._get_action_space())
            observation = f"Choose an action from these valid actions: {valid_actions}"
            return observation, 0, False
        else:
            observation, _, _, info = self.env.step(action)
            self._complete_sub_goals(observation)    
            done = self._check_is_done()

            if observation == "No known action matches that input.":
                reward = -1
            else:
                reward = 1 if done else 0
            return observation, reward, done
    
    def feedback(self) -> tuple[float, bool, str]:
        
        progress_rate: float = self._get_progress_rate()
        done: bool = self._check_is_done()
        if progress_rate == 1 and not done or progress_rate != 1 and done:
            raise RuntimeError('Inconsistent state of SciWorld Env.')
        
        if done:
            message: str = "You successfully finished this task!"
        else:
            finished_subgoals: str = '\n'.join([f'{i + 1}. {goal}' for i, goal in enumerate(self._get_finished_subgoals())])
            unfinished_subgoals: str = '\n'.join([f'{i + 1}. {goal}' for i, goal in enumerate(self._get_unfinished_subgoals())])
            message: str = f"""\nIn this task, you successfully finished these subgoals:\n{finished_subgoals}.\nBut you failed in the following subgoals:\n{unfinished_subgoals}""" 
        return progress_rate, done, message
    
    def _get_action_space(self, abstract=True):
        svalid_actions = []
        if abstract:
            for a in self.env.getPossibleActions():
                if "reset" not in a:
                    svalid_actions.append(a)
        else:
            valid_actions = self.env.getValidActionObjectCombinationsWithTemplates()
            forbidden_words = ["teleport",
                               "connect",
                               "dunk",
                               "eat",
                               "flush",
                               "close door",
                               ]
            for valid_action in valid_actions:
                v = valid_action['action']
                for fw in forbidden_words:
                    if fw in v:
                        break
                svalid_actions.append(valid_action['action'])
        if "check valid actions" not in svalid_actions:
            svalid_actions.append("check valid actions")
        return svalid_actions
    
    def _get_progress_rate(self) -> float:
        return sum(self.finished_sub_goal) * 1.0 / len(self.finished_sub_goal)
    
    def _get_finished_subgoals(self) -> list[str]:
  
        finished_subgoals: list[str] = []
        for i in range(len(self.finished_sub_goal)):
            if self.finished_sub_goal[i] == 1:
                finished_subgoals.append(self.selected_obs[i])
        return finished_subgoals
    
    def _get_unfinished_subgoals(self) -> list[str]:

        unfinished_subgoals: list[str] = []
        for i in range(len(self.finished_sub_goal)):
            if self.finished_sub_goal[i] == 0:
                unfinished_subgoals.append(self.selected_obs[i])
        return unfinished_subgoals
    
    @staticmethod
    def _parse_task_main(task_config: dict) -> str:

        modified_goal: str = task_config.get('modified_goal')
        id: str = task_config.get('id')

        if modified_goal is None:
            raise ValueError('task config dict must have attribute: `modified_goal`.')
        if id is None:
            raise ValueError('task config dict must have attribute: `id`.')
        
        return f'{modified_goal}___{id}'
    
    @staticmethod
    def _parse_task_description(task_config: dict, init_obs: str) -> str:

        modified_goal: str = task_config.get('modified_goal')
        if modified_goal is None:
            raise ValueError('task config dict must have attribute: `modified_goal`.')
        
        return f"## Here is your start point(you should check it carefully before you start to action):\n{init_obs}\n ## {modified_goal}"
    
    def _complete_sub_goals(self, obs: str):

        for i, pattern in enumerate(self.selected_obs):
            match = re.search(pattern, obs)
            if match:
                self.finished_sub_goal[i] = 1.

    def _check_is_done(self) -> bool:
        return sum(self.finished_sub_goal) >= len(self.selected_obs)

@dataclass
class SciworldRecorder(BaseRecorder):
    
    def __post_init__(self):
        super().__post_init__()
        self.task = 'sciworld'
        self.counts = 0
        self.dones = 0
        self.rewards = 0
    
    def task_begin(self, task_id, task_config):
        super().task_begin(task_id, task_config)
        
        message: str = f'---------- Task: {task_id} ----------'
        self.log(message)
    
    def task_end(self, reward: float, done: bool):
        
        self.rewards += reward
        self.dones += done
        self.counts += 1

        message = f'reward: {reward}, done: {done}.\nave reward: {self.rewards / self.counts}, ave done: {self.dones / self.counts}'
        self.log(message)