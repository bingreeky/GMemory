from dataclasses import dataclass
from typing import Any
import nltk
import json
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'tasks', 'envs', 'pddl_env')) 
sys.path.append(os.path.join(os.getcwd(), 'envs', 'pddl_env')) 

import pddlgym
from pddlgym.structs import Literal, Predicate

from ..base_env import BaseEnv, BaseRecorder

def get_all_environment_configs(game_names: list[str], label_path: str):
    def load_annotation(path):
        all_annotations = None  
        difficulty = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue
                line = json.loads(line.strip())
                if "difficulty" in line:
                    difficulty.append(line["difficulty"])
                else:
                    raise ValueError("No difficulty in annotation file")
        return all_annotations, difficulty
    
    Num_Problems = {
        "barman":20, "blockworld":10,"gripper":20, "tyreworld":10
    }
    env_configs = []
    iter_num = 0
    _, difficulties = load_annotation(label_path)
    
    for game_name in game_names:
        num_problems = Num_Problems[game_name]
        for i in range(num_problems):
            env_configs.append({
                "game_name": game_name,
                "problem_index": i,
                "difficulty": difficulties[iter_num]
            })
            
            iter_num += 1
    
    return env_configs
    

class PDDLEnv(BaseEnv):
    def __init__(
        self, 
        env_config: dict[str, Any], 
        max_trials: int = 50
    ):     
        self.max_trials = max_trials
        self.last_obs = None

    def set_env(self, configs: dict) -> tuple[str, str]: 
        nltk.download('punkt')
        
        self.game_name: str = configs.get('game_name')
        problem_index: int = configs.get('problem_index')
        if self.game_name is None or problem_index is None:
            raise ValueError('Missing attributes in configs.')
        
        self.env = pddlgym.make("PDDLEnv{}-v0".format(self.game_name.capitalize()))
        self.env.fix_problem_index(problem_index)

        self.reset()
        task_main: str = self._get_goal()
        task_description: str = f'Here is your initial observation: {self._get_obs()}\n**Here is your task: {self._get_goal()}'
        return task_main, task_description  

    def reset(self):
        obs, debug_info = self.env.reset()
        self.goal, self.init_obs, self.goal_literals = self._get_goal_and_obs(obs) # g & s0
        self.last_obs = obs
        self.infos = dict()
        self.infos["goal_literal"] = obs.goal 
        self.states = [self.init_obs]  
        self.history = [("state", self.init_obs)] 
        self.steps = 0
        
        self.infos["goal"] = self.goal
        self.infos["states"] = self.states
        self.infos["history"] = self.history
        self.infos["steps"] = self.steps
        self.infos["state"] = self.states[-1]
        
        self.reward = 0
        self.done = False
        self.won = False
    
    def step(self, action: str):

        if 'think' in action:
            return 'Ok. But you should not think too much!', -1, self.done

        if "check" in action.lower():
            obs = "Valid actions are: " + ", ".join(self._get_action_space())
            self._update_info(action, obs)
            self.infos["action_is_valid"] = True
            return self._get_obs(), self.reward, self.done
        
        if "look around" in action.lower():
            obs = [self._literal_to_text(literal).capitalize() for literal in self.last_obs.literals]
            obs = " ".join(obs)
            self._update_info(action, obs)
            self.infos["action_is_valid"] = True
            return self._get_obs(), self.reward, self.done
                
        action_literal = self._text_to_action(action)
        
        if action_literal is not None:
            obs_temp, reward, done, infos = self.env.step(action_literal)
            print(reward)
            reward = max(self.reward, self._constraint_satisfaction_metric(obs_temp.literals, self.goal_literals))
            
            if obs_temp == self.last_obs: 
                obs = "The action is not valid and therefore takes no effect. You should use `check valid actions.` command to get some clues!"
                self._update_info(action, obs)
                self.infos["action_is_valid"] = False
                return self._get_obs(), -1, self.done
            else:
                _, obs, _ = self._get_goal_and_obs(obs_temp)
                self.last_obs = obs_temp
                self._update(action, obs, reward, done, infos)
                self.infos["action_is_valid"] = True
                return self._get_obs(), 1 if self.done else 0, self.done
        else:
            obs = "The action is not valid and therefore takes no effect. You should use `check valid actions.` command to get some clues!"
            self._update_info(action, obs)
            self.infos["action_is_valid"] = False
            return self._get_obs(), -1, self.done

    def feedback(self) -> tuple[float, bool, str]:
        
        return self.reward, self.done, ''

    @staticmethod
    def process_action(action: str) -> str:
        action = action.strip().replace('<', '').split('\n')[0]
        action = action.replace('>', '').replace('OK.', '').replace('OK', '').strip()
        if action.startswith("ACTION:") or action.startswith("Action:"):
            return action[7:].strip() 
        return action

    def _get_info(self):
        return self.infos
    
    def _get_obs(self):
        return self.states[-1]
    
    def _get_goal(self):
        return self.goal
    
    def _get_history(self):
        return self.history
    
    def _get_action_space(self):
        if self.game_name in ["barman", "tyreworld"]:
            return [self._literal_to_text(literal) for literal in self.env.action_space.all_ground_literals(self.last_obs)] + ["check valid actions", "look around"]
        return [self._literal_to_text(literal) for literal in self.env.action_space.all_ground_literals(self.last_obs)] + ["check valid actions"]
    
    def _update(self, action, obs, reward, done, infos):
        for k, v in infos.items():
            self.infos[k] = v
        
        goal_literals = self.infos["goal_literal"].literals
        obs_literals = self.last_obs.literals
        self.won = True
        for literal in goal_literals:
            if literal not in obs_literals:
                self.won = False
                break
        
        self.steps += 1
        
        self.reward = reward
        self.done = done
        
        if self.won: 
            obs += " The goal is satisfied."
        
        self.history.append(("action", action))
        self.history.append(("reward", reward))
        self.history.append(("state", obs))
        self.states.append(obs)
        
        self.infos["goal"] = self.goal
        self.infos["states"] = self.states
        self.infos["history"] = self.history
        self.infos["steps"] = self.steps
        self.infos["state"] = self.states[-1]
        
    def _update_info(self, action, info):
        self.history.append(("action", action))
        self.history.append(("reward", self.reward))
        self.history.append(("state", info))
        self.states.append(info)
        
        self.steps += 1
        self.infos["goal"] = self.goal
        self.infos["states"] = self.states
        self.infos["history"] = self.history
        self.infos["steps"] = self.steps
        self.infos["state"] = self.states[-1]
    
    def _constraint_satisfaction_metric(self, obs_literals, goal_literals):
        satisfied = 0
        all = 0
        for literal in goal_literals:
            if literal in obs_literals:
                satisfied += 1
                
            all += 1
        
        return satisfied / all
    
    def _get_goal_and_obs(self, obs):
        goal = obs.goal 
        goal = [self._literal_to_text(literal) for literal in goal.literals]
        goal.sort()
        goal_text = "The goal is to satisfy the following conditions: " + ", ".join(goal) 
        
        state = obs.literals 
        if self.game_name in ["barman", "tyreworld"]:
            if self.last_obs is not None:
                state = [literal for literal in state if literal not in self.last_obs.literals]
        state_text = [self._literal_to_text(literal).capitalize() for literal in state]
        state_text.sort()
        state_text = " ".join(state_text)
        return goal_text, state_text, obs.goal.literals
    
    def _literal_to_text(self, literal):
        predicate_name = literal.predicate.name
        objects = literal.variables
        if predicate_name in predicate_map:
            predicate_format = predicate_map[predicate_name]
            objects_name = [str(obj.name) for obj in objects]
            text = predicate_format.format(*objects_name)
        else:
            text = predicate_name + " " + " ".join([str(obj.name) for obj in objects])
        objects_name = [str(obj) for obj in objects]
        return text
    
    def _text_to_action(self, text):
        text = text.lower()
        all_valid_actions = self.env.action_space.all_ground_literals(self.last_obs)
        all_valid_predicates = [action.predicate for action in all_valid_actions]
        predicates_names = {predicate.name.lower():predicate for predicate in all_valid_predicates}
        predicates_obj_nums = {predicate.name.lower(): predicate.arity for predicate in all_valid_predicates}
        
        all_valid_objects = [obj for obj in self.last_obs.objects]
        all_valid_objects_name = [str(obj) for obj in all_valid_objects] 
        all_valid_objects_id = [obj.name for obj in all_valid_objects] 

        tokens = nltk.word_tokenize(text)
        
        predicate_name = None   
        for token in tokens:
            if token in predicates_names:
                predicate_name = token
                break
        
        if predicate_name is None:
            return None
        else:
            predicate_obj_nums = predicates_obj_nums[predicate_name]
            if predicate_obj_nums == 0:
                return Literal(Predicate(predicate_name))
            else:
                objects = []
                for token in tokens:
                    if token in all_valid_objects_name:
                        objects.append(token)
                    elif token in all_valid_objects_id:
                        objects.append(all_valid_objects[all_valid_objects_id.index(token)])
                    else:
                        continue
                        
                if len(objects) > predicate_obj_nums:
                    objects = objects[:predicate_obj_nums]
                elif len(objects) < predicate_obj_nums:
                    return None
                else:
                    pass
            
        predicate = predicates_names[predicate_name]
        literal = Literal(predicate, objects)
        return literal

@dataclass
class PDDLRecorder(BaseRecorder): 
    def __post_init__(self):
        super().__post_init__()

        self.task = 'pddl'
        self.cnts: dict[str, int] = {
            "barman": 0, 
            "blockworld": 0,
            "gripper": 0, 
            "tyreworld": 0
        }
        self.dones: dict[str, int] = {
            "barman": 0, 
            "blockworld": 0,
            "gripper": 0, 
            "tyreworld": 0
        }
        self.rewards: dict[str, int] = {
            "barman": 0, 
            "blockworld": 0,
            "gripper": 0, 
            "tyreworld": 0
        }
    
    def task_begin(self, task_id, task_config):
        super().task_begin(task_id, task_config)

        game_name: str = self.current_task_config.get('game_name') 
        if game_name is None:
            raise ValueError('The task should have an attribute: `game`.')
            
        message: str = f'---------- Task: {task_id} ----------'
        self.log(message)
    
    def task_end(self, reward: float, done: bool):
        game_name: str = self.current_task_config.get('game_name') 
        if game_name is None:
            raise ValueError('The task should have an attribute: `game`.')
        
        self.cnts[game_name] += 1
        self.rewards[game_name] += reward
        self.dones[game_name] += done

        message = f'reward: {reward}, done: {done}.\nave reward: {self._get_average_reward()}, ave done: {self._get_average_done()}'
        self.log(message)

    
    def _get_average_reward(self) -> float:
        return sum(self.rewards.values()) / sum(self.cnts.values())

    def _get_average_done(self) -> float:
        return sum(self.dones.values()) / sum(self.cnts.values())
    
        
# Define the mapping of predicate names to their natural language formats  
predicate_map = {  
    # Blocks
    "on": "{} is on {}.",
    "clear": "{} is clear.",
    "arm-empty": "Your arm is empty.",
    "holding": "You are holding {}.",
    "on-table": "{} is on the table.",
    
    "putdown": "Putdown {}.",
    "stack": "Stack {} on {}.",
    "pickup": "Pickup {}.",
    "unstack": "Unstack {} from {}.",
    
    # tyreworld
    "vehicle-at": "The vehicle is at {}.",
    "changetire": "Change tire at {}",  
    "spare-in": "The spare tire is in {}.",
    "road": "There is a road from {} to {}.",
    "movecar": "Move the car to {}.",
    
    "isopen": "{} is open.",
    "closed": "{} is closed.",
    "have": "You have {}.",
    "in": "{} is in {}.",
    "loose": "The nut {} on the hub {} is loose.",
    "tight": "The nut {} on the hub {} is tight.",
    "unlocked": "{} is unlocked.",
    "on-ground": "Hub {} is on the ground.",
    "not-on-ground": "Hub {} is not on the ground.",
    "inflated": "Wheel {} is inflated.",
    "not-inflated": "Wheel {} is not inflated.",
    "fastened": "Hub {} is fastened.",
    "unfastened": "Hub {} is unfastened.",
    "free": "Hub {} is free.",
    "on": "{} is on {}.",
    "intact": "Wheel {} is intact.",
    
    "open": "Open {}.",
    "close": "Close {}.",
    "fetch": "Fetch {} from {}.",
    "put-away": "Put-away {} in {}.",
    "loosen": "Loosen the nut {} on the hub {}.",
    "tighten": "Tighten the nut {} on the hub {}.",
    "jack-up": "jack-up the hub {}.",
    "jack-down": "Jack-down the hub {}.",
    "undo": "Undo the fastening of the nut {} on the hub {}.",
    "do-up": "Do-up the nut {} on the hub {}.",
    "remove-wheel": "Remove-wheel {} from the hub {}.",
    "put-on-wheel": "put-on-wheel {} on the hub {}.",
    "inflate": "Inflate the wheel {}.",
    
    # hanoi
    "clear": "The {} is clear.",
    "move": "Move {} to {}.",
    "smaller": "{} is smaller than {}.",
    
    # gripper
    "ball": "{} is a ball. ",
    "gripper": "{} is a gripper. ",
    "at-robby": "Robby is at {}. ",
    "at": "{} is at {}. ",
    "free": "{} is free. ",
    "carry": "{} is carrying {}. ",
    "move": "Move from {} to {}. ",
    "pick": "Pick up {} at {} with arm {}. ",
    # Add more predicate formats here  
    
    # barman
    "ontable": "{} is on the table. ",
    "holding": "You are holding {}. ",
    "empty": "{} is empty. ",
    "contains": "{} contains {}. ",
    "clean": "{} is clean. ",
    "used": "Pour {} from a shot glass to a used shaker {}",
    "dispenses": "{} dispenses {}. ",
    "shaker-empty-level": "{} is at empty level {}. ",
    "shaker-level": "{} is at level {}. ",
    "next": "level {} is next to level {}. ",
    "unshaked": "{} is unshaked. ",
    "shaked": "{} is shaked. ",
    "cocktail-part1": "{} part1 ingredient is {}. ",
    "cocktail-part2": "{} part2 ingredient is {}. ",
    
    "grasp": "{} grasp {}. ",
    "leave": "{} leave {}. ",
    "fill-shot": "fill-shot glass {} with {} with {} and {} holding {}. ",
    "refill-shot": "refill-shot {} with {} with {} and {} holding {}. ",
    "empty-shot": "use hand {} to empty-shot glass {} with beverage {}. ",
    "clean-shot": "clean-shot glass {} with {} with hand {} holding shot glass and {}. ",
    "pour-shot-to-clean-shaker": "pour-shot-to-clean-shaker from a shot glass {} with {} to a clean shaker {} with hand {} from level {} to level {}", #:parameters (?s - shot ?i - ingredient ?d - shaker ?h1 - hand ?l - level ?l1 - level)
    "pour-shot-to-used-shaker": "pour-shot-to-used-shaker from a shot glass {} with {} to a used shaker {} with hand {} from level {} to level {}", #:parameters (?s - shot ?i - ingredient ?d - shaker ?h1 - hand ?l - level ?l1 - level)
    "empty-shaker": "use hand {} to empty-shaker {} with ingredient {} from level {} to level {}.",
    "clean-shaker": "use hand {} and hand {} to clean-shaker {}",
    "shake": "shake a cocktail {} with ingredient {} and ingredient {} in a shaker {} with hand {} and hand {}",
    "pour-shaker-to-shot": "pour-shaker-to-shot to a shot glass {} the ingredient {} with hand {} from shaker {} from level {} to level {}", #:parameters (?s - shot ?i - ingredient ?d - shaker ?h1 - hand ?l - level ?l1 - level)
}  


description_map = {
    "blocks": '''
    The robot has four actions: pickup, putdown, stack, and unstack. The domain assumes a world where there are a set of blocks that can be stacked on top of each other, an arm that can hold one block at a time, and a table where blocks can be placed.
    The actions defined in this domain include:
    pickup <block>: allows the arm to pick up a block from the table if it is clear and the arm is empty. After the pickup action, the arm will be holding the block, and the block will no longer be on the table or clear.
    putdown <block>: allows the arm to put down a block on the table if it is holding a block. After the putdown action, the arm will be empty, and the block will be on the table and clear.
    stack <block> <block>: allows the arm to stack a block on top of another block if the arm is holding the top block and the bottom block is clear. After the stack action, the arm will be empty, the top block will be on top of the bottom block, and the bottom block will no longer be clear.
    unstack <block> <block>: allows the arm to unstack a block from on top of another block if the arm is empty and the top block is clear. After the unstack action, the arm will be holding the top block, the top block will no longer be on top of the bottom block, and the bottom block will be clear.
    ''',
    "barman": '''
    You are a robot barman that manipulates drink dispensers, shot glasses and a shaker. You have two hands. The goal is to find a plan that serves a desired set of drinks. Here are the actions you can do

    <hand> grasp <container>: Grasp a container
    <hand> leave <container>: Leave a container on the table
    fill-shot <shot> <ingredient> <hand1> <hand2> <dispenser>: Fill a shot glass with an ingredient from dispenser
    refill-shot <shot> <ingredient> <hand1> <hand2> <dispenser>: Refill a shot glass with an ingredient from dispenser
    empty-shot <hand> <shot> <beverage>: Empty a shot glass
    clean-shot <shot> <beverage> <hand1> <hand2>: Clean a shot glass
    pour-shot-to-clean-shaker <shot> <ingredient> <shaker> <hand1> <level1> <level2>: Pour an ingredient from a shot glass to a clean shaker from level1 to level2
    pour-shot-to-used-shaker <shot> <ingredient> <shaker> <hand1> <level1> <level2>: Pour an ingredient from a shot glass to a used shaker from level1 to level2
    empty-shaker <hand> <shaker> <cocktail> <level1> <level2>: Empty a shaker containing cocktail from level1 to level2
    clean-shaker <hand1> <hand2> <shaker>: Clean a shaker
    shake <cocktail> <ingredient1> <ingredient2> <shaker> <hand1> <hand2>: Shake a cocktail in a shaker
    pour-shaker-to-shot <beverage> <shot> <hand> <shaker> <level1> <level2>: Pour a beverage from a shaker to a shot glass from level1 to level2

    You have the following restrictions on your actions:
    You can only grasp a container if your hand is empty and it is on the table.
    You can only leave a container if you are holding it.
    You can only fill a shot glass if you are holding the shot glass, your other hand is empty, the shot glass is empty and clean.
    You can only refill a shot glass if you are holding the shot glass, your other hand is empty, the shot glass is empty and has contained the saree ingredient before.
    You can only empty a shot glass if you are holding the shot glass and it contains a beverage.
    You can only pour from a shot glass to a clean shaker if you are holding the shot glass, the shot glass contains an ingredient, and the shaker is empty and clean.
    You can only pour from a shot glass to a used shaker if you are holding the shot glass, the shot glass contains an ingredient, the shaker is unshaked and at a level not full.
    You can only empty a shaker if you are holding the shaker and the shaker contains a shaked beverage.
    You can only clean a shaker if you are holding the shaker, your other hand is empty, and the shaker is empty.
    You can only shake a cocktail if you are holding the shaker, your other hand is empty, the shaker is unshaked, and the shaker contains two ingredients, and both ingredients are parts of a cocktail.
    You can only pour from a shaker to a shot glass if you are holding the shaker, the shaker contains the cocktail, the shaker is shaked, and the shot glass is empty and clean.

    Once you grasp a container, you are holding the container and the container is not on the table.
    Once you leave a container on the table, your hand become empty.
    Once you pour an ingredient from a shot glass to a shaker, the shaker contains the ingredient and is at one level above the previous level, and the shot glass becomes empty.
    Once you empty a shaker, the shaker is at the empty level.
    Once you shake, the two ingredients in the shaker become a cocktail.
    Once you pour from a shaker to a shot glass, the shot glass contains the beverage in the shaker, the shot glass is no longer clean and empty, and the shaker is at one level below the previous level.
    ''',
    "gripper": '''
    You are a robot with a gripper that can move objects between different rooms.
    There are three actions defined in this domain:
    move <room1> <room2>: This action allows the robot to move from one room to another.The action has a single precondition, which is that the robot is currently in a room. The effect of this action is to move the robot to another room and to remove the fact that it is in the original room.
    pick <obj> <room> <gripper>: This action allows the robot to pick up an object using the gripper. The action has three preconditions: (1) the object is located in a room (2) the robot is currently in the same room and (3) the gripper is free (i.e., not holding any object). The effect of this action is to update the state of the world to show that the robot is carrying the object using the gripper, the object is no longer in the room, and the gripper is no longer free.
    drop <obj> <room> <gripper>: This action allows the robot to drop an object that it is carrying. The action has two preconditions: (1) the robot is currently carrying the object using the gripper, and (2) the robot is currently in a room. The effect of this action is to update the state of the world to show that the robot is no longer carrying the object using the gripper, the object is now located in the room, and the gripper is now free.
    ''',
    "tyreworld": '''
    Your goal is to replace flat tyres with intact tyres on the hubs. Intact tyres should be inflated. The nuts should be tight on the hubs. The flat tyres, wrench, jack, and pump should be in the boot. The boot should be closed.
    There are 13 actions defined in this domain:
    open <container>: The precondition for this action is that the container is unlocked and closed. The effect of this action is that the container is open and not closed.
    close <container>: The precondition for this action is that the container is open. The effect of this action is that the container is closed and not open.
    fetch <object> <container>: The precondition for this action is that the object is inside the container and the container is open. The effect of this action is that the object is held by the agent and not inside the container.
    put-away <object> <container>: The precondition for this action is that the object is held by the agent and the container is open. The effect of this action is that the object is inside the container and not held by the agent.
    loosen <nut> <hub>: The precondition for this action is that the agent has a wrench, the nut on hub is tight, and the hub is on the ground. The effect of this action is that the nut on hub is loose and not tight.
    tighten <nut> <hub>: The precondition for this action is that the agent has a wrench, the nut on hub is loose, and the hub is on the ground. The effect of this action is that the nut on hub is tight and not loose.
    jack-up <hub>: This action represents the process of lifting a hub off the ground using a jack. It requires the agent to have a jack and for the hub to be on the ground. After performing this action, the hub will no longer be on the ground and the agent will no longer have the jack.
    jack-down <hub>: This action represents the process of lowering a hub back to the ground from an elevated position using a jack. It requires the agent to have the hub off the ground. After performing this action, the hub will be back on the ground and the agent will have the jack.
    undo <nut> <hub>: This action undo the fastening of a nut on a hub. The preconditions are the hub is not on the ground (i.e., it has been jacked up), the hub is fastened, the agent has a wrench and the nut is loose. The effects are the agent has the nut, the hub is unfastened, the hub is no longer loose and the hub is not fastened anymore.
    do-up <nut> <hub>: This action fasten a nut on a hub. The preconditions are the agent has a wrench, the hub is unfastened, the hub is not on the ground (i.e., it has been jacked up) and the agent has the nut to be fastened. The effects are the nut is now loose on the hub, the hub is fastened, the hub is no longer unfastened and the agent no longer has the nut.
    remove-wheel <wheel> <hub>: This action removes a wheel from a hub. It can only be performed if the hub is not on the ground, the wheel is currently on the hub, and the hub is unfastened. After the action is performed, the agent will have the removed wheel and the hub will be free, meaning that the wheel is no longer on the hub.
    put-on-wheel <wheel> <hub>: This action puts a wheel onto a hub. It can only be performed if the agent has the wheel, the hub is free, the hub is unfastened, and the hub is not on the ground. After the action is performed, the wheel will be on the hub, the hub will no longer be free, and the agent will no longer have the wheel.
    inflate <wheel>: This action inflates a wheel using a pump. It can only be performed if the agent has a pump, the wheel is not inflated, and the wheel is intact. After the action is performed, the wheel will be inflated.
    ''',
}