import json

from .base_env import BaseEnv, BaseRecorder
from .alfworld_env import AlfworldEnv, AlfworldRecorder, get_env_name_from_gamefile, prefixes
from .fever_env import FeverEnv, FeverRecorder
from .pddl_env.pddl_env import PDDLEnv, PDDLRecorder, get_all_environment_configs

TASKS_PATH = {
    'alfworld': 'data/alfworld/alfworld_tasks_suffix.json',
    'fever': 'data/fever/fever_dev.jsonl',
    'pddl': 'data/pddl/test.jsonl'  
}

## Tasks
alfworld_tasks: list[dict] = [
    {
        'task': f'{row["goal"]}',
        'env_kwargs': {
            'config': 'alfworld',
            "gamefile": row["gamefile"],
        },
        'task_type': prefixes[get_env_name_from_gamefile(row["gamefile"])],
        'env_name': get_env_name_from_gamefile(row["gamefile"])
    } for row in json.load(open(TASKS_PATH['alfworld'], "r")) 
]

with open(TASKS_PATH['fever'], 'r') as f:
    fever_tasks = [
        {
            'task': row['claim'],
            'answer': row['label'],
            'env_name': 'fever',
        }
        for row in (json.loads(line) for line in f) 
    ][:100]


TASK_NAMES = ["barman", "blockworld", "gripper", "tyreworld"]
pddl_tasks: list[dict] = get_all_environment_configs(TASK_NAMES, TASKS_PATH['pddl'])


TASK_DATA = {
    'alfworld': alfworld_tasks,
    'fever': fever_tasks,
    'pddl': pddl_tasks
}

ENVS = {
    'alfworld': AlfworldEnv,
    'fever': FeverEnv,
    'pddl': PDDLEnv
}

RECORDERS = {
    'alfworld': AlfworldRecorder,
    'fever': FeverRecorder,
    'pddl': PDDLRecorder
}


def get_env(task: str, env_config: dict, max_trials: int) -> BaseEnv:
    
    if ENVS.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')
    
    return ENVS.get(task)(env_config, max_trials)

def get_recorder(task: str, working_dir: str, namespace: str) -> BaseRecorder:
    
    if RECORDERS.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')
    
    return RECORDERS.get(task)(working_dir=working_dir, namespace=namespace)

def get_task(task: str) -> list[dict]:

    if TASK_DATA.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')
    
    return TASK_DATA.get(task)