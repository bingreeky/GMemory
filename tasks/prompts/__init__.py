from .alfworld_prompt import alfworld_solver_system_prompt, alfworld_few_shots
from .fever_prompt import fever_solver_system_prompt, fever_few_shots
from .pddl_prompt import pddl_prompts

def get_dataset_system_prompt(task: str, task_config: dict) -> str:
    prompt_map: dict = {
        'alfworld': alfworld_solver_system_prompt,
        'fever': fever_solver_system_prompt,
        'pddl': pddl_prompts
    }

    if prompt_map.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')
    
    if task != 'pddl':
        return prompt_map.get(task)
    else:
        task_type: str = task_config.get('game_name')
        return pddl_prompts[task_type]['instruction']
        


def get_task_few_shots(dataset: str, task_config: dict, few_shots_num: int) -> list[str]:
    
    if dataset == 'alfworld':
        task_type = task_config.get('task_type')
        if task_type is None:
            raise ValueError('The task config must have the `task_type` attribute.')
        return [alfworld_few_shots[f'react_{task_type}_2'], alfworld_few_shots[f'react_{task_type}_0']][:few_shots_num]
    
    elif dataset == 'fever':
        return fever_few_shots[:few_shots_num]
    
    elif dataset == 'pddl':
        task_type = task_config.get('game_name')
        if task_type is None:
            raise ValueError('The task config must have the `game_name` attribute.')
        return pddl_prompts[task_type]['examples'][:few_shots_num]
    
    else:
        raise ValueError(f'Unsupported dataset type: {dataset}')

