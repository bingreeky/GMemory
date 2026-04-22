def get_model_type(model_name: str) -> str:
    
    valid_model_types: list[str] = [
        'gpt-4o-mini', 
        'qwen2.5-7b', 
        'qwen2.5-14b',
        'qwen2.5-32b', 
        'qwen2.5-72b',
        'intern', 
        'deepseek-v3',
        'llama3.2:1b',
        'mistral:7b',
        'llama3.2:3b',
        'qwen3:14b',
    ]

    for model_type in valid_model_types:
        if model_type in model_name.lower():
            return model_type
    
    return 'unknown'
