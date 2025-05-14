from mas.meta_mas import MetaMAS
from .autogen import AutoGen
from .graph_mas import GraphMAS
from .dylan import DyLAN

MAS = {
    'autogen': AutoGen,
    'graph': GraphMAS,
    'dylan': DyLAN
}

def get_mas(mas_type: str) -> MetaMAS:

    if MAS.get(mas_type) is None:
        raise ValueError('Unsupported mas type.')
    return MAS.get(mas_type)() 