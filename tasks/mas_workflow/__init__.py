from mas.mas import MetaMAS
from .autogen import AutoGen
from .macnet import MacNet
from .dylan import DyLAN
from .autogen.autogen_pddl import AutoGen as AutoGenPDDL

MAS = {
    'autogen': AutoGen,
    'macnet': MacNet,
    'dylan': DyLAN,
    'autogen_pddl': AutoGenPDDL
}

def get_mas(mas_type: str) -> MetaMAS:

    if MAS.get(mas_type) is None:
        raise ValueError('Unsupported mas type.')
    return MAS.get(mas_type)() 