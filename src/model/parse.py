from copy import deepcopy

from .spec import ModelSpec
from .vm import VelocityModule
from .cvm import CoupledVMArch
from .straightpcf import StraightPCF

def get_model(model_config, **kwargs) -> ModelSpec:
    MAP = {
        'VelocityModule': VelocityModule,
        'CoupledVMArch': CoupledVMArch,
        'StraightPCF': StraightPCF,
    }
    model_config = deepcopy(model_config)
    __target__ = model_config['__target__']
    del model_config['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    return MAP[__target__](model_config=model_config, **kwargs)
