from .utils import simple_parse_args_string

import abc
from typing import List, Optional, Tuple, Type, TypeVar

T = TypeVar("T", bound="LM")

class LM(abc.ABC):

    def __init__(self) -> None:
        self._rank = 0
        self._world_size = 1

    @abc.abstractmethod
    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        pass

    @abc.abstractmethod
    def loglikelihood_rolling(self, requests) -> List[Tuple[float]]:
        pass

    @abc.abstractmethod
    def generate_until(self, requests) -> List[str]:
        pass

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size
    
    @classmethod
    def create_from_arg_string(
        cls: Type[T], arg_string: str, additional_config: Optional[dict] = None
    ) -> T:
        additional_config = {} if additional_config is None else additional_config
        args = simple_parse_args_string(arg_string)
        args2 = {k : v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)
    
    @classmethod
    def create_from_arg_obj(
        cls: Type[T], arg_dict: dict, additional_config: Optional[dict] = None
    ) -> T:
        additional_config = {} if additional_config is None else additional_config
        additional_config = {
            k: v for k, v in additional_config.items() if v is not None
        }
        return cls(**arg_dict, **additional_config)    
 
    def get_model_info(self) -> dict:
        def get_model_num_params(model) -> int:
            if hasattr(model, "num_parameters"):
                return model.num_parameters()
            if hasattr(model, "parameters"):
                return sum(p.numel() for p in model.parameters())
            else:
                return -1

        def get_model_dtype(model) -> str:
            if hasattr(model, "dtype"):
                return model.dtype
            else:
                return ""

        model_info = {
            "model_num_parameters": get_model_num_params(self._model),
            "model_dtype": get_model_dtype(self._model),
        }
        return model_info        

    