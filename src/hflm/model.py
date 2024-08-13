import abc

from typing import Dict, List, Optional, Tuple, Type, TypeVar

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

    