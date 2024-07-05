import abc

from typing import Dict, List, Optional, Tuple, Type, TypeVar

#from tqdm import tqdm

class LM(abc.ABC):

    @abc.abstractmethod
    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        pass

    @abc.abstractmethod
    def loglikelihood_rolling(self, requests) -> List[Tuple[float]]:
        pass

    @abc.abstractmethod
    def generate_until(self, requests) -> List[str]:
        pass

    