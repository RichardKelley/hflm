from typing import List, Iterator, Tuple
import torch

def get_batches(in_arr: List, n: int = 1):
    arr = []
    for i, x in enumerate(in_arr):
        arr.append(x)
        if len(arr) == n:
            yield arr
            arr = []

    if arr:
        yield arr

class IdentityCollator:
    def __init__(self, arr: List):
        self._arr_with_indices = tuple(enumerate(arr))

    def get_batched(self, n: int = 1, batch_fn = None) -> Iterator:
        arr = []
        for i, x in self._arr_with_indices:
            arr.append(x)
            if len(arr) == n:
                yield arr
                arr = []

        if arr:
            yield arr

    def get_cache(self, 
        req_str, 
        cxt_toks, 
        cont_toks, 
        logits
    ) -> Iterator[Tuple[Tuple[str, str], List[int], torch.Tensor]]:
        yield req_str, cont_toks, logits
