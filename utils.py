
# from /lm_eval/utils.py and /lm_eval/models/utils.py

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch
import collections

import transformers

def get_rolling_token_windows(token_list, prefix_token, max_seq_len, context_len):
    """
    - context_len allows for a rolling window context, allowing each prediction window to potentially
      condition on some context

    :param token_list: list
        List of tokens to be PREDICTED
    :param max_seq_len: int
        max_seq_len of model (or max_seq_len we want to use)
    :param context_len: int
        Amount of desired token context for prediction. Needs to be at least 1.
    :param prefix_token: token
        Dummy token like <eos> so the first token has something to condition on
    :return: generator
        Generator of tuples
            (input_tokens, pred_tokens)
        Note: Score only the last len(pred_tokens) logits of the LM
    """
    assert 1 <= context_len <= max_seq_len
    if not token_list:
        return
    # +1 offset, going from input->preds
    pred_len = max_seq_len - context_len + 1
    predicted = 0

    # Special handling for first window: predict all tokens
    first_seq_len = min(max_seq_len, len(token_list))
    yield ([prefix_token] + token_list[: first_seq_len - 1], token_list[:first_seq_len])
    predicted += first_seq_len

    while predicted < len(token_list):
        window_pred_len = min(len(token_list) - predicted, pred_len)
        window_end = predicted + window_pred_len

        yield (
            token_list[window_end - max_seq_len - 1 : window_end - 1],
            token_list[window_end - window_pred_len : window_end],
        )
        predicted += window_pred_len


def make_disjoint_window(pair):
    a, b = pair
    return a[: len(a)-(len(b)-1)], b

class Collator:
    """
    A class for reordering and batching elements of an array.

    This class allows for sorting an array based on a provided sorting function, grouping elements based on a grouping function, and generating batches from the sorted and grouped data.

    Objects of this class have the group_by attribute which determines the method for grouping
    the data while batching it. Three options include "gen_kwargs", "contexts", or None:
        If group_by == "gen_kwargs" then requests will be grouped by gen_kwargs
        If group_by == "contexts" then requests will be grouped by context + cont[:-1]
        If None then requests will just be reordered by length descending.
    """

    def __init__(
        self,
        arr: List,
        sort_fn: Callable = lambda x: x,
        group_fn: Callable = lambda x: x[1],
        group_by: Union[Literal["gen_kwargs", "contexts"], None] = None,
    ) -> None:
        self._group_by = group_by
        # 0 indices are enumerated indices. Apply functions to original arr.
        self._sort_fn = lambda x: sort_fn(x[1])
        self._group_fn = lambda x: group_fn(x[1])
        self._reorder_indices: List = []
        self._size = len(arr)
        self._arr_with_indices: Union[Dict, Tuple[Tuple[int, Any], ...]] = tuple(
            enumerate(arr)
        )  # [indices, (arr)]
        if self._group_by == "contexts":
            self._group_by_context()
        elif self._group_by == "gen_kwargs":
            self._group_by_index()

    def _group_by_index(self) -> None:
        """Group the elements of a list based on their indices."""
        self._arr_with_indices = self.group(
            self._arr_with_indices, fn=self._group_fn, group_by="gen_kwargs"
        )

    def _group_by_context(self) -> None:
        """Group the array with indices by context."""
        self._arr_with_indices = self.group(
            self._arr_with_indices, fn=self._group_fn, group_by="contexts"
        )

    def get_batched(self, n: int = 1, batch_fn: Optional[Callable] = None) -> Iterator:
        """
        Generates and yields batches from the reordered array. The method of grouping and batching
        depends on the parameter `group_by`.
        If `group_by` is set to "gen_kwargs", it will batch the
        re-ordered values with same gen_kwargs for each batch.
        If `group_by` is "contexts", it caches the requests by context before batching.
        If `group_by` is neither "gen_kwargs" nor "contexts", it yields the reordered array

        Parameters:
        - n (int): The size of each batch. Defaults to 1.
        - batch_fn ([Callable[[int, Iterable], int]] | None): A function to determine the size of
          each batch. Optional, defaults to None.

        Returns:
        Iterator: An iterator over batches of reordered elements grouped as per the `group_by`
                  attribute.

        Yields:
        List of batched elements according to the `group_by` attribute.
        """
        if self._group_by == "gen_kwargs":
            for (
                key,
                values,
            ) in self._arr_with_indices.items():  # type: ignore
                values = self._reorder(values)
                batch = self.get_chunks(values, n=n, fn=batch_fn)
                yield from batch
        elif self._group_by == "contexts":
            # Get one sample from each key
            values = self._reorder(
                [value[0] for value in self._arr_with_indices.values()]
            )
            batch = self.get_chunks(values, n=n, fn=batch_fn)
            yield from batch
        else:
            values = self._reorder(self._arr_with_indices)  # type: ignore
            batch = self.get_chunks(values, n=n, fn=batch_fn)
            yield from batch

    def get_cache(
        self,
        req_str: Tuple[str, str] = None,
        cxt_toks: List[int] = None,
        cont_toks: List[int] = None,
        logits: torch.Tensor = None,
    ) -> Iterator[Tuple[Tuple[str, str], List[int], torch.Tensor]]:
        """
        Retrieves cached single-token continuations and their associated arguments, updating indices as necessary.

        The behavior of this function varies depending on how the `group_by` attribute is set:

        - When `group_by` is "contexts":
            The function identifies single-token continuations by checking for keys that equate to
            [context+continuation][-1] and logs the indices for re-ordering.
            In this mode, this function can work in two scenarios:

            1. Cache Hit - Single Match:
                If a single matching context-continuation pair is found in the cache,
                the function yields the original arguments.

            2. Cache Hit - Multiple Matches:
                If multiple matching context-continuation pairs are found in the cache,
                the function expands the logits batch dimension to match the number of cache hits.
                It updates the original requests and continuation tokens.

        - When `group_by` is not set to "contexts":
            This method yields the original arguments, logits and continuation tokens,
            without checking for one-token continuations.

        Parameters:
        - req_str (tuple[str, str]): Original strings used for CachingLM.
        - cxt_toks (list[int]): Full context tokens used for lookup.
        - cont_toks (list[int]): Continuation tokens for which logits were generated.
        - logits (torch.Tensor [1, seq_length, vocab_size]): Logits generated by the model given context and continuation keys.

        Yields:
        - Iterator:
            - req_str (tuple[str, str]): strings used for CachingLM.
            - cont_toks (list[int]) : continuation tokens.
            - logits (torch.Tensor [1, seq_length, vocab_size]): The original logits (repeated cache hit times)
        """
        if self._group_by == "contexts":
            cache_hit: List[
                Tuple[int, Tuple[Tuple[str, str], List[int], List[int]]]
            ] = self._arr_with_indices.pop(tuple(cxt_toks + cont_toks[:-1]))
            if (cache_size := len(cache_hit)) == 1:
                self._reorder_indices.extend(x[0] for x in cache_hit)
                yield req_str, cont_toks, logits
            else:
                # If we have matching requests then expand the batch dimension (no-op) and
                # yield each along with its corresponding args.
                multilogits = logits.expand(cache_size, -1, -1).chunk(cache_size)
                indices, req_str, cont_toks = zip(
                    *[(x[0], x[1][0], x[-1][-1]) for x in cache_hit]
                )
                self._reorder_indices.extend(indices)
                for c_key, cont_tok, logit in zip(req_str, cont_toks, multilogits):
                    yield c_key, cont_tok, logit
        else:
            yield req_str, cont_toks, logits

    def _reorder(self, arr: Union[List, Tuple[Tuple[int, Any], ...]]) -> Iterator:
        """
        Reorders the elements in the array based on the sorting function.

        Parameters:
        - arr (list | tuple[tuple[int, Any], ...]]): The array or iterable to be reordered.

        Yields:
            Iterator
        """
        arr = sorted(arr, key=self._sort_fn)
        if not self._group_by == "contexts":
            # If grouped by contexts then indices will be set in get_cache()
            self._reorder_indices.extend([x[0] for x in arr])
        yield from [x[1] for x in arr]

    def get_original(self, newarr: List) -> List:
        """
        Restores the original order of elements from the reordered list.

        Parameters:
        - newarr (list): The reordered array.

        Returns:
        list: The array with elements restored to their original order.
        """
        res = [None] * self._size
        cov = [False] * self._size

        for ind, v in zip(self._reorder_indices, newarr):
            res[ind] = v
            cov[ind] = True

        assert all(cov)

        return res

    def __len__(self):
        return self._size

    @staticmethod
    def group(
        arr: Iterable,
        fn: Callable,
        group_by: Literal["gen_kwargs", "contexts"] = "gen_kwargs",
    ) -> dict:
        """
        Groups elements of an iterable based on a provided function.


        The `group_by` parameter determines the method of grouping.
        If `group_by` is "contexts", the elements are grouped by [context + cont][:-1].
        If `group_by` is "gen_kwargs", the elements are grouped based on the gen_kwargs dict.

        Parameters:
        - arr (Iterable): The iterable to be grouped.
        - fn (Callable): The function to determine the grouping.
        - values (bool): If True, returns the values of the group. Defaults to False.

        Returns:
        Iterator: An iterable of grouped elements.
        """
        res = collections.defaultdict(list)
        for ob in arr:
            # where ob == [context + cont]
            if group_by == "contexts":
                res[tuple(fn(ob))].append(ob)
            else:
                try:
                    hashable_dict = tuple(
                        (
                            key,
                            tuple(value)
                            if isinstance(value, collections.abc.Iterable)
                            else value,
                        )
                        for key, value in sorted(fn(ob).items())
                    )
                    res[hashable_dict].append(ob)
                except (TypeError, AttributeError):
                    res[tuple(fn(ob))].append(ob)
        return res

    @staticmethod
    def get_chunks(_iter, n: int = 0, fn=None):
        """
        Divides an iterable into chunks of specified size or based on a given function.
        Useful for batching

        Parameters:
        - iter: The input iterable to be divided into chunks.
        - n: An integer representing the size of each chunk. Default is 0.
        - fn: A function that takes the current index and the iterable as arguments and returns the size of the chunk. Default is None.

        Returns:
        An iterator that yields chunks of the input iterable.

        Example usage:
        ```
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for chunk in chunks(data, 3):
            print(chunk)
        ```
        Output:
        ```
        [1, 2, 3]
        [4, 5, 6]
        [7, 8, 9]
        [10]
        ```
        """
        arr = []
        _iter = tuple(_iter)
        for i, x in enumerate(_iter):
            arr.append(x)
            if len(arr) == (fn(i, _iter) if fn else n):
                yield arr
                arr = []

        if arr:
            yield arr

class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        # print(sequence, self.sequence_ids)
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker

def stop_sequences_criteria(
        tokenizer: transformers.PreTrainedTokenizer,
        stop_sequences: List[str],
        initial_decoder_input_length: int,
        batch_size: int
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )

def pad_and_concat(
    max_length: int,
    tensors: List[torch.Tensor],
    padding_side: Literal["right", "left"] = "right",
):
    """
    Method for padding a list of tensors given the maximum tensor
    length in the batch. Used for batching inputs and continuations in
    seq2seq models.
    """
    assert (
        padding_side == "left" or padding_side == "right"
    ), f"Unrecognized padding type: '{padding_side}' not 'left' or 'right'"

    for i, tensor in enumerate(tensors):
        if len(tensor.shape) == 2:
            tensor = tensor.squeeze(0)  # squeeze, in case passed [1, seq] size
        tensor_len = tensor.shape[0]
        if tensor_len < max_length:
            if padding_side == "right":
                # right-pad
                tensors[i] = torch.cat(
                    [
                        tensor,  # [seq]
                        torch.zeros(
                            max_length - tensor_len,
                            dtype=torch.long,
                            device=tensor.device,
                        ),  # [padding_length - seq]
                    ],
                    dim=0,
                ).unsqueeze(0)
            else:
                # left-pad
                tensors[i] = torch.cat(
                    [
                        torch.zeros(
                            max_length - tensor_len,
                            dtype=torch.long,
                            device=tensor.device,
                        ),  # [padding_length - seq]
                        tensor,  # [seq]
                    ],
                    dim=0,
                ).unsqueeze(0)
        else:
            tensors[i] = tensor.unsqueeze(0)

    return torch.cat(tensors, dim=0)

