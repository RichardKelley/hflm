from .model import LM

import torch
import torch.nn.functional as F

import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer

from accelerate import (
    Accelerator,
    DistributedType,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)
from accelerate.utils import get_max_memory

import copy

from .utils import (
    get_rolling_token_windows,
    make_disjoint_window,
    configure_pad_token,
    pad_and_concat,
    stop_sequences_criteria,
    get_batches
)

from tqdm import tqdm

import os
import logging
from packaging import version
from datetime import timedelta
from typing import List, Tuple, Union, Optional

import gc

# from lm_eval/models/utils.py
def get_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype, str) and dtype != "auto":
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype

class HFLM(LM):

    AUTO_MODEL_CLASS = None # this is set in _get_backend
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(self, 
        model: Union[str, PreTrainedModel],
        tokenizer: Optional[
            Union[
                str, PreTrainedTokenizer
            ]
        ] = None,
        truncation : Optional[bool] = False,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto", # NB: MLX incompatibility
        add_bos_token: Optional[bool] = False, # Need for Gemma-2
        max_length: Optional[int] = None,
        prefix_token_id: Optional[int] = None,
        batch_size: Optional[Union[int,str]] = 1,
        max_batch_size: Optional[int] = 1024,
        parallelize: Optional[bool] = False,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[Union[str, os.PathLike]] = "./offload",
        **kwargs,                 
    ) -> None:
        super().__init__()

        if not isinstance(model, str):
            self._model = model
            self._device = model.device
            gpus = 0

            if tokenizer is not None:
                self.tokenizer = tokenizer
            else:
                model_name = model.name_or_path
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name
                )
        else:
            assert isinstance(device, str)
            assert isinstance(model, str)
            assert isinstance(batch_size, (int, str))

            gpus = torch.cuda.device_count()
            accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
            accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
            if accelerator.num_processes > 1:
                self.accelerator = accelerator

            if "npu" in accelerator.device.type:
                gpus = torch.npu.device_count()

            if not (parallelize or accelerator.num_processes > 1):
                device_list = set(
                    ["cuda", "cpu"]
                    + [f"cuda:{i}" for i in range(gpus)]
                    + ["mps", "mps:0"]
                    + [f"npu:{i}" for i in range(gpus)]
                )
                if device and device in device_list:
                    self._device = torch.device(device)
                    logging.info(f"Using device '{device}'")
                    if device in ("mps", "mps:0") and version.parse(
                        torch.__version__
                    ) < version.parse("2.1"):
                        raise RuntimeError(
                            f"mps requires torch >= 2.1. You have {torch.__version__}"
                        )
                else:
                    logging.info("Device not specified")
                    logging.info(f"Cuda Available? {torch.cuda.is_available()}")
                    self._device = (
                        torch.device("cuda")
                        if torch.cuda.is_available()
                        else torch.device("cpu")
                    )
            else:
                if device != "cuda":
                    logging.info(
                        f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                    )
                # TODO: include in warning that `load_in_8bit` etc. affect this too
                self._device = torch.device(device)
            
        self.truncation = truncation
        self._device = device
        self._max_length = max_length
        self.add_bos_token = add_bos_token
        self.custom_prefix_token_id = prefix_token_id
        
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            #self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)

        # get backend
        self._get_backend()

        # create tokenizer
        self._create_tokenizer(
            model,
            tokenizer
        )

        # create model
        if isinstance(model, str):
            self._create_model(model, dtype=dtype, device=device, parallelize=parallelize, **kwargs)

        self.tokenizer = configure_pad_token(self.tokenizer)


        if isinstance(model, str):
            if gpus >= 1 or str(self.device) == "mps":
                # TODO: can remove this whole snippet except in the mps case, perhaps?
                if not (parallelize or hasattr(self, "accelerator")):
                    # place model onto device requested manually,
                    # if not using HF Accelerate or device_map
                    # or any other option that preloads model onto device
                    try:
                        self._model.to(self.device)
                    except ValueError:
                        logging.debug(
                            "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                        )
            # multigpu data-parallel support when launched with accelerate
            if gpus > 1:
                if accelerator.num_processes > 1:
                    if parallelize:
                        logging.warning(
                            "You are both using a HF Accelerate `device_map` (`--model_args parallelize=True`) and launching via `accelerate launch`. This will attempt to do model and data parallelism depending on the resources available."
                        )
                    elif gpus > accelerator.num_processes:
                        logging.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {accelerator.num_processes} devices."
                        )
                        if self.accelerator.is_local_main_process:
                            logging.info(
                                f"Using {gpus} devices with data parallelism"
                            )

                    self._device = torch.device(f"{accelerator.device}")
                    self.accelerator = accelerator

                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
                else:
                    # if we aren't launching via accelerate, ditch
                    self._rank = 0
                    self._world_size = 1
        else:
            # if a PreTrainedModel was passed into HFLM, we forgo distributed setup.
            logging.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1


    def _get_accelerate_args(self,
        parallelize: bool = None,
        device_map: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        gpus: Optional[int] = None,
    ) -> dict:
        """Returns the kwargs needed to apply `accelerate` in `AutoModel.from_pretrained`."""
        num_local_processes = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        num_machines = int(os.environ.get("WORLD_SIZE", 0)) // num_local_processes
        if (
            num_machines == 0
            and hasattr(self, "accelerator")
            and self.accelerator is not None
        ):
            logging.info(
                "We are not in a distributed setting for accelerate. Setting model_parallel to False."
            )
            parallelize = False

        if parallelize is None:
            # If parallelism is unset by the user, we automatically assign model parallelism
            # if enough extra GPUs are available
            max_memory_all_gpus = get_max_memory()
            # We just want gpu, not cpu, max memory
            if "cpu" in max_memory_all_gpus:
                del max_memory_all_gpus["cpu"]
            parallelize = bool(num_local_processes < len(max_memory_all_gpus))
            logging.info(
                f"Setting model parallel to {parallelize} since "
                f"the number of local processes is {num_local_processes} "
                f"and the number of GPUs is {len(max_memory_all_gpus)}"
            )

        args = {}
        if parallelize:  # Model parallelism will be used
            max_memory = {}
            if max_memory_per_gpu is not None:  # Using the provided memory requirements
                max_memory_per_gpu_map = {
                    device_idx: max_memory_per_gpu for device_idx in range(gpus)
                }
            else:  # Estimating the possible memory requirements
                max_memory_all_gpus = get_max_memory()
                if "cpu" in max_memory_all_gpus:
                    del max_memory_all_gpus["cpu"]
                if not hasattr(self, "accelerator"):
                    max_memory_per_gpu_map = {
                        k: v for k, v in max_memory_all_gpus.items()
                    }
                else:
                    # use only 1 / num_processes of the GPUs if we are running under accelerate launch
                    max_memory_per_gpu_map = {
                        k: v
                        for k, v in max_memory_all_gpus.items()
                        if k % num_local_processes
                        == (self.accelerator.process_index % num_local_processes)
                    }
            args["max_memory"] = max_memory_per_gpu_map
            args["device_map"] = "auto"
            logging.info(
                f"Model parallel was set to True, setting max memory per GPU to {max_memory_per_gpu_map} and device map to 'auto'"
            )

            if max_cpu_memory is not None:
                max_memory["cpu"] = max_cpu_memory

            args["offload_folder"] = offload_folder
        elif (
            device_map is None
        ):  # No model parallelism, we use the default provided device for our model
            if hasattr(self, "accelerator"):
                device_map = {"": f"{self.accelerator.device}"}
            else:
                device_map = {"": str(self.device)}
            args["max_memory"] = None
            args["device_map"] = device_map
            logging.info(
                f"Model parallel was set to False, max memory was not set, and device map was set to {device_map}"
            )
        else:
            args["max_memory"] = None
            args["device_map"] = None
            logging.info("Model parallel was set to False.")

        return args

    def _get_backend(
        self,
        backend = "default",
        trust_remote_code = False
    ) -> None:
        assert backend in ["default", "causal", "seq2seq"]

        if backend != "default":
            if backend == "causal":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
            elif backend == "seq2seq":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM
        else:
            self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM # TEMPORARY

        assert self.AUTO_MODEL_CLASS in [
            transformers.AutoModelForCausalLM,
            transformers.AutoModelForSeq2SeqLM,
        ]
        return None

    def _create_tokenizer(
            self,
            model : Union[str, PreTrainedModel],
            tokenizer: Optional[
                Union[
                    str,
                    PreTrainedTokenizer
                ]
            ]
    ) -> None:
        if tokenizer is not None:
            if isinstance(tokenizer, str):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer
                )
            else:
                assert isinstance(tokenizer, PreTrainedTokenizer)
                self.tokenizer = tokenizer
        else:
            if isinstance(model, str):
                model_name = model
            else:
                model_name = model.name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return None

    def _create_model(
        self,
        model: str,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = False,
        parallelize: Optional[bool] = False,
        gpus: Optional[int] = None,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        device: str = "auto",
        **kwargs
    ) -> None:
        
        model_kwargs = kwargs if kwargs else {}

        model_kwargs.update(
            self._get_accelerate_args(
                parallelize,
                kwargs.get("device_map", None),
                max_memory_per_gpu,
                max_cpu_memory,
                offload_folder,
                gpus
            )
        )
        if "device_map" not in model_kwargs:
            if hasattr(self, "accelerator"):
                model_kwargs.update({"device_map": {"": f"{self.accelerator.device}"}})
            else:
                model_kwargs.update({"device_map": {"": str(self.device)}})

        if model_kwargs.get("load_in_4bit", None):
            assert (
                transformers.__version__ >= "4.30.0"
            ), "load_in_4bit requires transformers >= 4.30.0"
        if transformers.__version__ >= "4.30.0":
            if model_kwargs.get("load_in_4bit", None):
                if model_kwargs.get("bnb_4bit_compute_dtype", None):
                    model_kwargs["bnb_4bit_compute_dtype"] = get_dtype(
                        model_kwargs["bnb_4bit_compute_dtype"]
                    )

        self._model = self.AUTO_MODEL_CLASS.from_pretrained(
            model,
            torch_dtype=get_dtype(dtype),
            trust_remote_code=trust_remote_code,
            **model_kwargs
        )

        return None

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model
    
    @property
    def max_length(self):
        if self._max_length is not None:
            return self._max_length
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH
    
    @property
    def prefix_token_id(self):
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id
    
    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device
    
    @property
    def world_size(self):
        return self._world_size
    
    def _detect_batch_size(self, requests=None, pos: int = 0):
        if requests:
            context_enc, continuation_enc = requests[pos]
            max_length = len(
                (context_enc + continuation_enc)[-(self.max_length + 1) : ][:-1]
            )
            max_context_enc = len(context_enc[-(self.max_length + 1) :])
            max_cont_enc = len(continuation_enc[-(self.max_length + 1 ) :])
        else:
            max_length = self.max_length
            max_context_enc = max_length
            max_cont_enc = max_length

        # if OOM, halve batch size and retry.
        @find_executable_batch_size(starting_batch_size=self.max_batch_size)
        def forward_batch(batch_size):
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                length = max(max_context_enc, max_cont_enc)
                batched_conts = torch.ones(
                    (batch_size, length), device=self.device
                ).long()
                test_batch = torch.ones((batch_size, length), device=self.device).long()
                call_kwargs = {
                    "attn_mask": test_batch,
                    "labels": batched_conts,
                }
            else:
                call_kwargs = {}
                test_batch = torch.ones(
                    (batch_size, max_length), device=self.device
                ).long()
            for _ in range(5):
                out = F.log_softmax(self._model_call(test_batch, **call_kwargs), dim=-1)  # noqa: F841

            return batch_size

        try:
            batch_size = forward_batch()
        except RuntimeError as e:
            if "No executable batch size found" in str(e):
                batch_size = 1
            else:
                raise

        if self.world_size > 1:
            # if multi-GPU, always take minimum over all selected batch sizes
            max_rnk_bs = torch.tensor([batch_size], device=self.device)
            gathered = (
                self.accelerator.gather(max_rnk_bs).cpu().detach().numpy().tolist()
            )
            batch_size = min(gathered)
            gc.collect()
            torch.cuda.empty_cache()
            return batch_size

        gc.collect()
        torch.cuda.empty_cache()
        return batch_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None):
        special_tokens_kwargs = {}

        if add_special_tokens is None:
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                add_special_tokens = {
                    "add_special_tokens": False or self.add_bos_token
                }
        else:
            special_tokens_kwargs = {"add_special_tokens" : add_special_tokens}

        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        if left_truncate_len is not None:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())

        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        model_class = getattr(self, "AUTO_MODEL_CLASS", None)

        if model_class == transformers.AutoModelForSeq2SeqLM:
            context_enc = self.tok_encode(context)
            continuation_enc = self.tok_encode(continuation, add_special_tokens=False)
        else:
            whole_enc = self.tok_encode(context + continuation)
            context_enc = self.tok_encode(context)

            context_enc_len = len(context_enc)
            continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests: List[Tuple[str,str]], disable_tqdm: bool = False) -> List[Tuple[float, bool]]:
        '''
        requests is a list of (context, continuation) pairs
        '''
        new_reqs = []

        adaptive_batch_size = None
        if self.batch_size == "auto":
            print("Passed batch size auto. Detecting largest batch size.")
            batch_size = self._detect_batch_size(requests=requests)
            print(f"Determined largest batch size: {batch_size}.")
            adaptive_batch_size = batch_size

        for context, continuation in requests:
            if context == '':
                context_enc, continuation_enc = (
                    [self.prefix_token_id],
                    self.tok_encode(continuation)
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs, override_bs=adaptive_batch_size)

    def _model_call(self, inps, attn_mask=None, labels=None):
        with torch.no_grad():
            assert self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
            return self.model(inps).logits
        
    def _select_cont_toks(self, logits: torch.Tensor, contlen:int = None, inplen: int = None):
        assert (contlen and inplen)
        logits = logits[inplen - contlen : inplen]

        return logits

    def _loglikelihood_tokens(self, requests, disable_tqdm : bool = False, override_bs : int = None) -> List[float]:
        res = []

        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0
        )
        
        chunks = get_batches(requests, n=batch_size)

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = None
            padding_len_cont = None

            for _, context_enc, continuation_enc in chunk:
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # we assume we're in the causal case
                inp = torch.tensor(
                    (context_enc+continuation_enc)[-(self.max_length + 1):][:-1],
                    dtype=torch.long,
                    device=self.device
                )
                (inplen,) = inp.shape

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )
                inps.append(inp)
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            call_kwargs = {}

            batched_inps = pad_and_concat(
                padding_len_inp, inps, padding_side="right"
            )

            multi_logits = F.log_softmax(
                self._model_call(batched_inps, **call_kwargs), dim=-1
            )

            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                contlen = len(cont_toks)
                ctx_len = inplen + (logits.shape[0] - padding_len_inp)
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0) 

                greedy_tokens = logits.argmax(dim=-1)

                cont_toks = torch.tensor(cont_toks, dtype=torch.long, device=self.device).unsqueeze(0)
                max_equal = (greedy_tokens == cont_toks).all()
                
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)

                answer = (float(logits.sum()), bool(max_equal))
                res.append(answer)
                pbar.update(1)

        return res
            
    def loglikelihood_rolling(self, requests : List[str], disable_tqdm : bool = False) -> List[Tuple[float]]:
        '''
        We will assume that `requests` has type List[str] for this implementation
        '''
        loglikelihoods = []

        adaptive_batch_size = None
        if self.batch_size == "auto":
            print("Passed batch size auto. Detecting largest batch size.")
            batch_size = self._detect_batch_size()
            print(f"Determined largest batch size: {batch_size}.")
            adaptive_batch_size = batch_size

        for req in tqdm(requests, disable=disable_tqdm):
            string = req
            rolling_token_windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
                disable_tqdm=True,
                override_bs=adaptive_batch_size
            )

            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods
        
    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id
    
    @property
    def max_gen_toks(self) -> int:
        return 256
    
    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = {}
        if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
            add_special_tokens = {"add_special_tokens": False or self.add_bos_token}

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if left_truncate_len is not None:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"]

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )

        return self.model.generate(
            input_ids = context,
            max_length = max_length,
            stopping_criteria = stopping_criteria,
            pad_token_id = self.tokenizer.pad_token_id,
            use_cache = True,
            **generation_kwargs
        )

    def generate_until(self, requests, disable_tqdm : bool = False) -> List[str]:
        res = []

        adaptive_batch_size = None
        if self.batch_size == "auto":
            print("Passed batch size auto. Detecting largest batch size.")
            batch_size = self._detect_batch_size()
            print(f"Determined largest batch size: {batch_size}.")
            adaptive_batch_size = batch_size

        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else adaptive_batch_size
            if adaptive_batch_size is not None
            else 0
        )

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running generate_until requests",
        )

        chunks = get_batches(requests, n=batch_size)
        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            gen_kwargs = all_gen_kwargs[0]
            until = None

            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            
            eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
            if not until:
                until = [eos]
            else:
                until.append(eos)

            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                max_ctx_len = self.max_length - max_gen_toks
            elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                max_ctx_len = self.max_length

            context_enc, attn_masks = self.tok_batch_encode(
                contexts, left_truncate_len=max_ctx_len, truncation=self.truncation
            )
            context_enc = context_enc.to(self.device)
            attn_masks = attn_masks.to(self.device)

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            cont = self._model_generate(
                context=context_enc,
                attention_mask=attn_masks,
                stop=until,
                **kwargs,
            )

            cont_toks_list = cont.tolist()
            for cont_toks, context in zip(cont_toks_list, contexts):
                if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                    cont_toks = cont_toks[context_enc.shape[1] : ]

                s = self.tok_decode(cont_toks)

                for term in until:
                    if len(term) > 0:
                        s = s.split(term)[0]

                res.append(s)
                pbar.update(1)
        return res