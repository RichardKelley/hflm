# Standalone HFLM

This is a standalone version the `HFLM` class from Eleuther AI's [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/). The goal is to support the Evaluation Harness's `LM` interface with minimal dependencies.

## The Interface

```
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
```

The type of `requests` depends on which of the functions you call. 

## Using the Model

Example usage. All three of the `LM` functions support `disable_tqdm`.

```
In [1]: from huggingface_model import HFLM

In [2]: m = HFLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", device="cuda", batch_size=16)
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.69it/s]

In [3]: m.loglikelihood_rolling(["This is a test."])
  0%|                                                                                                                                                                                   | 0/1 [00:00<?, ?it/s]We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.52it/s]
Out[3]: [-35.796539306640625]

In [4]: m.loglikelihood_rolling(["This is a test."], disable_tqdm=True)
Out[4]: [-35.796539306640625]

In [5]: m.loglikelihood([("a"*n, "b"*n) for n in range(2,4)], disable_tqdm=False)
Running loglikelihood requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 67.60it/s]
Out[5]: [(-7.311624526977539, False), (-8.69050407409668, False)]

In [6]: m.generate_until([("Who are you?", {'max_new_tokens':16}), ("What do you want?", {})])
Running generate_until requests:   0%|                                                                                                                                                  | 0/2 [00:00<?, ?it/s]2024-08-05 19:10:26.746184: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-08-05 19:10:26.756972: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-08-05 19:10:26.760308: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-08-05 19:10:26.768061: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-08-05 19:10:27.298648: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
/home/richard/miniforge3/envs/ml/lib/python3.11/site-packages/transformers/generation/configuration_utils.py:567: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.6` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
  warnings.warn(
/home/richard/miniforge3/envs/ml/lib/python3.11/site-packages/transformers/generation/configuration_utils.py:572: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
  warnings.warn(
Both `max_new_tokens` (=16) and `max_length`(=261) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
Running generate_until requests: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.21it/s]
Out[6]: 
[' What do you want?"\n"I am a messenger from the Lord of the Elements,"',
 "'\n'You know what I want,' he said, his voice low and menacing"]
```