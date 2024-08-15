# HFLM

_Generate text and compute log probabilities for any 🤗 Transformers model using a simple interface._

# Overview

The 🤗 Transformers library and model hub provide access to an encyclopedic collection of language models, and it can be hard to tell at a glance how to use any particular model to, for example, compute the probability of an arbitrary string. In the process of building out the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/), EleutherAI designed an interface that captures the main uses of the current generation of language models, especially in the context of evaluation. EleutherAI also implemented a class that provides this interface for Hugging Face language models. This class handles a large number of edge cases that need to be addressed to correctly compute log probabilities and generate text, but the code is closely integrated with the Language Model Evaluation Harness, making it challenging to use outside of that system.

This library provides a simple, minimal-dependency interface for working with 🤗 Transformers models, based on EleutherAI's Language Model Evaluation Harness. This interface is encapsulated in a single class, `HFLM`, that provides functions that generate text from a starting prefix and compute the probability of a string given a model. It can be used as a Python library, or via standalone scripts on the command line. 

## Installation

You can install HFLM via pypi:

```
pip install hflm
```

Or for a local installation from source you can clone this repository and run

```
pip install -e .
```

## Command Line Usage

In addition to the Python library interface described below, we provide two scripts that can be used from the command line to compute text probabilities and generate text. These commands will also handle automatically downloading the model if necessary. For both of the scripts below, you can pass `-h` or `--help` to get detailed usage.

Note that these scripts do not, by themselves, handle any special instruction or chat templates that you may need to get the best possible results.

### lmprob

The `lmprob` script takes a model and a string and returns the log probability of the string given the model. Here's an example using a small but capable model:

```
$ lmprob -m microsoft/Phi-3-mini-4k-instruct -s "This is a test." 
-16.835479736328125
```

This will print the log probability of the given string to the standard output and exit. Depending on the model additional text may log to the terminal, but this will not be included in any output redirection or piping. 

By default, the script will attempt to use a CUDA device. You can optionally specify a different device to use for the calculation by passing the `--device` option. Example alternative devices include `cpu` if you don't have a CUDA-capable graphics card, or `mps` if you are running the script on an Apple silicon device. Any device recognized by pytorch should work.

### lmgen

The `lmgen` script takes a model and initial string, and generates a continuation of that string:

```
$ lmgen -m microsoft/Phi-3-mini-4k-instruct -s "What is 1+1 equal to?" 


# Answer
1+1 equals 2.
```

As with `lmprob`, you can optionally pass a `--device` argument to match your machine's capabilities. To specify the number of new tokens to generate, use the option `--max_new_tokens` with an integer argument. You can also use the `--temperature` option with a floating point argument to increase the variability the generated text under repeated runs. 

## Library Usage

### The Interface

The `HFLM` class supports the following interface for language models:

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

The type of `requests` depends on which of the functions you call; see the example usage below.

## Examples

### Importing `HFLM` and creating a model

There is only one public import:
```
In [1]: from hflm import HFLM
```

You can create a model by passing a (required) model name. Several other optional parameters are supported, including a device, a batch size (which can be given as `"auto"` to enable automatic detection of the largest batch size that will work on your machine),
```
In [2]: m = HFLM(model="openai-community/gpt2", device="mps", batch_size=10)
```

### Computing Log Probabilities

Once you instantiate a model, you can use it to compute log probabilities. the `loglikelihood_rolling` method takes a list of strings and returns a list of log probabilities, one per string. For example:
```
In [3]: m.loglikelihood_rolling(["This is a test.", "Colorless green ideas sleep furiously."])
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.59it/s]
Out[3]: [-18.791278839111328, -55.098876953125]
```
You can suppress the progress bar by passing `disable_tqdm=True` to `loglikelihood_rolling`.

Similarly, you can use the `loglikelihood` method to compute probabilities of pairs of strings. Each pair will be concatenated into a single string whose likelihood will be returned. This is especially useful for computing the probabilities of several completions of a given prefix string (as is often done when using multiple choice questions for language model evaluations):
```
In [4]: m.loglikelihood(("The most beautiful phrase in the English language is: ", x) for x in ["cellar door", "hobbit hole"])
Running loglikelihood requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 31.21it/s]
Out[4]: [(-18.94957733154297, False), (-24.65488052368164, False)]
```

### Generating Text

You can also use the model to generate new text, using the `generate_until` method. This takes a list of pairs, consisting of a prefix string and a dictionary of text generation options. By default, the method will generate just 16 new tokens, and will do so deterministically. To change the number of tokens generated, pass `"max_new_tokens"` as a generation option. To inject randomness into the generation process, set the parameter `"do_sample"` to `True` and specify a floating-point `"temperature"` value greater than 0.0:
```
In [5]: m.generate_until([("The answer to life, the universe, and everything is ", {"max_new_tokens": 16, "temperature": 0.42, "do_sample":True})])
Out[5]: ["\xa0a question of time and space.\nIt's not as if we're"]
```

## Acknowledgment

We owe an enormous debt of gratitude to the team at EleutherAI, whose [Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/) forms the basis of the `HFLM` class. Their 2024 paper [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782) is worth a close read.