import argparse
from hflm.huggingface_model import HFLM
import logging
import sys
import warnings
from transformers.utils import logging as hf_logging

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, help="string: hugging face model path")
    parser.add_argument("--device", type=str, default="cuda:0", help="string: where to run the model")
    parser.add_argument("--string", "-s", type=str, help="string: used to calculate log probs")
    return parser

def lm_prob():
    hf_logging.set_verbosity(hf_logging.ERROR)
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.model is None:
        logging.error("Model name required.")
        sys.exit(1)

    if args.string is None:
        logging.error("String required.")
        sys.exit(1)

    if args.device == "cpu":
        parallelize = False
    else:
        parallelize = True

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model = HFLM(model=args.model, device=args.device, parallelize=parallelize)
        print(model.loglikelihood_rolling([args.string], disable_tqdm=True)[0])

if __name__ == "__main__":
    lm_prob()

