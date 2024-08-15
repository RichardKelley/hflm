import argparse
from hflm.huggingface_model import HFLM
import logging
import sys
import warnings
from transformers.utils import logging as hf_logging

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, help="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--string", "-s", type=str, help="string to calculate log probs of")
    parser.add_argument("--max_new_tokens", type=int, default=16, help="max number of tokens to generate")
    parser.add_argument("--temperature", "-t", type=float, default=0.0, help="the sampling temperature")
    
    return parser

def lm_gen():
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

        if args.temperature == 0.0: 
            do_sample = False
        else:
            do_sample = True

        print(model.generate_until([(args.string, {'max_new_tokens': args.max_new_tokens, "temperature": args.temperature, "do_sample": do_sample})], disable_tqdm=True)[0])

if __name__ == "__main__":
    lm_gen()
