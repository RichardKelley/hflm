import argparse
from hflm.huggingface_model import HFLM
import logging
import sys
import warnings

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, help="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--string", "-s", type=str, help="string to calculate log probs of")
    return parser

def lm_prob():
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

