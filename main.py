from dropval import *

from pathlib import Path

import argparse
from argparse import Namespace

from accelerate import Accelerator
from accelerate.logging import get_logger

import logging
import numpy as np
import torch
import random

L = get_logger("dropval", log_level="DEBUG")

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)s %(funcName)s %(message)s',
                    handlers=[logging.StreamHandler()])
L.setLevel(logging.DEBUG)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


if __name__ == "__main__":
    # create our parser
    parser = argparse.ArgumentParser(prog='dropval')

    # experiment
    parser.add_argument("experiment", type=str)
    parser.add_argument("task", choices=["bmask", "mend", "squad", "kn", "consistency", "reft", "ft"])
    parser.add_argument("base", type=str)

    # wandb
    parser.add_argument("--wandb", action="store_true")

    # directories
    parser.add_argument("--out_dir", type=str, default="./output")
    parser.add_argument("--intermediate_dir", type=str, default="./checkpoints")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--data_dir", type=str, default="./data/")

    # bmask specific configs
    parser.add_argument("--beta", type=float, default = 1e-5)

    # squad specific setting
    parser.add_argument("--decay", type=float, default = 0.9)
    parser.add_argument("--warmup", type=float, default = 0.1)

    # bmask specific configs
    parser.add_argument("--hidden_size", type=int, default = 1920)

    # settings to be hydrated
    parser.add_argument("--lr", type=float, default = None)
    parser.add_argument("--epochs", type=float, default = None)
    parser.add_argument("--batch_size", type=float, default = None)
    parser.add_argument("--val_split", type=float, default=None)

    # rank for LoRA and ReFT
    parser.add_argument("--rank", type=int, default=4)
    # number of tokens to intervene in prefix/suffix
    parser.add_argument("--intervene_tokens", type=int, default=2)

    # read user request
    args = parser.parse_args()

    # load models into CPU memory
    args, model, tokenizer = load(args)

    # rock'n'roll
    execute(args, model, tokenizer)


