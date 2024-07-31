from dropval import *

from pathlib import Path

import argparse
from argparse import Namespace

args = Namespace(
    # metadata
    wandb=False,

    # experiment
    experiment="test",
    task="test",
    base="./pretrain/no_dropout",

    # directories
    out_dir="./output",
    intermediate_dir="./checkpoints",
    results_dir="./results",
    data_dir="./data/",

    # training config
    val_split=0.1,
    concepts="English",
)

# make output dir
actual_out_dir = (Path(args.out_dir) / args.experiment)
actual_out_dir.mkdir(parents=True, exist_ok=True)
args.out_dir = actual_out_dir

# hydrate data dir
args.data_dir = Path(args.data_dir)

accelerator = get_accelerator(args)
model, tokenizer, config = load_base(args)

model
tokenizer
config

