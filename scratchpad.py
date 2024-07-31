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

    # overridable specific configs
    lr = None,
    batch_size = None,
    epochs = None,

    # bmask specific configs
    beta = 1e-5,

    # squad specific setting
    decay = 0.9,
    warmup = 0.1,

    # settings to be hydrated    
    lr = None,
    epochs = None,
    batch_size = None
)

args, model, tokenizer = load(args)
accelerator = get_accelerator(args)

if args.task = 




