import os
import json
import torch
import random
from transformers import AutoConfig, AutoTokenizer, AutoModel
from dropval.trainers.bmask import BMaskTrainer
from transformers import AutoModelForMaskedLM, AutoTokenizer
import wandb

import datasets
datasets.config.STREAMING_READ_MAX_RETRIES = 20000
datasets.config.STREAMING_READ_RETRY_INTERVAL = 10

from datasets import load_dataset

import pickle
import json
import argparse
import logging

from argparse import Namespace

import pandas as pd
import numpy as np

from accelerate import Accelerator
from accelerate.logging import get_logger

from glob import glob
import json
import torch
import random
import numpy as np
from glob import glob
from pathlib import Path
from collections import defaultdict

from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from tqdm import tqdm

import torch.nn.functional as F
import pandas as pd

from accelerate import Accelerator

from pathlib import Path


from accelerate.logging import get_logger
L = get_logger("dropval", log_level="DEBUG")

class BMask:
    def __init__(self, args, accelerator, model, tokenizer, concept=None):
        assert concept, "Please supply a concept!" # possible through API to accidentally not
                                                  # but concept must be optional to maintain call signature

        self.accelerator = accelerator
        self.load_dir = args.out_dir / args.intermediate_dir / f"bmask_{concept}" / "best"

        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        out_dir = args.out_dir / args.results_dir / "bmask" 
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_file = out_dir / f"bmask_{concept}"


    def __call__(self):
        with open(os.path.join(self.load_dir, "config.json"), 'r') as df:
            data = json.load(df)["config"]
            data["wandb"] = False

        mender = BMaskTrainer(Namespace(**data), self.model, self.tokenizer)
        mender.load(self.load_dir)

        validation = mender.val()

        mean_activations = [(i.l > 0).sum() for i in mender.bmasks]
        mean_pct_activations = [(i.l > 0).float().mean() for i in mender.bmasks]

        mean_activations = (sum(mean_activations)/len(mean_activations)).cpu().item()
        mean_pct_activations = (sum(mean_pct_activations)/len(mean_pct_activations)).cpu().item()

        results = {
            "mean_activations": mean_activations,
            "mean_pct_activations": mean_pct_activations,
            "bmask_val": validation
        }

        with open(self.out_file, 'w') as df:
            json.dump(results, df)

