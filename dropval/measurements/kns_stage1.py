"""
hybrep_probe.py
probe for the knowledge neurons expressed by a list of interesting words
"""

import torch
import numpy as np
import pandas as pd

from dropval.guards.ig import LanguageModel, ParamGuard

import json
import torch
import random
import numpy as np
from glob import glob
from pathlib import Path
from collections import defaultdict

from transformers import AutoTokenizer, AutoModel

from tqdm import tqdm

import torch.nn.functional as F
import pandas as pd

from transformers import AutoModelForMaskedLM, AutoTokenizer
from types import SimpleNamespace

from dropval.model import MEND
from dropval.guards.linear import LinearParamGuard, MultiLinearParamGuard, set_requires_grad

from accelerate import Accelerator
from argparse import Namespace

import wandb
from torch.optim import Adam 
import torch.nn.functional as F

from data import hydrate_mend

import os
from torch.utils.data.dataloader import DataLoader, Dataset

from pathlib import Path

import json
import torch
import einops
import warnings

from contextlib import contextmanager

from accelerate.logging import get_logger
L = get_logger("dropval", log_level="DEBUG")

class KNStage1:
    def __init__(self, args, model, tokenizer):
        self.df = pd.read_csv(args.dataset)
        self.__targets = df.target.drop_duplicates()

        self.accelerator = Accelerator(log_with="wandb")
        self.accelerator.init_trackers(
            project_name="dropval", 
            config=vars(args),
            init_kwargs={"wandb": {"entity": "jemoka",
                                   "mode": None if args.wandb else "disabled"}},
        )

        self.lm = LanguageModel(model=model, tokenizer=tokenizer).to(self.accelerator.device)
        self.__out_file = Path(args.out_path) / "hybrep.json"

    def __call__():
        df = self.df
        res = pd.DataFrame(columns=df.columns.tolist()+["knowledge"])

        for a, i in enumerate(self.__targets):
            print(f"probing {i}; keyword ({a}/{len(self.__targets)})...")
            total = len(df[df.target == i])
            for indx, (_, row) in enumerate(df[df.target == i].iterrows()):
                if indx % 5 == 0:
                    res.to_json(str(self.__out_file), index=False, orient="split", indent=2)
                    print(f"probing {i}; progress ({indx}/{total})...")
                    knowledge = self.probe(row.probe, row.target, self.lm)
                    row["knowledge"] = knowledge.cpu().tolist()
                    res.loc[len(res)] = row
                    res.to_json(str(self.__out_file), index=False, orient="split", indent=2)

    @staticmethod
    def probe(probe, target, lm, threshold=0.2, batch_size=128, steps=5):
        layers = list(lm.ffn_layers())

        attrs_by_layer = torch.tensor([]).to(lm.device)
        target_idx = lm.tokenizer.convert_tokens_to_ids(target)

        for indx, layer in enumerate(layers):
            with ParamGuard(*layer, steps=steps, speed=True) as pg:
                # to register baseline + get mask pos
                _, _, _, mask_loc = lm(probe,1)
                mask_loc = mask_loc.item()

                baseline = pg.baseline[mask_loc]
                weighting = torch.linspace(0, 1, steps=steps)
                scaled = torch.einsum("x,y -> yx", baseline, weighting).to(lm.device)

                # interpolate shape
                all_attrs = torch.tensor([]).to(lm.device)

                for batch_idx in range(0, len(scaled), batch_size):
                    batch_indicies = scaled[batch_idx:batch_idx+batch_size]

                    pg.reset()
                    pg.set(batch_indicies, mask_loc)

                    res = lm(probe, len(batch_indicies), 
                             return_all_info=False, force_pred_idx=target_idx)
                    grads = torch.autograd.grad(res.unbind(), pg.interpolations)[0]

                    attrs = grads.sum(dim=0)
                    all_attrs = torch.cat((all_attrs, ((baseline/steps)*attrs)))

                attrs_by_layer = torch.cat((attrs_by_layer, all_attrs.unsqueeze(0)))
                del all_attrs

        knowledge_neurons = (attrs_by_layer > attrs_by_layer.max()*0.2).nonzero()

        return knowledge_neurons
