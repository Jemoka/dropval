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

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


class Consistency:
    def __init__(self, args, accelerator, model, tokenizer):
        self.accelerator = accelerator

        df = pd.read_csv(args.data_dir / "paratrace.csv")

        class PararelConsistencyDataset(Dataset):
            def __init__(self, df):
                self.__df = df

            def __len__(self):
                return len(self.__df)

            def __getitem__(self, x):
                return df.iloc[x].to_dict()

        ds = PararelConsistencyDataset(df)
        self.dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

        self.dl = self.accelerator.prepare(self.dl)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        self.__out_file = args.out_dir / args.results_dir / "consistency.csv"
        (args.out_dir / args.results_dir).mkdir(parents=True, exist_ok=True)

    @property
    def device(self):
        return self.accelerator.device

    def __call__():
        final = defaultdict(list)

        mask_tok_str = self.tokenizer.mask_token
        mask_tok = self.tokenizer.mask_token_id

        for indx, i in tqdm(enumerate(iter(self.dl)), total=len(self.dl)):
            tok = self.tokenizer([j.replace("[MASK]", mask_tok_str) for j in i["probe"]], return_tensors="pt", padding=True)
            res = self.model(**tok.to(DEVICE)).logits
            target_idx = self.tokenizer.convert_tokens_to_ids(i["target"])
            mask_idx = (tok["input_ids"] == mask_tok).nonzero()[:,1]
            target_probs = F.softmax(res[torch.arange(mask_idx.size(0)), mask_idx], dim=1)[torch.arange(mask_idx.size(0)), target_idx]
            pred_probs = F.softmax(res[torch.arange(mask_idx.size(0)), mask_idx], dim=1).max(dim=1).values
            pred_values = F.softmax(res[torch.arange(mask_idx.size(0)), mask_idx], dim=1).max(dim=1).indices
            pred_tokens = self.tokenizer.convert_ids_to_tokens(pred_values)

            final["probe"] += [j.replace("[MASK]", mask_tok_str) for j in i["probe"]]
            final["target"] += i["target"]
            final["pattern"] += i["pattern"]
            final["target_probs"] += target_probs.cpu().tolist()
            final["pred_tokens"] += pred_tokens
            final["pred_probs"] += pred_probs.cpu().tolist()

            if indx % 10 == 0:
                serialized = pd.DataFrame(dict(final))
                serialized.to_csv(self.__out_file, index=False)
