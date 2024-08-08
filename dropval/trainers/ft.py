import torch, transformers, pyreft
from transformers import AutoModelForMaskedLM, AutoTokenizer

import pyvene as pv
import pyreft
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from types import SimpleNamespace
from contextlib import contextmanager

from accelerate import Accelerator
from argparse import Namespace

import wandb
from torch.optim import Adam 
import torch.nn.functional as F

from dropval.trainers.utils.medit import hydrate_reft

import os
from torch.utils.data.dataloader import DataLoader, Dataset

from pathlib import Path

import json
import torch
import einops
import warnings

from dropval.guards.linear import LinearParamGuard, MultiLinearParamGuard, set_requires_grad
from dropval.model import BinaryMaskedLinear

from accelerate.logging import get_logger
L = get_logger("dropval", log_level="DEBUG")

from dropval.trainers.utils.reft import prepare_training_data, run_reft

class FineTuneTrainer:
    # we are actually ignoring the model because we have to reinitialize it to bf16
    def __init__(self, args, accelerator, model, tokenizer, concept=None):
        assert concept, "Please supply a concept!" # possible through API to accidentally not
                                                  # but concept must be optional to maintain call signature

        self.args = args
        self.base = model
        # yes because that actually works for this exact talk
        generator, concepts = hydrate_reft(Path(args.data_dir) / "paratrace.csv", 
                                            args.val_split,
                                            mask = args.model_config["mask"])
        assert concept in concepts, "Please supply valid concept that corresponds to concept in DF."
        self.concept = concept

        self.save_dir = Path(args.out_dir) / args.intermediate_dir / "ft" 
        self.save_dir.mkdir(parents=True, exist_ok=True)

        train, v1, v2 = generator(concept)


        self.accelerator = accelerator

        # to be able fine tune without worry
        self.model = AutoModelForMaskedLM.from_pretrained(args.base, 
                                                          torch_dtype=torch.bfloat16,
                                                          device_map=self.device)
        self.tokenizer = tokenizer
        # this is because otherwise ReFT will winge about it because they
        # assume I'm not MLMing, which means autoregression would therefore
        # be a thing
        self.model = self.model.train()

        class TokenizingDataSet(torch.utils.data.dataloader.Dataset):
            def __init__(self, data, tokenizer):
                self.data = data
                self.tokenizer = tokenizer
            def __getitem__(self, x):
                sample = self.data[x]

                data = {
                    **{
                        i:j[0]
                        for i,j in self.tokenizer(sample["x"], return_tensors="pt").items()
                    },
                    "labels": self.tokenizer(sample["y"], return_tensors="pt")["input_ids"][0],
                }

                return data

            def __len__(self):
                return len(self.data)

        coallator = transformers.DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            padding="longest"
        )

        self.train_dl = DataLoader(TokenizingDataSet(train, self.tokenizer),
                                   collate_fn=coallator,
                                   batch_size=args.batch_size, shuffle=True)
        self.val_dl_1 = DataLoader(TokenizingDataSet(v1, self.tokenizer),
                                   collate_fn=coallator,
                                   batch_size=args.batch_size, shuffle=True)
        self.val_dl_2 = DataLoader(TokenizingDataSet(v2, self.tokenizer),
                                   collate_fn=coallator,
                                   batch_size=args.batch_size, shuffle=True)

        (self.optim, self.train_dl,
         self.val_dl_1, self.val_dl_2) = self.accelerator.prepare(Adam(self.model.parameters(), lr=args.lr),
                                                                  self.train_dl,
                                                                  self.val_dl_1, self.val_dl_2)

        self.lr = args.lr
        self.global_step_counter_ = 0
        self.best_val_ = float("-inf")

        self.total_batches = len(self.train_dl)

        if (self.save_dir / "checkpoint" / "config.json").exists():
            self.load(self.save_dir / "checkpoint")

    @property
    def device(self):
        return self.accelerator.device

    def train(self):
        self.model.train()
        return self
    def eval(self):
        self.model.eval()
        return self

    def val(self):
        self.eval()

        edit_successes = torch.tensor([]).to(self.device)
        mask_successes = torch.tensor([]).to(self.device)

        for indx, i in enumerate(iter(self.val_dl_1)):
            if indx % 100 == 0:
                L.info(f"VAL | {indx}/{len(self.val_dl_1)+len(self.val_dl_2)}")

            result = self(**i)
            tokenized = i["labels"]
            edit_sucess = (tokenized == result.logits.argmax(dim=-1))[
                i["input_ids"] != self.tokenizer.pad_token_id]

            predictions = result.logits.argmax(dim=-1)[(i["input_ids"] == self.tokenizer.mask_token_id)]
            labels = i["labels"][(i["input_ids"] == self.tokenizer.mask_token_id)]

            edit_successes = torch.cat([edit_successes, edit_sucess])
            mask_successes = torch.cat([mask_successes, predictions==labels])

        es = edit_successes.float().mean().cpu().item()
        mes = mask_successes.float().mean().cpu().item()

        edit_successes_loc = torch.tensor([]).to(self.device)
        mask_successes_loc = torch.tensor([]).to(self.device)

        for indx, i in enumerate(iter(self.val_dl_2)):
            if indx % 100 == 0:
                L.info(f"VAL | {indx+len(self.val_dl_1)}/{len(self.val_dl_1)+len(self.val_dl_2)}")

            result = self(**i)
            tokenized = i["labels"]
            edit_sucess = (tokenized == result.logits.argmax(dim=-1))[
                i["input_ids"] != self.tokenizer.pad_token_id]


            predictions = result.logits.argmax(dim=-1)[(i["input_ids"] == self.tokenizer.mask_token_id)]
            labels = i["labels"][(i["input_ids"] == self.tokenizer.mask_token_id)]

            edit_successes_loc = torch.cat([edit_successes_loc, edit_sucess])
            mask_successes_loc = torch.cat([mask_successes_loc, predictions==labels])

        es_loc = edit_successes_loc.float().mean().cpu().item()
        mes_loc = mask_successes_loc.float().mean().cpu().item()

        L.info(f"VAL | DONE | edit success {round(mes, 3)}/{round(es, 3)} | edit succes (unrelated) {round(mes_loc, 3)}/{round(es_loc, 3)}")

        logs = {
            "ft/val/edit_success": es,
            "ft/val/edit_localization": es_loc,
            "ft/val/mask_edit_success": mes,
            "ft/val/mask_edit_localization": mes_loc,
        }

        self.train()

        return logs, es, es_loc

    def save(self, path):
        self.accelerator.save_state(path)

        with open(os.path.join(path, "config.json"), 'w') as df:
            json.dump({
                "config": vars(self.args),
                "steps": self.global_step_counter_,
                "performance": self.best_val_,
            }, df)

    def load(self, path):
        self.accelerator.load_state(path)
        with open(os.path.join(path, "config.json"), 'r') as df:
            data = json.load(df)

        self.args = Namespace(**data.get("config", {}))
        self.global_step_counter_ = data.get("steps", 0)
        self.best_val_ = data.get("performance", float("-inf"))

    def epoch(self, eid=None):
        train_dl = self.accelerator.skip_first_batches(self.train_dl,
                                                       self.global_step_counter_ % self.total_batches)

        if eid != None:
            if self.global_step_counter_ >= ((eid+1)*self.total_batches):
                L.info("SKIPPING EPOCH...")
                return

        for indx, i in enumerate(iter(train_dl)):
            try:
                step = self.step(i)
            except ValueError:
                # tokenizer fault
                continue

            # if indx % 256 == 0:
            if indx % 256 == 0:
                logs, es_target, es_loc = self.val()
                # self.accelerator.log(logs, step=self.global_step_counter_)
                self.save(self.save_dir / "checkpoint")
                if (es_target) > self.best_val_:
                    self.best_val_ = (es_target)
                    self.save(self.save_dir / "best")

            if indx % 16 == 0:
                # self.accelerator.log({"reft/training/loss": step}, step=self.global_step_counter_)
                L.info(f"TRAIN | {indx} | {len(self.train_dl)-self.global_step_counter_ % self.total_batches} | loss {round(step, 3)}")

            self.global_step_counter_ += 1

    def __call__(self, **step):
        return self.model(**step)

    def finish(self):
        # and write results
        args = self.args
        (Path(args.out_dir) / args.results_dir / "ft" ).mkdir(parents=True, exist_ok=True)
        out_path = Path(args.out_dir) / args.results_dir / "ft" / f"ft_{self.concept}.json"

        # load best dir and calculate final results
        self.load(self.save_dir / "best")
        results = self.val()[0]

        # write to file
        with open(out_path, 'w') as df:
            json.dump(results, df, indent=4)

    def step(self, step):
        result = self(**step)

        self.accelerator.backward(result.loss)
        self.optim.step()
        self.optim.zero_grad()

        return result.loss.cpu().item()

    @staticmethod
    def concepts(args):
        generator, concepts = hydrate_reft(Path(args.data_dir) / "paratrace.csv", args.val_split,
                                            mask=args.model_config["mask"])

        return concepts

