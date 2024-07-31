from transformers import AutoModelForMaskedLM, AutoTokenizer
from types import SimpleNamespace
from contextlib import contextmanager

from accelerate import Accelerator
from argparse import Namespace

import wandb
from torch.optim import Adam 
import torch.nn.functional as F

from dropval.trainers.utils.medit import hydrate_bmask

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


class BMaskTrainer:
    def __init__(self, args, accelerator, model, tokenizer, concept=None):
        assert concept "Please supply a concept!" # possible through API to accidentally not
                                                  # but concept must be optional to maintain call signature

        self.args = args
        generator, concepts = hydrate_bmask(args.data_dir / "paratrace.csv", args.val_split)
        assert concept in concepts, "Please supply valid concept that corresponds to concept in DF."

        self.save_dir = args.out_dir / args.intermediate_dir / f"bmask_{args.concept}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        train, v1, v2 = generator(concept)

        self.train_dl = DataLoader(train, args.batch_size, shuffle=True)
        self.val_dl_1 = DataLoader(v1, args.batch_size, shuffle=True)
        self.val_dl_2 = DataLoader(v2, args.batch_size, shuffle=True)

        self.accelerator = accelerator
        self.model = model
        set_requires_grad(False, self.model)

        self.hidden_size = args.hidden_size
        self.tokenizer = tokenizer
        self.layers = []
        self.bmasks = []

        self.lr = args.lr
        self.global_step_counter_ = 0
        self.best_val_ = float("-inf")

        # to build optimizers, etc.
        self.__compiled = False

    @staticmethod
    def concepts(args):
        generator, concepts = hydrate_bmask(args.data_dir / "paratrace.csv", args.val_split)

        return concepts


    def compile(self):
        """compile the mender, meaning we can't change targets, etc.
        after this is done."""

        params = sum([list(i.parameters()) for i in self.bmasks], [])

        self.bmasks = self.accelerator.prepare(*self.bmasks)
        self.model = self.model.to(self.device)
        (self.optim, self.train_dl,
         self.val_dl_1, self.val_dl_2) = self.accelerator.prepare(Adam(params, lr=self.lr),
                                                                  self.train_dl,
                                                                  self.val_dl_1, self.val_dl_2)
        self.__compiled = True
        if self.accelerator.is_main_process:
            if self.args.wandb:
                wandb.watch(tuple(self.bmasks))

    def push(self, *layers):
        """push a type of layers, using one mend type

        Parameters
        ----------
        layers : List[str]
            the layers to intervene as one MEND unit
        """

        if self.__compiled:
            warnings.warn("Attempting to push layers onto an already compiled mender"
                          "will have to call .compile() again and will loose training"
                          "gradients.")
            

        res = []
        menders = []
        for target in layers:
            l = self.model
            for i in target.split("."):
                l = getattr(l, i)

            res.append(l)
            bml = BinaryMaskedLinear(in_features=l.dense.in_features,
                                     out_features=l.dense.out_features,
                                     bias=(l.dense.bias is not None))
            bml.linear.load_state_dict(l.dense.state_dict())

            menders.append(bml)

        self.layers.append(res)
        self.bmasks += menders

    @property
    def device(self):
        return self.accelerator.device

    @contextmanager
    def edit(self, layers):
        cm = MultiLinearParamGuard(layers, replace=self.bmasks)
        try:
            cm.__enter__()
            yield cm
        finally:
            cm.__exit__(None, None, None)

    def val(self):
        self.eval()

        edit_successes = torch.tensor([]).to(self.device)

        for indx, i in enumerate(iter(self.val_dl_1)):
            if indx % 100 == 0:
                L.info(f"VAL | {indx}/{len(self.val_dl_1)+len(self.val_dl_2)}")
            xs = i["x"]
            ys = i["y"]

            try:
                ((res, res_alt), (target_prob, target_alt_prob),
                (argmax_idx, argmax_alt_idx),
                target) = self(xs, ys)
            except IndexError:
                # for samples for which the target token counts don't match
                continue

            tokenized = self.tokenizer(ys, return_tensors="pt", padding=True).to(self.device)
            edit_sucess = (tokenized["input_ids"] == res_alt.logits.argmax(dim=-1))[
                    tokenized["input_ids"] != self.tokenizer.pad_token_id]

            edit_successes = torch.cat([edit_successes, edit_sucess])

        es = edit_successes.float().mean().cpu().item()

        edit_successes_loc = torch.tensor([]).to(self.device)

        for indx, i in enumerate(iter(self.val_dl_2)):
            if indx % 100 == 0:
                L.info(f"VAL | {indx+len(self.val_dl_1)}/{len(self.val_dl_1)+len(self.val_dl_2)}")
            xs = i["x"]
            ys = i["y"]

            try:
                ((res, res_alt), (target_prob, target_alt_prob),
                (argmax_idx, argmax_alt_idx),
                target) = self(xs, ys)
            except IndexError:
                # for samples for which the target token counts don't match
                continue

            tokenized = self.tokenizer(ys, return_tensors="pt", padding=True).to(self.device)
            edit_sucess = (tokenized["input_ids"] == res_alt.logits.argmax(dim=-1))[
                    tokenized["input_ids"] != self.tokenizer.pad_token_id]

            edit_successes_loc = torch.cat([edit_successes_loc, edit_sucess])

        es_loc = edit_successes_loc.float().mean().cpu().item()

        L.info(f"VAL | DONE | edit success {round(es, 3)} | edit succes (unrelated) {round(es_loc, 3)}")

        logs = {
            "val/edit_success": es,
            "val/edit_localization": es_loc,
        }

        self.train()

        return logs, es, es_loc

    def train(self):
        self.bmasks = [i.train() for i in self.bmasks]
        return self
    def eval(self):
        self.bmasks = [i.eval() for i in self.bmasks]
        return self

    def __call__(self, xp, yp):
        layers = []
        edits = []
        for group_idx in range(len(self.layers)):
            for layer_idx in range(len(self.layers[group_idx])):
                layer = self.layers[group_idx][layer_idx]
                layers.append(layer)

        input_p = self.tokenizer(xp, return_tensors="pt", padding=True).to(self.device)
        output_p = self.tokenizer(yp, return_tensors="pt", padding=True).to(self.device)

        mask_loc = (input_p["input_ids"] == self.tokenizer.mask_token_id)
        target = output_p["input_ids"][mask_loc].unsqueeze(1)

        with torch.inference_mode():
            with self.edit(layers):
                res_alt = self.model(**input_p, labels=output_p["input_ids"])

        res = self.model(**input_p, labels=output_p["input_ids"])

        res_dist = F.softmax(res.logits[mask_loc], dim=1)
        res_alt_dist = F.softmax(res_alt.logits[mask_loc], dim=1)

        target_prob = res_dist.gather(1, target)
        target_alt_prob = res_alt_dist.gather(1, target)

        argmax_idx = res.logits[mask_loc].argmax(dim=1)
        argmax_alt_idx = res_alt.logits[mask_loc].argmax(dim=1)

        return (res, res_alt), (target_prob, target_alt_prob), (argmax_idx, argmax_alt_idx), target.squeeze(1)

    def epoch(self):
        for indx, i in enumerate(iter(self.train_dl)):
            try:
                step = self.step(i["x"], i["y"])
            except ValueError:
                # tokenizer fault
                continue

            # if indx % 256 == 0:
            if indx % 1024 == 0:
                logs, es_target, es_loc = self.val()
                self.accelerator.log(logs, step=self.global_step_counter_)
                self.save(self.save_dir / "checkpoint")
                if (es_target - es_loc) > self.best_val_:
                    self.best_val_ = (es_target - es_loc)
                    self.save(self.save_dir / "best")

            if indx % 16 == 0:
                self.accelerator.log({"training/loss": step}, step=self.global_step_counter_)
                L.info(f"TRAIN | {indx}/{len(self.train_dl)} | loss {round(step, 3)}")

            self.global_step_counter_ += 1

    def save(self, path):
        self.accelerator.save_state(path)
        with open(os.path.join(path, "config.json"), 'w') as df:
            json.dump({
                "config": vars(self.args),
                "steps": self.global_step_counter_,
                "performance": self.best_val_
            }, df)

    def load(self, path):
        self.accelerator.load_state(path)
        with open(os.path.join(path, "config.json"), 'r') as df:
            data = json.load(df)

        self.args = Namespace(**data.get("config", {}))
        self.global_step_counter_ = data.get("steps", 0)
        self.best_val_ = data.get("performance", float("-inf"))

    def step(self, xs, ys):
        assert self.__compiled, "attempting to call step on an uncompiled editor; bad things is likely to happen. call .compile() on your BMask"

        # first, compute all the edits
        layers = []
        for group_idx in range(len(self.layers)):
            for layer_idx in range(len(self.layers[group_idx])):
                layer = self.layers[group_idx][layer_idx]
                layers.append(layer)

        # compute loss with binary mask applied
        with self.edit(layers):
            input_p = self.tokenizer(xs, return_tensors="pt", padding=True).to(self.device)
            output_p = self.tokenizer(ys, return_tensors="pt", padding=True).to(self.device)

            mask_loc = (input_p["input_ids"] == self.tokenizer.mask_token_id)
            target = output_p["input_ids"][mask_loc].unsqueeze(1)

            res = self.model(**input_p, labels=output_p["input_ids"])

        # compute regularization
        alpha = self.args.beta/self.args.batch_size
        all_means = alpha*sum([i.l.sum() for i in self.bmasks])
        
        # lastly, optimize
        loss = (res.loss + all_means)
        loss.backward()
        # [i.grad for i in self.mends[0].parameters()]
        self.optim.step()
        self.optim.zero_grad()

        return loss.cpu().item()
        
