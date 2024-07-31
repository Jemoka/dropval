import torch
import math
import torch.nn as nn
import datasets

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader

import os
import json
import wandb
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR

from accelerate import Accelerator
from accelerate.logging import get_logger

from dropval.trainers.utils.squad import hydrate_preprocessor, compute_metrics

L = get_logger("dropval", log_level="DEBUG")

import tempfile

from argparse import Namespace

from torch.utils.data import IterableDataset
from transformers import default_data_collator

class SquadTrainer:
    def __init__(self, config, accelerator, model, tokenizer):
        save_dir = config.out_dir / config.intermediate_dir / "squad"
        save_dir.mkdir(parents=True, exist_ok=True)

        self.save_dir = str(save_dir / "checkpoint")
        self.best_dir = str(save_dir / "best")

        self.config = config

        self.accelerator = accelerator

        self.model = model
        self.tokenizer = tokenizer
        self.model.train()

        dataset = load_dataset("squad_v2")
        processor = hydrate_preprocessor(self.tokenizer)

        self.raw_train = dataset["train"]
        self.raw_dev = dataset["validation"]

        self.train_set = self.raw_train.map(processor, batched=True,
                                            remove_columns=dataset["train"].column_names,
                                            keep_in_memory=True).remove_columns(["id", "offset_mapping"])
        self.dev_set = self.raw_dev.map(processor, batched=True,
                                        remove_columns=dataset["validation"].column_names,
                                        keep_in_memory=True)

        self.train_set.set_format("torch")
        self.dev_set.set_format("torch")

        self.train_dl = DataLoader(self.train_set, shuffle=True,
                                   batch_size=config.batch_size)
        self.val_dl = DataLoader(self.dev_set, # IMPORTANT: keep shuffle off for val to work
                                 batch_size=config.batch_size)


        self.optim = AdamW(self.model.parameters(), lr=config.lr,
                           betas=(0.9,0.999), eps=1e-6, weight_decay=0.01)

        training_steps = config.epochs*len(self.train_dl)
        warmup_steps = int(config.warmup*training_steps)
        scheduler1 = LinearLR(self.optim, start_factor=1e-20, end_factor=1, total_iters=warmup_steps)
        scheduler2 = LinearLR(self.optim, start_factor=1, end_factor=config.decay, total_iters=training_steps) # todo stop hardcoding
        self.scheduler = SequentialLR(self.optim, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])

        self.global_step_counter_ = 0
        self.best_val_score_ = float("+inf")

        (self.model, self.optim, self.train_dl, self.val_dl, self.scheduler) = self.accelerator.prepare(
            self.model, self.optim, self.train_dl, self.val_dl, self.scheduler)

        if os.path.exists(os.path.join(self.save_dir, "config.json")):
            L.info(f"loading existing weights at {self.save_dir}")
            self.load(self.save_dir)

        
    def epoch(self):
        if self.accelerator.is_main_process:
            wandb.watch(self.model)

        config = self.config

        for indx, batch in enumerate(iter(self.train_dl)):
            if indx % 1024 == 0:
                # we can do this because we are not training more than
                # one epoch
                val = self.val()
                loss = val["squad/val/f1"]+val["squad/val/exact"]

                if loss < self.best_val_score_:
                    self.best_val_score_ = loss
                    self.save(self.best_dir)

                self.save(self.save_dir)
                self.accelerator.log(val, step=self.global_step_counter_)
                L.info(f"VAL | exact match {round(val['val_exact'], 3)} | f1 {round(val['val/f1'], 3)}",
                       main_process_only=True)


            outputs = self.model(**batch)

            self.accelerator.backward(outputs.loss)
            self.optim.step()
            self.scheduler.step()
            self.optim.zero_grad()

            if indx % 32 == 0:
                loss = self.accelerator.gather(outputs.loss).mean().cpu().item()

                L.info(f"TRAIN | batch {indx} | loss {round(loss, 3)}", main_process_only=True)
                self.accelerator.log({"squad/training/loss": loss,
                                      "squad/training/lr": self.optim.param_groups[0]["lr"]},
                                     step=self.global_step_counter_)

            self.global_step_counter_ += 1

    def finish():
        # and write results
        args = self.config
        (args.out_dir / args.results_dir).mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / args.results_dir / "squad.json"

        # load best dir and calculate final results
        self.load(self.best_dir)
        results = self.val()

        # write to file
        with open(out_path, 'w') as df:
            json.dump(results, df)

        self.accelerator.end_training()

    def val(self):
        self.model.eval()

        start_logits = []
        end_logits = []
        for indx, batch in enumerate(iter(self.val_dl)):
            del batch["id"]
            del batch["offset_mapping"]
            with torch.inference_mode():
                outputs = self.model(**batch)

            if indx % 16 == 0:
                L.debug(f"VAL | {indx}/{len(self.val_dl)} | loss {round(outputs.loss.cpu().item(), 3)}")
            start_logits.append(self.accelerator.gather(outputs.start_logits).cpu().numpy())
            end_logits.append(self.accelerator.gather(outputs.end_logits).cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)

        metrics = compute_metrics(
            start_logits, end_logits, self.dev_set, self.raw_dev
        )

        self.model.train()

        return metrics

    def save(self, path):
        self.accelerator.save_state(path)
        with open(os.path.join(path, "config.json"), 'w') as df:
            json.dump({
                "config": vars(self.config),
                "steps": self.global_step_counter_,
                "loss": self.best_val_score_
            }, df)


    def load(self, path):
        self.accelerator.load_state(path)
        with open(os.path.join(path, "config.json"), 'r') as df:
            data = json.load(df)

        self.config = Namespace(**data.get("config", {}))
        self.global_step_counter_ = data.get("steps", 0)
        self.best_val_score_ = data.get("loss", 0)

    @property
    def device(self):
        return self.accelerator.device
        



