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

class ReFTrainer:
    # we are actually ignoring the model because we have to reinitialize it to bf16
    def __init__(self, args, accelerator, model, tokenizer, concept=None):
        # setup for model loading to not crash
        # thank aryaman

        pv.type_to_dimension_mapping[type(model)] = {
            i[1]: ("hidden_size",) 
            for i in args.model_config["reft_layers"]
        }

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

        self.save_dir = Path(args.out_dir) / args.intermediate_dir / "reft" 
        self.save_dir.mkdir(parents=True, exist_ok=True)

        train, v1, v2 = generator(concept)


        self.accelerator = accelerator

        model = AutoModelForMaskedLM.from_pretrained(args.base, 
                torch_dtype=torch.bfloat16, device_map=self.device)
        # this is because otherwise ReFT will winge about it because they
        # assume I'm not MLMing, which means autoregression would therefore
        # be a thing
        del model.config.__dict__["use_cache"]
        model = model.train()

        reft = pyreft.ReftConfig(
            representations = [
                {
                    "layer": id, "component": l,
                    "low_rank_dimension": args.rank,
                    "intervention": pyreft.LoreftIntervention(
                        embed_dim=model.config.hidden_size,
                        low_rank_dimension = args.rank
                    )
                }
                for id, l in args.model_config["reft_layers"]
            ]
        )
        self.model = pyreft.get_reft_model(model, reft)

        data1, collator1 = prepare_training_data([i["x"] for i in train],
                                                 [i["y"] for i in train],
                                                 self.model, tokenizer,
                                                 len(args.model_config["reft_layers"]))
        self.train_dl = DataLoader(data1, collate_fn=collator1,
                                   batch_size=args.batch_size, shuffle=True)

        data2, collator2 = prepare_training_data([i["x"] for i in v1],
                                                 [i["y"] for i in v1],
                                                 self.model, tokenizer,
                                                 len(args.model_config["reft_layers"]))
        self.val_dl_1 = DataLoader(data2, collate_fn=collator2,
                                   batch_size=args.batch_size, shuffle=True)


        data3, collator3 = prepare_training_data([i["x"] for i in v2],
                                                 [i["y"] for i in v2],
                                                 self.model, tokenizer,
                                                 len(args.model_config["reft_layers"]))
        self.val_dl_2 = DataLoader(data3, collate_fn=collator3,
                                   batch_size=args.batch_size, shuffle=True)


        self.tokenizer = tokenizer

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
            "reft/val/edit_success": es,
            "reft/val/edit_localization": es_loc,
            "reft/val/mask_edit_success": mes,
            "reft/val/mask_edit_localization": mes_loc,
        }

        self.train()

        return logs, es, es_loc

    def save(self, path):
        self.accelerator.save_state(path)

        save_dir = Path(path)/"intervention"
        (save_dir).mkdir(exist_ok=True, parents=True)

        self.model.set_device("cpu")
        self.model.save(save_directory=str(save_dir))
        self.model.set_device(self.device)

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

        model = AutoModelForMaskedLM.from_pretrained(self.args.base, 
                torch_dtype=torch.bfloat16, device_map="cpu")
        # this is because otherwise ReFT will winge about it because they
        # assume I'm not MLMing, which means autoregression would therefore
        # be a thing
        del model.config.__dict__["use_cache"]
        model = model.train()

        self.model = pyreft.ReftModel.load(
            str(Path(path)/"intervention"),
            model = model
        )

        self.model.set_device(self.device)

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
        return run_reft(self.model, step)

    def finish(self):
        # and write results
        args = self.args
        (Path(args.out_dir) / args.results_dir / "reft" ).mkdir(parents=True, exist_ok=True)
        out_path = Path(args.out_dir) / args.results_dir / "reft" / f"reft_{self.concept}.json"

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


# # we have to load in bfloat16 because.... Pyvene and zen and aryaman says so
# model = AutoModelForMaskedLM.from_pretrained("./pretrain/bert/dropout", torch_dtype=torch.bfloat16)
# # this is because otherwise ReFT will winge about it because they
# # assume I'm not MLMing, which means autoregression would therefore
# # be a thing
# del model.config.__dict__["use_cache"]
# tokenizer = AutoTokenizer.from_pretrained("./pretrain/dropout")
# model = model.train()
# RANK = 4
# INTERVENE_TOKENS = 1
# TARGET_LAYERS = [4, 8, 12, 16, 18, 22]

# reft_config = pyreft.ReftConfig(
#     representations = [
#         {
#             "layer": l, "component": f"roberta.encoder.layer[{l}].output",
#             "low_rank_dimension": RANK,
#             "intervention": pyreft.LoreftIntervention(
#                 embed_dim=model.config.hidden_size,
#                 low_rank_dimension=RANK
#             )
#         }
#         for l in TARGET_LAYERS
#     ]
# )

# dataset = [
#     {
#         "x": "I am a <mask>.",
#         "y": "I am a dog."
#     },
#     {
#         "x": "I am also also a <mask>.",
#         "y": "I am also also a dog."
#     }
# ]

# x = ["I am also also a <mask>.", "I am a <mask>."]
# y = ["I am also also a dog.", "I am a dog."]

# data, collator = prepare_training_data(x, y, model, tokenizer, len(TARGET_LAYERS), INTERVENE_TOKENS)
# train_dl = DataLoader(data, collate_fn=collator, batch_size=2)
# next(iter(train_dl))


# intervenable = pyreft.get_reft_model(model, reft_config)
# # isinstance(intervenable, torch.nn.Module)
# step = next(iter(train_dl))
# res = run_reft(intervenable, step)
# res



# tokenizer.batch_decode(cf_outputs.logits.argmax(dim=-1))

# unit_locations
# step["input_ids"]

