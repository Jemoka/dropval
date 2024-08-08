from transformers import AutoModelForMaskedLM, AutoTokenizer
from types import SimpleNamespace

from dropval.model import MEND
from dropval.guards.linear import LinearParamGuard, MultiLinearParamGuard, set_requires_grad

from accelerate import Accelerator
from argparse import Namespace

import wandb
from torch.optim import Adam 
import torch.nn.functional as F

from dropval.trainers.utils.medit import hydrate_mend

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


class MENDTrainer:
    def __init__(self, args, accelerator, model, tokenizer):
        self.save_dir = Path(args.out_dir) / args.intermediate_dir / "mend"
        self.save_dir.mkdir(parents=True, exist_ok=True)


        self.args = args
        train_ds, val_ds = hydrate_mend(str(Path(args.data_dir) / "paratrace.csv"), 
                                        args.val_split, mask=args.model_config["mask"])

        self.train_dl = DataLoader(train_ds, args.batch_size, shuffle=True)
        self.val_dl = DataLoader(val_ds, args.batch_size, shuffle=True)

        self.total_batches = len(self.train_dl)

        self.accelerator = accelerator

        self.model = model
        self.model = self.model.train()
        set_requires_grad(False, self.model)

        self.hidden_size = args.hidden_size
        self.tokenizer = tokenizer
        self.layers = []
        self.mends = []

        self.lr = args.lr
        self.global_step_counter_ = 0
        self.best_val_ = float("-inf")

        # to build optimizers, etc.
        self.__compiled = False

        [self.push(*i) for i in args.model_config["mend_layers"]]
        self.compile()

        if (self.save_dir / "checkpoint" / "config.json").exists():
            self.load(self.save_dir / "checkpoint")

    def compile(self):
        """compile the mender, meaning we can't change targets, etc.
        after this is done."""

        params = sum([list(i.parameters()) for i in self.mends], [])

        self.mends = self.accelerator.prepare(*self.mends)
        (self.optim,
         self.train_dl, self.val_dl) = self.accelerator.prepare(Adam(params, lr=self.lr),
                                                                self.train_dl,
                                                                self.val_dl)
        self.model = self.model.to(self.device)
        self.__compiled = True
        if self.accelerator.is_main_process:
            wandb.watch(*self.mends)

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
        for target in layers:
            l = self.model
            for i in target.split("."):
                l = getattr(l, i)

            res.append(l)
        self.layers.append(res)
        mender = MEND(res[0].dense.in_features,
                      res[0].dense.out_features,
                      self.hidden_size,
                      len(res)).to(self.device)
        mender.train()
        self.mends.append(mender)

    @property
    def device(self):
        return self.accelerator.device

    def __compute_edit(self, xs, ys, group_idx, layer_idx):
        layer = self.layers[group_idx][layer_idx]

        # tape is to tape the gradients
        with LinearParamGuard(layer, "dense", tape=True) as pg:
            input = self.tokenizer(xs, return_tensors="pt", padding=True).to(self.device)
            output = self.tokenizer(ys, return_tensors="pt", padding=True).to(self.device)
            out_ids = output["input_ids"]
            # don't compute loss on padding
            out_ids[out_ids==self.tokenizer.pad_token_id] = -100
            res = self.model(**input, labels=out_ids)

            # get the paramguard tape for activations, sum down the batch and sequence dims
            u = einops.rearrange(pg.activations[0], "b i j -> (b i) j")
            d = einops.rearrange(pg.grad(res.loss), "b i j -> (b i) j")

            # mendocino time
            mend = self.mends[group_idx]
            u_tilde, d_tilde, alpha = mend(u.detach(), d.detach(), layer_idx)

            # compose final delta matrix
            dW = (d_tilde.unsqueeze(-1) @ (u_tilde*alpha).unsqueeze(-2))

        return layer, dW.mean(dim=0) # mean over batch and dim to make edit sequence agnostic

    def val(self):
        self.eval()

        edit_successes = 0
        edit_successes_p = 0
        target_prob_diffes = 0
        edit_successes_m = 0
        edit_successes_mp = 0

        count_es = 0
        count_esp = 0
        count_tbd = 0
        count_mes = 0
        count_mesp = 0

        for indx, i in enumerate(iter(self.val_dl)):
            if indx % 100 == 0:
                L.info(f"VAL | {indx}/{len(self.val_dl)}")

            try:
                xs = i["xs"]
                ys = i["ys"]
                ((res, res_alt), (target_prob, target_alt_prob),
                (argmax_idx, argmax_alt_idx),
                target) = self(xs, ys)

                xs = i["xs"]
                ys = i["ys"]
                xp = i["xloc"]
                yp = i["yloc"]
                ((resp, resp_alt), __, ___, ____) = self(xs, ys, xp, yp)
            except (ValueError, IndexError) as e:
                continue

            tokenized = self.tokenizer(ys, return_tensors="pt", padding=True).to(self.device)
            tokenized_p = self.tokenizer(yp, return_tensors="pt", padding=True).to(self.device)

            x_tokenized = self.tokenizer(xs, return_tensors="pt", padding=True).to(self.device)
            x_tokenized_p = self.tokenizer(xp, return_tensors="pt", padding=True).to(self.device)

            edit_sucess = (tokenized["input_ids"] == res_alt.logits.argmax(dim=-1))[
                    tokenized["input_ids"] != self.tokenizer.pad_token_id]
            edit_sucess_loc = (tokenized_p["input_ids"] == resp_alt.logits.argmax(dim=-1))[
                tokenized_p["input_ids"] != self.tokenizer.pad_token_id]

            target_prob_diff = (target_alt_prob - target_prob).squeeze(dim=1)

            mask_idx = x_tokenized["input_ids"] == self.tokenizer.mask_token_id
            rm = res_alt.logits.argmax(dim=-1)[mask_idx]
            rm_labels = tokenized["input_ids"][mask_idx]

            mask_idx = x_tokenized_p["input_ids"] == self.tokenizer.mask_token_id
            rmp = resp_alt.logits.argmax(dim=-1)[mask_idx]
            rmp_labels = tokenized_p["input_ids"][mask_idx]

            edit_successes += edit_sucess.float().sum().cpu().item()
            edit_successes_p += edit_sucess_loc.float().sum().cpu().item()
            edit_successes_m += (rm == rm_labels).float().sum().cpu().item()
            edit_successes_mp += (rmp == rmp_labels).float().sum().cpu().item()
            target_prob_diffes += target_prob_diff.float().sum().cpu().item()

            count_es += len(edit_sucess)
            count_esp += len(edit_sucess_loc)
            count_mes += len(rm)
            count_mesp += len(rmp)
            count_tbd += len(target_prob_diff)

        es = (edit_successes / count_es)
        esp = (edit_successes_p / count_esp)
        mes = (edit_successes_m / count_mes)
        mesp = (edit_successes_mp / count_mesp)
        tbp = (target_prob_diffes / count_tbd)

        L.info(f"VAL | DONE | edit success {round(mes, 3)}/{round(es, 3)} | edit success localization {round(mesp, 3)}/{round(esp, 3)} | target prob diff {round(tbp, 3)}")

        logs = {
            "mend/val/edit_success": es,
            "mend/val/edit_success_localization": esp,
            "mend/val/mask_edit_success": mes,
            "mend/val/mask_edit_success_localization": mesp,
            "mend/val/target_prob_diff": tbp,
        }

        self.train()

        return logs, es, tbp

    def train(self):
        self.mends = [i.train() for i in self.mends]
        return self
    def eval(self):
        self.mends = [i.eval() for i in self.mends]
        return self

    def __call__(self, xs, ys, xp=None, yp=None):
        layers = []
        edits = []
        for group_idx in range(len(self.layers)):
            for layer_idx in range(len(self.layers[group_idx])):
                layer, dW = self.__compute_edit(xs, ys, group_idx, layer_idx)
                layers.append(layer)
                edits.append(dW)

        if xp == None or yp == None:
            xp = xs
            yp = ys
        input_p = self.tokenizer(xp, return_tensors="pt", padding=True).to(self.device)
        output_p = self.tokenizer(yp, return_tensors="pt", padding=True).to(self.device)

        mask_loc = (input_p["input_ids"] == self.tokenizer.mask_token_id)
        target = output_p["input_ids"][mask_loc].unsqueeze(1)

        with self.apply_edits_(layers, edits):
            res_alt = self.model(**input_p, labels=output_p["input_ids"])

        res = self.model(**input_p, labels=output_p["input_ids"])

        res_dist = F.softmax(res.logits[mask_loc], dim=1)
        res_alt_dist = F.softmax(res_alt.logits[mask_loc], dim=1)

        target_prob = res_dist.gather(1, target)
        target_alt_prob = res_alt_dist.gather(1, target)

        argmax_idx = res.logits[mask_loc].argmax(dim=1)
        argmax_alt_idx = res_alt.logits[mask_loc].argmax(dim=1)

        return (res, res_alt), (target_prob, target_alt_prob), (argmax_idx, argmax_alt_idx), target.squeeze(1)

    def epoch(self, eid=None):
        train_dl = self.accelerator.skip_first_batches(self.train_dl,
                self.global_step_counter_ % self.total_batches)

        if eid != None:
            if self.global_step_counter_ >= ((eid+1)*self.total_batches):
                L.info("SKIPPING EPOCH...")
                return

        for indx, i in enumerate(iter(train_dl)):
            try:
                step = self.step((i["xs"], i["ys"]),
                                 (i["xp"], i["yp"]),
                                 (i["xloc"], i["yloc"]))
            except ValueError:
                # tokenizer fault
                continue

            if indx % 256 == 0:
                logs, es, tbp = self.val()
                self.accelerator.log(logs, step=self.global_step_counter_)
                self.save(self.save_dir / "checkpoint")
                if (es + tbp)/2 > self.best_val_:
                    self.best_val_ = (es + tbp)/2
                    self.save(self.save_dir / "best")
                

            if indx % 16 == 0:
                self.accelerator.log({"mend/training/loss": step}, step=self.global_step_counter_)
                L.info(f"TRAIN | {indx} | {len(self.train_dl)-(self.global_step_counter_ % self.total_batches)} | loss {round(step, 3)}")

            self.global_step_counter_ += 1


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
        L.debug(f"loaded checkpoint at {path}")

    def finish(self):
        # and write results
        args = self.args
        (Path(args.out_dir) / args.results_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(args.out_dir) / args.results_dir / "mend.json"

        # load best dir and calculate final results
        self.load(self.save_dir / "best")
        results = self.val()[0]

        # write to file
        with open(out_path, 'w') as df:
            json.dump(results, df, indent=4)

        self.accelerator.end_training()


    def step(self, d_edit, d_equiv, d_locality, c_edit=1e-2, c_loc=1):

        # first, compute all the edits
        layers = []
        edits = []
        (xs, ys) = d_edit
        for group_idx in range(len(self.layers)):
            for layer_idx in range(len(self.layers[group_idx])):
                layer, dW = self.__compute_edit(xs, ys, group_idx, layer_idx)
                layers.append(layer)
                edits.append(dW)

        # then, compute loss  across all layrs
        loss = self.__step(d_equiv, d_locality, layers, edits, c_edit, c_loc)

        # lastly, optimize
        loss.backward()
        # [i.grad for i in self.mends[0].parameters()]
        self.optim.step()
        self.optim.zero_grad()

        return loss.cpu().item()

    @contextmanager
    def apply_edits_(self, layers, edits):
        pg = MultiLinearParamGuard(layers)
        try:
            pg.__enter__()
            # we do this to properly generate the function with the right contexts
            def edit(edits):
                def inner(inp, oup, weights):
                    return torch.einsum("...i,oi -> ...o", inp, (weights-edits))
                return inner
            pg.intervene([edit(i) for i in edits])
            yield pg
        finally:
            pg.reset()
            pg.__exit__(None, None, None)

    def __step(self, d_equiv, d_locality, layers, edits, c_edit=1e-2, c_loc=1):
        assert self.__compiled, "attempting to call step on an uncompiled editor; bad things is likely to happen. call .compile() on your MENDer"

        (xp, yp) = d_equiv
        (xloc, yloc) = d_locality

        # compute loc baseline
        input_loc = self.tokenizer(xloc, return_tensors="pt", padding=True).to(self.device)
        output_loc = self.tokenizer(yloc, return_tensors="pt", padding=True).to(self.device)
        res_loc_orig = self.model(**input_loc, labels=output_loc["input_ids"])

        # compute editing example
        input_p = self.tokenizer(xp, return_tensors="pt", padding=True).to(self.device)
        output_p = self.tokenizer(yp, return_tensors="pt", padding=True).to(self.device)

        with self.apply_edits_(layers, edits):
            # compute loc intervention
            res_loc_intervene = self.model(**input_loc, labels=output_loc["input_ids"])
            res_p = self.model(**input_p, labels=output_p["input_ids"])

        res_loc_orig = F.softmax(res_loc_orig.logits, dim=2)
        res_loc_intervene = F.softmax(res_loc_intervene.logits, dim=2)

        # for editing loss
        res_p = F.softmax(res_p.logits, dim=2)
        target_alt = output_p["input_ids"]

        # locality loss: KL divergence
        L_loc =  einops.rearrange(F.kl_div(res_loc_intervene, 
                                           res_loc_orig,
                                           reduction='none'), "i j k -> (i j) k").sum(dim=1).mean(dim=0)

        # editing loss with nearby sample: nll
        # ignore padding tokens
        loss_sec = res_p.gather(2, target_alt.unsqueeze(-1))[~(target_alt == self.tokenizer.pad_token_id)]
        L_e = -loss_sec.log().sum()

        L = L_loc*c_loc + L_e*c_edit

        return L


# from transformers import AutoModelForMaskedLM, AutoTokenizer
# from argparse import Namespace

# args = Namespace(
    # experiment="mend",
    # output="./models",
    # lr=1e-4,
    # batch_size=10,
    # dataset="./data/pararel.csv",
    # val_split=0.1,
    # hidden_size=1920,
    # wandb=False,
# )

# tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")
# model = AutoModelForMaskedLM.from_pretrained("FacebookAI/roberta-large")

# mender = MENDer(args, model, tokenizer)

# mender.push("roberta.encoder.layer.21.intermediate",
            # "roberta.encoder.layer.22.intermediate",
            # "roberta.encoder.layer.23.intermediate")

# mender.push("roberta.encoder.layer.21.output",
            # "roberta.encoder.layer.22.output",
            # "roberta.encoder.layer.23.output")

# mender.compile()
# mender.epoch()


