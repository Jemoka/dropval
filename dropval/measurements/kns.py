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


import torch
import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score

import json
import torch
import random
import numpy as np
from glob import glob
from pathlib import Path
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from tqdm import tqdm

import torch.nn.functional as F
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt



from contextlib import contextmanager

from accelerate.logging import get_logger
L = get_logger("dropval", log_level="DEBUG")

class KN:
    def __init__(self, args, accelerator, model, tokenizer):
        self.df = pd.read_csv(args.dataset)

        self.accelerator = accelerator

        self.lm = LanguageModel(model=model, tokenizer=tokenizer).to(self.accelerator.device)

        out_path = args.out_dir / args.results_dir / "kns"
        out_path.mkdir(parents=True, exist_ok=True)

        self.__out_file_s1 = out_path / "kns.json"
        self.__out_file_s2 = (str(out_path / f"kns_CONCEPT_intervene.json"),
                              str(out_path / f"kns_CONCEPT_baseline.json"))

    def __call__(self):
        df = self.df
        
        if self.__out_file_s1.exists():
            res = pd.read_json(self.__out_file_s1, orient="split")
        else:
            res = pd.DataFrame(columns=df.columns.tolist()+["knowledge"])

        rdf = df[["target", "pattern"]].drop_duplicates()

        for a, v in rdf.iterrows():
            i = v.target

            L.info(f"probing {i}; keyword ({a}/{len(rdf)})...")

            if (Path(self.__out_file_s2[0].replace(CONCEPT, i)).exists() and
                Path(self.__out_file_s2[1].replace(CONCEPT, i)).exists()):
                continue

            total = len(df[df.target == i])
            for indx, (_, row) in enumerate(df[df.target == i].iterrows()):
                if indx % 5 == 0:
                    res.to_json(str(self.__out_file_s1), index=False, orient="split", indent=2)
                    L.info(f"probing {i}; progress ({indx}/{total})...")
                    knowledge = self.probe(row.probe, row.target, self.lm)
                    row["knowledge"] = knowledge.cpu().tolist()
                    res.loc[len(res)] = row
                    res.to_json(str(self.__out_file_s1), index=False, orient="split", indent=2)

            self.stage2(res, i)

    def stage2_(self, df, concept):
        lm = self.lm

        rel = df[(df.target == concept)
                & ~(df.pred_tokens.isin([".", ";",]))]
        rel_wrong = rel[rel.target != rel.pred_tokens]
        rel_right = rel[rel.target == rel.pred_tokens]

        # serialize each knoweldge neuron with an ID
        neuron_id = defaultdict(lambda:len(neuron_id))
        serialized_neurons = []
        for i in rel.knowledge:
            tmp = []
            for j in i:
                tmp.append(neuron_id[tuple(j)])
            serialized_neurons.append(tmp)
        neuron_id = {j:i for i,j in neuron_id.items()}

        # create one-hot mapping of each cluster
        onehot_neurons = []
        for i in serialized_neurons:
            z = torch.zeros((len(neuron_id,)))
            z[i] = 1
            onehot_neurons.append(z)
        onehot_neurons = torch.stack(onehot_neurons) # num samples x num neurons

        res_cls = None

        best_scores = -1
        best_cluster = None

        ohn = onehot_neurons.numpy()
        for clusters in range(2, 25):
            cls = KMeans(n_clusters=clusters, n_init="auto", random_state=12)
            try:
                score = silhouette_score(ohn, cls.fit_predict(ohn))
            except ValueError:
                break
            if best_scores < score: 
                best_scores = score
                best_cluster = clusters
                res_cls = cls.labels_

        if best_cluster == None:
            continue

        results = []

        for k1 in range(0, best_cluster):
            for k2 in range(k1, best_cluster):
                knowledge = rel[res_cls == k1].knowledge

                # for each knowledge neuron, identify its count
                kn_counts = defaultdict(int)
                for i in knowledge:
                    for j in i:
                        kn_counts[tuple(j)] += 1
                kn_counts = list(dict(kn_counts).items())
                kn_counts = sorted(kn_counts, key=lambda x:x[1], reverse=True)

                sample = rel[res_cls == k2]

                for indx, i in sample.iterrows():
                    probe = i.probe
                    target = i.target

                    res_suppress = intervene(probe, target, lm,
                                            [i[0] for i in kn_counts[:10]], mode=0)
                    res_augment = intervene(probe, target, lm,
                                            [i[0] for i in kn_counts[:10]], mode=3)

                    results.append({
                        "match": k1 == k2,
                        "was_correct": i.pred_tokens == i.target,
                        "suppress_success": res_suppress[1][0][0] != i.target,
                        "augment_success":res_augment[1][0][0] == i.target,
                        "knowledge_cluster": k1,
                        "intervene_cluster": k2,
                    })

        results = pd.DataFrame(results)
        results.to_csv(self.__out_file_s2[0].replace(CONCEPT, concept), index=False)

        np.random.shuffle(res_cls)

        results = []

        for k1 in range(0, best_cluster):
            for k2 in range(k1, best_cluster):
                knowledge = rel[res_cls == k1].knowledge

                # for each knowledge neuron, identify its count
            kn_counts = defaultdict(int)
                for i in knowledge:
                    for j in i:
                        kn_counts[tuple(j)] += 1
                kn_counts = list(dict(kn_counts).items())
                kn_counts = sorted(kn_counts, key=lambda x:x[1], reverse=True)

                sample = rel[res_cls == k2]

                for indx, i in sample.iterrows():
                    probe = i.probe
                    target = i.target

                    res_suppress = intervene(probe, target, lm,
                                            [i[0] for i in kn_counts[:10]], mode=0)
                    res_augment = intervene(probe, target, lm,
                                            [i[0] for i in kn_counts[:10]], mode=3)

                    results.append({
                        "match": k1 == k2,
                        "was_correct": i.pred_tokens == i.target,
                        "suppress_success": res_suppress[1][0][0] != i.target,
                        "augment_success":res_augment[1][0][0] == i.target,
                        "knowledge_cluster": k1,
                        "intervene_cluster": k2,
                    })

        results = pd.DataFrame(results)
        results.to_csv(self.__out_file_s2[1].replace(CONCEPT, concept), index=False)


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

    @staticmethod
    def intervene(probe, target, lm, kn, mode="suppress"):
        layers = list(lm.ffn_layers())

        attrs_by_layer = torch.tensor([]).to(lm.device)
        target_idx = lm.tokenizer.convert_tokens_to_ids(target)

        # example neuron to probe
        # layer, numbber

        # setup parameter guards, one for eac hlayer
        guards = []
        enter_order = list(set([i[0] for i in kn]))
        for layer in enter_order:
            guard = ParamGuard(*layers[layer], mode=mode, speed=True)
            guards.append(guard)
            guards[-1].__enter__()

        # to register baseline + get mask pos
        _, _, _, mask_loc = lm(probe,1)

        # ask the guards to suppress/double results
        for layer, guard in zip(enter_order, guards):
            probe_idx = torch.tensor([[mask_loc, i]
                                    for i in set([i[1] for i in kn
                                                    if i[0] == layer])], device=mask_loc.device)
            guard.interpolate(probe_idx)

        # probe new output
        res_post = lm(probe, 1, force_pred_idx=target_idx)

        # tear down guards
        while len(guards) > 0:
            guards.pop(-1).__exit__(None, None, None)

        # probe old output
        res_pre = lm(probe, 1, force_pred_idx=target_idx)
        return res_pre, res_post

