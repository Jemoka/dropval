"""
hybrep_parse.py
get and count the knowledge neurons for a relation in question
"""
"""
hybrep_probe.py
probe for the knowledge neurons expressed by a list of interesting words
"""

import torch
import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score
from paratrace.intervene import intervene
from paratrace.core import LanguageModel

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

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

class KNStage2:
    def __init__(self, args, model, tokenizer):
        self.accelerator = Accelerator(log_with="wandb")
        self.accelerator.init_trackers(
            project_name="dropval", 
            config=vars(args),
            init_kwargs={"wandb": {"entity": "jemoka",
                                   "mode": None if args.wandb else "disabled"}},
        )

        self.lm = LanguageModel(model=model, tokenizer=tokenizer).to(self.accelerator.device)
        self.__out_file = (str(Path(args.out_path) / f"hybrep_{args.concept}_intervene.json"),
                           str(Path(args.out_path) / f"hybrep_{args.concept}_baseline.json"))
        self.__in_file = Path(args.in_path) / "hybrep.json"

        self.df = pd.read_json(str(self.__in_file), orient="split")

    def __call__():
        df = self.dm
        lm = self.lm

        rdf = df[["target", "pattern"]].drop_duplicates()

        for indx, v in tqdm(rdf.iterrows(), total=len(rdf)):
            concept = v.target
            synset = v.pattern # assuming we currently only care about one pattern's different representations

            rel = df[(df.target == concept) # & (df.pattern == synset)
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

            # calculate representatinos
            # rpr = TSNE(2, random_state=12)
            # res_rpr = rpr.fit_transform(onehot_neurons.numpy())

            # import seaborn as sns
            # import matplotlib.pyplot as plt

            # sns.scatterplot(x=res_rpr[:,0], y=res_rpr[:,1])
            # plt.show()

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
            results.to_csv(self.__out_file[0], index=False)

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
            results.to_csv(self.__out_file[1], index=False)
