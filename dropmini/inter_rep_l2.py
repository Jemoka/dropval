import torch
import random
from synthetic import *
from copy import deepcopy
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

import seaborn as sns
sns.set()

def mean_l2_distance(vecs):
    vecs_layer1 = [i[0].detach() for i in vecs]
    vecs_layer2 = [i[1].detach() for i in vecs]

    l2_layer1 = ((vecs_layer1[0] - vecs_layer1[1])**2).sum().sqrt()
    l2_layer2 = ((vecs_layer2[0] - vecs_layer2[1])**2).sum().sqrt()

    return l2_layer1, l2_layer2

checkpoint = torch.load("./models/dropout.pt")
gen = checkpoint.generator

dropout_model = checkpoint.model
no_dropout_model = torch.load("./models/no_dropout.pt").model

dropout_distances = []
no_dropout_distances = []

prefix = random.sample(gen.iid_tokens, 3)
suffix = random.sample(gen.iid_tokens, 3)

for seq_base in tqdm(gen.sequences):
    # we need to register seperate hooks to write them to different arrays
    pre_proj_dropout = []
    def inspect_hook_proj_dropout(module, input, output):
        global pre_proj_dropout
        pre_proj_dropout.append(input[0])
        return output
    remove_hooks_dropout = [i.linear2.register_forward_hook(inspect_hook_proj_dropout)
                            for i in dropout_model.encoder.layers]
    pre_proj_no_dropout = []
    def inspect_hook_proj_no_dropout(module, input, output):
        global pre_proj_no_dropout
        pre_proj_no_dropout.append(input[0])
        return output
    remove_hooks_no_dropout = [i.linear2.register_forward_hook(inspect_hook_proj_no_dropout)
                            for i in no_dropout_model.encoder.layers]

    collected_activations_dropout = defaultdict(list)
    collected_activations_no_dropout = defaultdict(list)

    # we want to test different preturbations of this seq_base
    for seq in seq_base:
        pre_proj_dropout = []
        pre_proj_no_dropout = []

        seq = list(seq)
        seq = torch.tensor([gen.enc_token] + prefix + seq + suffix + [gen.enc_token]).unsqueeze(0).cuda()

        # and this will also be the token whose embedding we care about
        label_token = seq[0][5].cpu().detach().item()
        seq[0][5] = gen.mask_token

        res_dropout = dropout_model(seq, ~torch.ones_like(seq).bool())
        res_no_dropout = no_dropout_model(seq, ~torch.ones_like(seq).bool())

        dropout_correct = (res_dropout.logits.argmax(dim=-1)[0][5] == label_token).float()
        no_dropout_correct = (res_no_dropout.logits.argmax(dim=-1)[0][5] == label_token).float()

        # dropout_emb= 
        do_l1, do_l2 = pre_proj_dropout
        dropout_embs = (do_l1.squeeze(1)[5], do_l2.squeeze(1)[5])
        df_l1, df_l2 = pre_proj_no_dropout
        no_dropout_embs = (df_l1.squeeze(1)[5], df_l2.squeeze(1)[5])

        collected_activations_dropout[label_token].append(dropout_embs)
        collected_activations_no_dropout[label_token].append(no_dropout_embs)

    [i.remove() for i in remove_hooks_dropout]
    [i.remove() for i in remove_hooks_no_dropout]

    collected_activations_dropout = dict(collected_activations_dropout)
    collected_activations_no_dropout = dict(collected_activations_no_dropout)

    collected_activations_dropout.keys()
    collected_activations_no_dropout.keys()

    mean_distance_dropout = [mean_l2_distance(i) for i in collected_activations_dropout.values()]
    mean_distance_no_dropout = [mean_l2_distance(i) for i in collected_activations_no_dropout.values()]

    dropout_distances.append(mean_distance_dropout)
    no_dropout_distances.append(mean_distance_no_dropout)

dropout_distances_layer1 = torch.stack([j[0] for i in dropout_distances for j in i])
dropout_distances_layer2 = torch.stack([j[1] for i in dropout_distances for j in i])

no_dropout_distances_layer1 = torch.stack([j[0] for i in no_dropout_distances for j in i])
no_dropout_distances_layer2 = torch.stack([j[1] for i in no_dropout_distances for j in i])

dropout_distances_layer1.mean()
dropout_distances_layer2.mean()

no_dropout_distances_layer1.mean()
no_dropout_distances_layer2.mean()

df_dropout_layer1 = pd.DataFrame(dropout_distances_layer1.cpu())
df_dropout_layer1["model"] = "dropout"
df_no_dropout_layer1 = pd.DataFrame(no_dropout_distances_layer1.cpu())
df_no_dropout_layer1["model"] = "no dropout"
df_layer1 = pd.concat([df_dropout_layer1, df_no_dropout_layer1])

df_dropout_layer2 = pd.DataFrame(dropout_distances_layer2.cpu())
df_dropout_layer2["model"] = "dropout"
df_no_dropout_layer2 = pd.DataFrame(no_dropout_distances_layer2.cpu())
df_no_dropout_layer2["model"] = "no dropout"
df_layer2 = pd.concat([df_dropout_layer2, df_no_dropout_layer2])

df_layer1["layer"] = 1
df_layer2["layer"] = 2

df = pd.concat([df_layer1, df_layer2])
df.columns = ["intra-term l2", "model", "layer"]

sns.boxplot(df, x="layer", y="intra-term l2", hue="model")


