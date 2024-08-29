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

    l2_layer1 = []
    l2_layer2 = []

    for i in range(len(vecs_layer1)):
        for j in range(i+1, len(vecs_layer1)):
            l2_layer1.append(((vecs_layer1[i] - vecs_layer1[j])**2).sum().sqrt())

    for i in range(len(vecs_layer2)):
        for j in range(i+1, len(vecs_layer2)):
            l2_layer2.append(((vecs_layer2[i] - vecs_layer2[j])**2).sum().sqrt())

    return sum(l2_layer1)/len(l2_layer1), sum(l2_layer2)/len(l2_layer2)

pre_proj_dropout = []
pre_proj_no_dropout = []

def analyze_unrelated_l2(dropout_checkpoint="./models/dropout.pt", no_dropout_checkpoint="./models/no_dropout.pt"):
    global pre_proj_dropout, pre_proj_no_dropout

    checkpoint = torch.load(dropout_checkpoint)
    gen = checkpoint.generator

    dropout_model = checkpoint.model
    no_dropout_model = torch.load(no_dropout_checkpoint).model

    dropout_distances = []
    no_dropout_distances = []

    prefix = random.sample(gen.iid_tokens, 3)
    suffix = random.sample(gen.iid_tokens, 3)

    flattened_seq = sum(gen.sequences, [])
    seqs_grouped = defaultdict(list)
    for i in flattened_seq:
        seqs_grouped[i[1]].append(i)
    seqs_grouped = dict(seqs_grouped)

    # we only want one of each two pairs of values because
    # they are repeated permulations, which is tested by
    # inter_rep_l2.py
    unrelated_seqs = [[j for indx, j in enumerate(i) if indx % 2 == 1]
                    for i in seqs_grouped.values()]

    for seq_base in tqdm(unrelated_seqs):
        if len(seq_base) < 2:
            continue
        # we need to register seperate hooks to write them to different arrays
        def inspect_hook_proj_dropout(module, input, output):
            global pre_proj_dropout
            pre_proj_dropout.append(input[0])
            return output
        remove_hooks_dropout = [i.linear2.register_forward_hook(inspect_hook_proj_dropout)
                                for i in dropout_model.encoder.layers]
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
    df.columns = ["inter_term_l2", "model", "layer"]

    return df

if __name__ == "__main__":
    df = analyze_unrelated_l2("./models/dropout/best.pt", "./models/no_dropout/best.pt")
    sns.boxplot(df, x="layer", y="inter_term_l2", hue="model")
