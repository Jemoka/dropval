from inter_rep_l2 import analyze_rep_l2
from unrelated_rep_l2 import analyze_unrelated_l2

import numpy as np
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

df_reps = []
df_unrs = []

for i in [0, 5000, 10000, 15000, 20000, 25000, 30000]:
    df_rep = analyze_rep_l2(f"./models/dropout/checkpoint_{i}.pt", f"./models/no_dropout/checkpoint_{i}.pt")
    df_unr = analyze_unrelated_l2(f"./models/dropout/checkpoint_{i}.pt", f"./models/no_dropout/checkpoint_{i}.pt")

    df_rep["checkpoint"] = i
    df_unr["checkpoint"] = i

    df_reps.append(df_rep)
    df_unrs.append(df_unr)

df_reps_concat = pd.concat(df_reps)
df_unrs_concat = pd.concat(df_unrs)

dat = []

for i in df_reps_concat[["model","layer","checkpoint"]].value_counts().index:
    itl = df_reps_concat[(df_reps_concat.model == i[0]) &
                         (df_reps_concat.layer == i[1]) &
                         (df_reps_concat.checkpoint == i[2])].intra_term_l2
    a,b = mean_confidence_interval(itl)
    dat.append({
        "model": i[0].replace(" ", "_"),
        "layer": i[1],
        "checkpoint": i[2],
        "intra_term_l2": a,
        "intra_term_l2_min": a-b,
        "intra_term_l2_max": a+b,
        "unrelated_term_l2": -1.0,
        "unrelated_term_l2_min": -1.0,
        "unrelated_term_l2_max": -1.0,
    })

dat = pd.DataFrame(dat)

for i in df_unrs_concat[["model","layer","checkpoint"]].value_counts().index:
    itl = df_unrs_concat[(df_unrs_concat.model == i[0]) &
                         (df_unrs_concat.layer == i[1]) &
                         (df_unrs_concat.checkpoint == i[2])].inter_term_l2
    a,b = mean_confidence_interval(itl)

    index = ((dat.model == i[0].replace(" ", "_")) &
             (dat.layer == i[1]) &
             (dat.checkpoint == i[2]))

    dat.loc[index, "unrelated_term_l2"] = a
    dat.loc[index, "unrelated_term_l2_min"] = a-b
    dat.loc[index, "unrelated_term_l2_max"] = a+b

dat.to_csv("dropmini.dat", sep=" ", index=False)


