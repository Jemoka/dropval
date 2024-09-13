from inter_rep_l2 import analyze_rep_l2
from unrelated_rep_l2 import analyze_unrelated_l2

import numpy as np
import scipy.stats

import pandas as pd


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

df_reps = []
df_unrs = []

for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    df_rep = analyze_rep_l2(f"./models/dropout_{i}/best.pt", f"./models/no_dropout/best.pt")
    df_unr = analyze_unrelated_l2(f"./models/dropout_{i}/best.pt", f"./models/no_dropout/best.pt")

    df_rep["do_pct"] = i
    df_unr["do_pct"] = i

    df_reps.append(df_rep)
    df_unrs.append(df_unr)

df_reps_concat = pd.concat(df_reps)
df_unrs_concat = pd.concat(df_unrs)

dat = []

for i in df_reps_concat[["model","layer","do_pct"]].value_counts().index:
    itl = df_reps_concat[(df_reps_concat.model == i[0]) &
                         (df_reps_concat.layer == i[1]) &
                         (df_reps_concat.do_pct == i[2])].intra_term_l2
    a,b = mean_confidence_interval(itl)
    dat.append({
        "model": i[0].replace(" ", "_"),
        "layer": i[1],
        "do_pct": i[2],
        "intra_term_l2": a,
        "intra_term_l2_min": a-b,
        "intra_term_l2_max": a+b,
        "unrelated_term_l2": -1.0,
        "unrelated_term_l2_min": -1.0,
        "unrelated_term_l2_max": -1.0,
    })

dat = pd.DataFrame(dat)

for i in df_unrs_concat[["model","layer","do_pct"]].value_counts().index:
    itl = df_unrs_concat[(df_unrs_concat.model == i[0]) &
                         (df_unrs_concat.layer == i[1]) &
                         (df_unrs_concat.do_pct == i[2])].inter_term_l2
    a,b = mean_confidence_interval(itl)

    index = ((dat.model == i[0].replace(" ", "_")) &
             (dat.layer == i[1]) &
             (dat.do_pct == i[2]))

    dat.loc[index, "unrelated_term_l2"] = a
    dat.loc[index, "unrelated_term_l2_min"] = a-b
    dat.loc[index, "unrelated_term_l2_max"] = a+b

dat.sort_values(by="do_pct")[(dat.model=="dropout") & (dat.layer==1)].to_csv("./output/dropmini_dropout_l1_pcts.dat", sep=" ", index=False)
dat.sort_values(by="do_pct")[(dat.model=="dropout") & (dat.layer==2)].to_csv("./output/dropmini_dropout_l2_pcts.dat", sep=" ", index=False)

