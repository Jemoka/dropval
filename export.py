from pathlib import Path
import pandas as pd
from glob import glob
import json
from collections import defaultdict

from dropexp import analyze_kns, analyze_bmask, analyze_reft, analyze_ft
from dropexp.utils import mean_confidence_interval, ks

from scipy.stats import ttest_rel

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='export',
                    description='What the program does')
    parser.add_argument('output', type=str)
    parser.add_argument('--dropout', type=str, required=True)
    parser.add_argument('--no_dropout', type=str, required=True)
    args = parser.parse_args()

    DROPOUT = args.dropout
    DROPFREE = args.no_dropout
    OUTPUT = args.output

    dropout = Path(DROPOUT) / "results"
    dropfree = Path(DROPFREE) / "results"
    assert dropout.exists(), "the dropout data folder to analyze doesn't exist!"
    assert dropfree.exists(), "the dropfree data folder to analyze doesn't exist!"

    final = {}

    # for kns, we want to see if randomly shuffled clusters have equal amounts
    # of success in terms of intervention
    if (dropout / "kns").exists() and (dropfree / "kns").exists():
        final["kn"] = analyze_kns(dropout, dropfree)

    if (dropout / "bmask").exists() and (dropfree / "bmask").exists():
        final["bmask"] = analyze_bmask(dropout, dropfree)

    if (dropout / "reft").exists() and (dropfree / "reft").exists():
        final["reft"] = analyze_reft(dropout, dropfree)

    if (dropout / "ft").exists() and (dropfree / "ft").exists():
        final["ft"] = analyze_ft(dropout, dropfree)

    if (dropout / "squad.json").exists() and (dropfree / "squad.json").exists():
        with open(dropfree / "squad.json", 'r') as d:
            df = json.load(d)
        df = {i.split("/")[-1]: j for i,j in df.items()}
        with open(dropout / "squad.json", 'r') as d:
            do = json.load(d)
        do = {i.split("/")[-1]: j for i,j in do.items()}

        final["squad"] = {
            i+"_value": {
                "dropout": do[i],
                "no_dropout": df[i]
            }
            for i in df.keys()
        }

    if (dropout / "mend.json").exists() and (dropfree / "mend.json").exists():
        with open(dropfree / "mend.json", 'r') as d:
            df = json.load(d)
        df = {i.split("/")[-1]: j for i,j in df.items()}
        with open(dropout / "mend.json", 'r') as d:
            do = json.load(d)
        do = {i.split("/")[-1]: j for i,j in do.items()}

        final["mend"] = {
            "editing": {
                "target_successes_value": {
                    "dropout": do["edit_success"],
                    "no_dropout": df["edit_success"],
                },
                "unrelated_sucesses_value": {
                    "dropout": do["edit_success_localization"],
                    "no_dropout": df["edit_success_localization"],
                }
            },
            "masked_value_editing": {
                "target_successes_value": {
                    "dropout": do["mask_edit_success"],
                    "no_dropout": df["mask_edit_success"],
                },
                "unrelated_sucesses_value": {
                    "dropout": do["mask_edit_success_localization"],
                    "no_dropout": df["mask_edit_success_localization"],
                }
            },
            "target_probability_change": {
                "dropout": do["target_prob_diff"],
                "no_dropout": df["target_prob_diff"],
            }
        }

    if (dropout / "consistency.csv").exists() and (dropfree / "consistency.csv").exists():
        df = pd.read_csv(str(dropout/"consistency.csv"))
        df.pred_tokens = df.pred_tokens.apply(lambda x:x.replace("Ġ", "").strip())
        repr_do = df.groupby(["target", "pattern"]).pred_tokens.unique().apply(lambda x:len(x))
        do_consistency = mean_confidence_interval(repr_do)

        df = pd.read_csv(str(dropfree/"consistency.csv"))
        df.pred_tokens = df.pred_tokens.apply(lambda x:str(x).replace("Ġ", "").strip())
        repr_df = df.groupby(["target", "pattern"]).pred_tokens.unique().apply(lambda x:len(x))
        df_consistency = mean_confidence_interval(repr_df)

        final["consistency"] = {
            "num_representations_p95": {
                "dropout": do_consistency,
                "no_dropout": df_consistency
            },
            "num_representations_df_minus_do_pairedt": {
                "statistic": ttest_rel(repr_df, repr_do).statistic,
                "pval": ttest_rel(repr_df, repr_do).pvalue,
            }
        }

    with open(OUTPUT, 'w') as df:
        json.dump(final, df, indent=4)


