from pathlib import Path
import pandas as pd
from glob import glob
import json
from collections import defaultdict

from dropexp import analyze_kns, analyze_bmask, analyze_reft
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

    dropout = Path(DROPOUT)
    dropfree = Path(DROPFREE)
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

    if (dropout / "consistency.csv").exists():
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
            "paired_increase_in_consistency_pairedt": {
                "statistic": ttest_rel(repr_df, repr_do).statistic,
                "pval": ttest_rel(repr_df, repr_do).pvalue,
            }
        }

        with open(OUTPUT, 'w') as df:
            json.dump(final, df, indent=4)


