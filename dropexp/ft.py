from pathlib import Path
import pandas as pd
from glob import glob
import json
from collections import defaultdict

from dropexp import analyze_kns, analyze_bmask
from dropexp.utils import mean_confidence_interval, ks

from scipy.stats import ttest_rel

def analyze_ft(dropout, dropfree):
    dropout_concepts = set([Path(i).stem for i in glob(str(dropout / "ft" / "ft*json"))])
    dropfree_concepts = set([Path(i).stem for i in glob(str(dropfree / "ft" / "ft*json"))])

    grouped_concepts = list(dropout_concepts.intersection(dropfree_concepts))

    do_values = defaultdict(list)
    for item in sorted(glob(str(dropout / "ft" / "ft*json"))):
        if not any([i in item for i in grouped_concepts]):
            continue
        with open(item, 'r') as df:
            data = json.load(df)
        for k,v in data.items():
            do_values[k].append(v)
    do_values_final = {}
    for k,v in do_values.items():
        do_values_final[k.split("/")[-1].strip()] = mean_confidence_interval(v)

    df_values = defaultdict(list)
    for item in sorted(glob(str(dropfree / "ft" / "ft*json"))):
        if not any([i in item for i in grouped_concepts]):
            continue
        with open(item, 'r') as df:
            data = json.load(df)
        for k,v in data.items():
            df_values[k].append(v)
    df_values = dict(df_values)
    df_values_final = {}
    for k,v in df_values.items():
        df_values_final[k.split("/")[-1].strip()] = mean_confidence_interval(v)


    paired_final = {}
    for k,v in df_values.items():
        df_values_final[k.split("/")[-1].strip()] = mean_confidence_interval(v)

        v_do = do_values[k]
        v_df = v
        paired_final[k.split("/")[-1].strip()] = {
            "statistic": ttest_rel(v_df, v_do).statistic,
            "pval": ttest_rel(v_df, v_do).pvalue
        }

    return {
        "editing": {
            "target_successes_p95": {
                "dropout": do_values_final["edit_success"],
                "no_dropout": df_values_final["edit_success"],
            },
            "target_successes_df_minus_do_pairedt": paired_final["edit_success"],
            "unrelated_sucesses_p95": {
                "dropout": do_values_final["edit_localization"],
                "no_dropout": df_values_final["edit_localization"],
            },
            "unrelated_sucesses_df_minus_do_pairedt": paired_final["edit_localization"],
        },
        "masked_value_editing": {
            "target_successes_p95": {
                "dropout": do_values_final["mask_edit_success"],
                "no_dropout": df_values_final["mask_edit_success"],
            },
            "target_successes_df_minus_do_pairedt": paired_final["mask_edit_success"],
            "unrelated_sucesses_p95": {
                "dropout": do_values_final["mask_edit_localization"],
                "no_dropout": df_values_final["mask_edit_localization"],
            },
            "unrelated_sucesses_df_minus_do_pairedt": paired_final["mask_edit_localization"],
        }
    }

