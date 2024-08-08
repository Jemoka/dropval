from pathlib import Path
import pandas as pd
from glob import glob
import json
from collections import defaultdict

from dropexp import analyze_kns, analyze_bmask
from dropexp.utils import mean_confidence_interval, ks

def analyze_reft(dropout, dropfree):
    do_values = defaultdict(list)
    for item in glob(str(dropout / "reft" / "reft*json")):
        with open(item, 'r') as df:
            data = json.load(df)
        for k,v in data.items():
            do_values[k].append(v)
    do_values_final = {}
    for k,v in do_values.items():
        do_values_final[k.split("/")[-1].strip()] = mean_confidence_interval(v)

    df_values = defaultdict(list)
    for item in glob(str(dropfree / "reft" / "reft*json")):
        with open(item, 'r') as df:
            data = json.load(df)
        for k,v in data.items():
            df_values[k].append(v)
    df_values = dict(df_values)
    df_values_final = {}
    for k,v in df_values.items():
        df_values_final[k.split("/")[-1].strip()] = mean_confidence_interval(v)

    return {
        "editing": {
            "target_successes_p95": {
                "dropout": do_values_final["edit_success"],
                "no_dropout": df_values_final["edit_success"],
            },
            "unrelated_sucesses_p95": {
                "dropout": do_values_final["edit_localization"],
                "no_dropout": df_values_final["edit_localization"],
            }
        },
        "masked_value_editing": {
            "target_successes_p95": {
                "dropout": do_values_final["mask_edit_success"],
                "no_dropout": df_values_final["mask_edit_success"],
            },
            "unrelated_sucesses_p95": {
                "dropout": do_values_final["mask_edit_localization"],
                "no_dropout": df_values_final["mask_edit_localization"],
            }
        }
    }

