from pathlib import Path
import pandas as pd
from glob import glob
import json

from dropexp import analyze_kns
from dropexp.utils import mean_confidence_interval, ks

def analyze_bmask(dropout, dropfree):
    do_activations = []
    do_edit_successes = []
    do_edit_localizations = []

    bmasks = glob(str(dropout / "bmask" / "*.json"))
    for bmask in bmasks:
        with open(bmask, 'r') as df:
            data = json.load(df)
            do_activations.append(data["mean_activations"])
            do_edit_successes.append(data["bmask_val"][1])
            do_edit_localizations.append(data["bmask_val"][2])


    df_activations = []
    df_edit_successes = []
    df_edit_localizations = []

    bmasks = glob(str(dropfree / "bmask" / "*.json"))
    for bmask in bmasks:
        with open(bmask, 'r') as df:
            data = json.load(df)
            df_activations.append(data["mean_activations"])
            df_edit_successes.append(data["bmask_val"][1])
            df_edit_localizations.append(data["bmask_val"][2])

    df_values = mean_confidence_interval(df_activations)
    do_values = mean_confidence_interval(do_activations)

    df_edit_successes = mean_confidence_interval(df_edit_successes)
    do_edit_successes = mean_confidence_interval(do_edit_successes)

    df_edit_localizations = mean_confidence_interval(df_edit_localizations)
    do_edit_localizations = mean_confidence_interval(df_edit_localizations)

    return {
        "localization": {
            "activations_p95": {
                "dropout": do_values,
                "no_dropout": df_values
            }
        },
        "editing": {
            "target_successes_p95": {
                "dropout": do_edit_successes,
                "no_dropout": df_edit_successes
            },
            "unrelated_sucesses_p95": {
                "dropout": do_edit_localizations,
                "no_dropout": df_edit_localizations
            }
        }
    }
