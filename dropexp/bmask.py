from pathlib import Path
import pandas as pd
from glob import glob
import json

from dropexp import analyze_kns
from dropexp.utils import mean_confidence_interval, ks

from scipy.stats import ttest_rel

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
    names_do = [Path(i).stem for i in bmasks]

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
    names_ndo = [Path(i).stem for i in bmasks]

    unioned = set(names_do).intersection(set(names_ndo))
    do_indicies = [names_do.index(i) for i in list(unioned)]
    ndo_indicies = [names_ndo.index(i) for i in list(unioned)]

    do_activations = [do_activations[i] for i in do_indicies]
    do_edit_successes = [do_edit_successes[i] for i in do_indicies]
    do_edit_localizations = [do_edit_localizations[i] for i in do_indicies]

    df_activations = [df_activations[i] for i in ndo_indicies]
    df_edit_successes = [df_edit_successes[i] for i in ndo_indicies]
    df_edit_localizations = [df_edit_localizations[i] for i in ndo_indicies]

    df_values = mean_confidence_interval(df_activations)
    do_values = mean_confidence_interval(do_activations)

    df_edit_successes = mean_confidence_interval(df_edit_successes)
    do_edit_successes = mean_confidence_interval(do_edit_successes)

    df_edit_localizations = mean_confidence_interval(df_edit_localizations)
    do_edit_localizations = mean_confidence_interval(do_edit_localizations)

    return {
        "localization": {
            "activations_p95": {
                "dropout": do_values,
                "no_dropout": df_values
            },
            "activations_df_minus_do_pairedt": {
                "statistic": ttest_rel(df_activations, do_activations).statistic,
                "pval": ttest_rel(df_activations, do_activations).pvalue,
            }
        },
        "editing": {
            "target_successes_p95": {
                "dropout": do_edit_successes,
                "no_dropout": df_edit_successes
            },
            "target_successes_df_minus_do_pairedt": {
                "statistic": ttest_rel(df_edit_successes, do_edit_successes).statistic,
                "pval": ttest_rel(df_edit_successes, do_edit_successes).pvalue
            },
            "unrelated_sucesses_p95": {
                "dropout": do_edit_localizations,
                "no_dropout": df_edit_localizations
            },
            "unrelated_sucesses_df_minus_do_pairedt": {
                "statistic": ttest_rel(df_edit_localizations, do_edit_localizations).statistic,
                "pval": ttest_rel(df_edit_localizations, do_edit_localizations).pvalue
            }

        }
    }
