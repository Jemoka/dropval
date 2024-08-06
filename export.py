from pathlib import Path
import pandas as pd
from glob import glob
import json

from dropexp import analyze_kns

DROPOUT = "./output/dropout"
DROPFREE = "./output/no_dropout"

dropout = Path(DROPOUT)
dropfree = Path(DROPFREE)
assert dropout.exists(), "the dropout data folder to analyze doesn't exist!"
assert dropfree.exists(), "the dropfree data folder to analyze doesn't exist!"

final = {}

# for kns, we want to see if randomly shuffled clusters have equal amounts
# of success in terms of intervention
if (dropout / "kns").exists() and (dropfree / "kns").exists():
    final["kn"] = analyze_kns(dropout, dropfree)

# if (dropout / "bmask").exists() and (dropfree / "bmask").exists():

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


df_activations
do_activations

