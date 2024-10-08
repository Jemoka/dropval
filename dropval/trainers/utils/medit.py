import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import pickle
from tqdm import tqdm

from pathlib import Path

MASK = "[MASK]"

class ParatraceConceptSplitDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, x):
        item = self.df.iloc[x]
        return {
            "x": item.probe.replace("[MASK]", MASK),
            "y": item.probe.replace("[MASK]", item.target)
        }

    def __len__(self):
        return len(self.df)


def hydrate_bmask(url, p=0.1, mask=MASK):
    global MASK 

    # in case people change it
    MASK = mask

    df = pd.read_csv(url)
    targets = df.target.value_counts().index

    def hydrate_inner(target):
        subset = df[df.target == target]
        val = subset.sample(frac=p, random_state=7)
        train = subset[~subset.index.isin(val.index)]
        subset = df[df.target != target]
        incong = subset.sample(n=len(val), random_state=7)

        val = ParatraceConceptSplitDataset(val)
        train = ParatraceConceptSplitDataset(train)
        incong = ParatraceConceptSplitDataset(incong)

        return train, val, incong

    return hydrate_inner, targets.tolist()

# this is to mix in localization and target samples into one
# dataset
def hydrate_reft(url, p=0.1, mask=MASK):
    global MASK 

    # in case people change it
    MASK = mask

    df = pd.read_csv(url)
    targets = df.target.value_counts().index

    def hydrate_inner(target):
        subset = df[(df.target == target)]
        val = subset.sample(frac=p, random_state=7)
        train1 = subset[~subset.index.isin(val.index)]

        subset = df[(df.target != target)]
        incong = subset.sample(n=len(val), random_state=7)
        train2 = subset[~subset.index.isin(incong.index)].sample(n=len(train1),
                                                                 random_state=7)

        val = ParatraceConceptSplitDataset(val)
        train = ParatraceConceptSplitDataset(pd.concat([train1, train2]).sample(frac=1, random_state=7))
        incong = ParatraceConceptSplitDataset(incong)

        return train, val, incong

    return hydrate_inner, targets.tolist()


class ParatraceConsistencyDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        df = self.df
        elem = df.iloc[idx]
        xs = elem.probe
        try:
            xp = df[(df.target == elem.target) &
                    (df.pattern == elem.pattern) &
                    (df.source == elem.source) &
                    (df.instantiation != elem.instantiation)].sample(n=1, random_state=idx).iloc[0].probe
        except:
            return None
        ys = xs.replace("[MASK]", elem.target)
        yp = xp.replace("[MASK]", elem.target)

        loc = df[(df.pattern != elem.pattern)].sample(n=1, random_state=idx).iloc[0]
        xloc = loc.probe
        yloc = loc.probe.replace("[MASK]", loc.target)

        return {
            "xs": xs,
            "ys": ys,
            "xp": xp,
            "yp": yp,
            "xloc": xloc,
            "yloc": yloc,
        }

class DatasetPrefetchWrapper(Dataset):
    def __init__(self, ds):
        self.data = [ds[i] for i in tqdm(range(len(ds)))]
        self.data = [i for i in self.data if i != None]
        self.mask_ = MASK
    def __len__(self):
        return len(self.data)
    def __getitem__(self, x):
        dat = self.data[x]

        return {
            "xs": dat["xs"].replace("[MASK]", self.mask_),
            "xp": dat["xp"].replace("[MASK]", self.mask_),
            "xloc": dat["xloc"].replace("[MASK]", self.mask_),
            "ys": dat["ys"],
            "yp": dat["yp"],
            "yloc": dat["yloc"],
        }

def hydrate_mend(url, p=0.1, mask=MASK):
    global MASK 
    MASK = mask

    url_train = Path(url.replace("csv", f"{p}.train"))
    url_val = Path(url.replace("csv", f"{p}.val"))


    if url_train.exists() and url_val.exists():
        with open(url_val, 'rb') as d:
            ds_val = pickle.load(d)
        with open(url_train, 'rb') as d:
            ds = pickle.load(d)
        ds.mask_ = mask
        ds_val.mask_ = mask

        return ds, ds_val

    df = pd.read_csv(url)
    targets = df.target.value_counts()
    val_targets = np.random.choice(targets.index,
                                   size=(int(p*len(targets)),),
                                   replace=False,
                                   p=(targets / sum(targets)))
    df_val = df[df.target.isin(val_targets)]
    df = df[~df.target.isin(val_targets)]

    ds_val = DatasetPrefetchWrapper(ParatraceConsistencyDataset(df_val))
    with open(url_val, 'wb') as d:
        pickle.dump(ds_val, d)
    ds = DatasetPrefetchWrapper(ParatraceConsistencyDataset(df))
    with open(url_train, 'wb') as d:
        pickle.dump(ds, d)

    ds.mask_ = mask
    ds_val.mask_ = mask

    return ds, ds_val
