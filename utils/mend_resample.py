import pickle
from random import Random
from data import DatasetPrefetchWrapper

# original split
op = 0.1
# desired new split
p = 0.05
R = Random(0)

with open(f"./data/paratrace.{op}.train", 'rb') as df:
    train = pickle.load(df)
    
with open(f"./data/paratrace.{op}.val", 'rb') as df:
    val = pickle.load(df)
    
conceptmap = defaultdict(list)
for i in val.data:
    front, back = i["xs"].split("[MASK]")
    concept = i["ys"].replace(front, "", 1)[::-1].replace(back[::-1], "", 1)[::-1]
    conceptmap[concept].append(i)
    
for i in train.data:
    front, back = i["xs"].split("[MASK]")
    concept = i["ys"].replace(front, "", 1)[::-1].replace(back[::-1], "", 1)[::-1]
    conceptmap[concept].append(i)
    
conceptmap = dict(conceptmap)
concepts = conceptmap.keys()
val_concepts = R.sample(list(concepts), int(p*len(concepts)))

train_final = [i for a,b in conceptmap.items() for i in b if a not in val_concepts]
val_final = [i for a,b in conceptmap.items() for i in b if a in val_concepts]

train_ds = DatasetPrefetchWrapper(train_final)
val_ds = DatasetPrefetchWrapper(val_final)
   
with open(f"./data/paratrace.{p}.train", 'wb') as df:
    pickle.dump(train_ds, df)
    
with open(f"./data/paratrace.{p}.val", 'wb') as df:
    pickle.dump(val_ds, df)

