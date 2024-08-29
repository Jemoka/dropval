from random import randint, sample, shuffle, choice
from itertools import permutations
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy
from tqdm import tqdm

from types import SimpleNamespace
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

class ToyGenerator:
    def __init__(self,
                 seq_size=3, # size of a sequence, like <paris, captial, france>
                 seq_pool=30, # pool of possible sequences
                 seq_tokens=20, # number of tokens to use in the sequences
                 iid_tokens=5, # number of toknes to use in the non-sequence sets, "stopwords"
                 max_seqs_per_ut=20,
                 max_length=64): # maximum number of sequences to insert into an ut

        assert max_length >= seq_size+1, "max length shorter than sequence size"
        assert max_seqs_per_ut <= seq_pool, "seq pool must be larger than maximum number of seqs"

        self._vocab_size = seq_tokens+iid_tokens
        self.seq_size = seq_size
        self.seq_pool = seq_pool
        self.seq_tokens = list(range(1,seq_tokens+1))
        self.iid_tokens = list(range(seq_tokens+1,self._vocab_size+1))
        self.max_seqs_per_ut = max_seqs_per_ut
        self.max_length = max_length

        self.populate()

    def populate(self):
        self._seqs = []

        while len(self._seqs) < self.seq_pool:
            seq = set(sample(self.seq_tokens, self.seq_size))
            while len(seq) != self.seq_size:
                # because the sequence probably had duplicated tokens
                seq = set(sample(self.seq_tokens, self.seq_size))
            if seq not in self._seqs:
                self._seqs.append(seq)

        self.seqs = [list(permutations(i)) for i in self._seqs]

    @property
    def vocab_size(self):
        # because padding token 0 and mask token self._vocab_size+1
        return self._vocab_size + 3
    @property
    def padding_token(self):
        return 0
    @property
    def mask_token(self):
        return self._vocab_size+1
    @property
    def enc_token(self):
        return self._vocab_size+2

    @property
    def sequences(self):
        return self.seqs

    def __call__(self):
        num_seqs = randint(1, self.max_seqs_per_ut)
        seqs = [list(choice(i)) for i in sample(self.seqs, num_seqs)]
        shuffle(seqs)
        seq_tokens = sum(seqs, [])

        seq_length = randint(len(seq_tokens)+1, self.max_length-2)
        missing = seq_length - len(seq_tokens)

        # double array because we will sum() it with seqs eventually
        seq = [[choice(self.iid_tokens)] for _ in range(missing)] + seqs
        shuffle(seq)
        seq = sum(seq, [])

        return [self.enc_token]+seq+[self.enc_token]

    def stringify(self, ut):
        seq_stack = []
        res = []
        for i in ut:
            if len(seq_stack) == self.seq_size:
                res.append("["+"-".join(seq_stack)+"]")
                seq_stack = []

            if i in self.seq_tokens:
                seq_stack.append(str(i))
            else:
                res.append(str(i))

        if len(seq_stack) == self.seq_size:
            res.append("["+"-".join(seq_stack)+"]")
            seq_stack = []

        return " ".join(res)

class ToyDataset(Dataset):
    def __init__(self, N, generator, mask_pct=0.15):
        super().__init__()

        self.gen = generator
        self.data = [self.gen() for _ in tqdm(range(N))]
        self.mask_pct = mask_pct
        self.mask_token = generator.mask_token

    def __getitem__(self, indx):
        dat = self.data[indx]
        x = deepcopy(dat)

        for i in sample(range(len(x)), int(len(x)*self.mask_pct)):
            x[i] = self.mask_token

        return {
            "x": x,
            "y": dat
        }

    def __len__(self):
        return len(self.data)

class ToyModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, upproj_size=3072,
                 nhead=12, num_layers=2, dropout_pct=0.1, padding_idx=0,
                 max_length=64):
        super().__init__()

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)

        enc = nn.TransformerEncoderLayer(hidden_size, nhead,
                                         dim_feedforward=upproj_size, 
                                         dropout=dropout_pct)
        self.encoder = nn.TransformerEncoder(enc, num_layers)

        self.ln = nn.LayerNorm(hidden_size, eps=1e-12)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, attention_mask, labels=None):
        # x: batch, seq, hidden
        embs = self.word_embeddings(inputs)
        pos = self.position_embeddings(
            torch.arange(0, inputs.size(1)).to(self.out.weight.device)
        ).repeat(embs.size(0), 1, 1)

        res = self.encoder((embs + pos).permute(1,0,2),
                           src_key_padding_mask=attention_mask).permute(1,0,2)

        logits = self.decoder(self.ln(F.gelu(self.out(res))))

        loss = None
        if labels != None:
            loss = self.criterion(logits.permute(0,2,1), labels)

        return SimpleNamespace(logits=logits, loss=loss)

def collate(dat):
    pad_size = max([len(j) for j in sum([list(i.values()) for i in dat], [])])

    result = defaultdict(list)
    for i in dat:
        for k,v in i.items():
            result[k].append(torch.tensor(v+[0 for _ in range(pad_size-len(v))]))

    result = { k: torch.stack(v) for k,v in result.items() }

    labels = result["y"]
    labels[labels == 0] = -100

    attention_mask = (labels == -100)
    
    return {
        "inputs": result["x"],
        "attention_mask": attention_mask,
        "labels": labels
    }


class Trainer:
    def __init__(self, save_dir, epochs, batch_size, dropout_pct=0.1, lr=1e-4, generator_args={}, device="cuda"):
        self.device = device
        self.save_dir = save_dir

        gen = ToyGenerator(**generator_args)
        self.model = ToyModel(gen.vocab_size,
                              dropout_pct=dropout_pct).to(self.device)

        train = ToyDataset(500000, gen)
        self.train_dl = DataLoader(train, batch_size=batch_size, collate_fn=collate)

        dev = ToyDataset(4096, gen)
        self.dev_dl = DataLoader(dev, batch_size=batch_size, collate_fn=collate)

        self.generator = gen

        self.epochs = epochs

        self.best_acc_ = 0

        self.optim = Adam(self.model.parameters(), lr=lr)

    def val(self):
        seq_tokens = torch.tensor(self.generator.seq_tokens).to(self.device)

        seq_accs = []
        masked_seq_accs = []
        overall_accs = []

        for indx, batch in enumerate(iter(self.dev_dl)):
            if indx % 64 == 0:
                print(f"VAL | {indx}/{len(self.dev_dl)-1}")
            batch = {i: j.to(self.device) for i,j in batch.items()}
            output = self.model(**batch)

            preds = output.logits.argmax(dim=-1)
            targets = batch["labels"]
            is_seq = torch.isin(targets, seq_tokens) 
            is_masked_seq = (is_seq) & (batch["inputs"] == self.generator.mask_token)

            # seq accuracy
            seq_acc = preds[is_seq] == targets[is_seq] 

            # masked seq accuracy
            masked_seq_acc = preds[is_masked_seq] == targets[is_masked_seq] 

            # overall
            overall_acc = preds[targets != -100] == targets[targets != -100]

            if len(seq_acc) > 0:
                seq_accs.append((seq_acc.sum()/len(seq_acc)).cpu().item())
            if len(masked_seq_acc) > 0:
                masked_seq_accs.append((masked_seq_acc.sum()/len(masked_seq_acc)).cpu().item())
            if len(overall_acc) > 0:
                overall_accs.append((overall_acc.sum()/len(overall_acc)).cpu().item())

        seq_acc = sum(seq_accs)/len(seq_accs)
        masked_seq_acc = sum(masked_seq_accs)/len(masked_seq_accs)
        overall_acc = sum(overall_accs)/len(overall_accs)

        print(f"VAL | SEQ: {round(seq_acc, 3)} | MASKED: {round(masked_seq_acc, 3)} | OVERALL: {round(overall_acc, 3)}")

        return seq_acc, masked_seq_acc, overall_acc

    def train(self):
        for i in range(self.epochs):
            print(f"EPOCH | {i}/{self.epochs-1}")
            for indx, batch in enumerate(iter(self.train_dl)):
                batch = {i: j.to(self.device) for i,j in batch.items()}
                output = self.model(**batch)

                output.loss.backward()
                self.optim.step()
                self.optim.zero_grad()

                if indx % 24 == 0:
                    print(f"TRAIN | {indx}/{len(self.train_dl)-1} | LOSS {round(output.loss.item(), 3)}")

                if indx % 1024 == 0:
                    a,b,c = self.val()
                    if b > self.best_acc_:
                        print("BEST MODEL!")
                        self.best_acc_ = b
                        torch.save(self, self.save_dir)

                        
