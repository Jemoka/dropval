import torch, transformers, pyreft
from transformers import AutoModelForMaskedLM, AutoTokenizer

from datasets import Dataset
from torch.utils.data import DataLoader

from dropval.trainers.utils.reft import prepare_training_data

# we have to load in bfloat16 because.... Pyvene and zen and aryaman says so
model = AutoModelForMaskedLM.from_pretrained("./pretrain/dropout", torch_dtype=torch.bfloat16)
# this is because otherwise ReFT will winge about it because they
# assume I'm not MLMing, which means autoregression would therefore
# be a thing
del model.config.__dict__["use_cache"]
tokenizer = AutoTokenizer.from_pretrained("./pretrain/dropout")
model = model.train()
RANK = 4
INTERVENE_TOKENS = 1
TARGET_LAYERS = [4, 8, 12, 16, 18, 22]

reft_config = pyreft.ReftConfig(
    representations = [
        {
            "layer": l, "component": f"roberta.encoder.layer[{l}].output",
            "low_rank_dimension": RANK,
            "intervention": pyreft.LoreftIntervention(
                embed_dim=model.config.hidden_size,
                low_rank_dimension=RANK
            )
        }
        for l in TARGET_LAYERS
    ]
)

dataset = [
    {
        "x": "I am a <mask>.",
        "y": "I am a dog."
    },
    {
        "x": "I am also also a <mask>.",
        "y": "I am also also a dog."
    }
]

x = ["I am also also a <mask>.", "I am a <mask>."]
y = ["I am also also a dog.", "I am a dog."]

data, collator = prepare_training_data(x, y, model, tokenizer, len(TARGET_LAYERS), INTERVENE_TOKENS)
train_dl = DataLoader(data, collate_fn=collator, batch_size=2)


intervenable = pyreft.get_reft_model(model, reft_config)
step = next(iter(train_dl))

unit_locations = None
if "intervention_locations" in step:
    unit_locations={
        "sources->base": (
            None,
            step["intervention_locations"].permute(1, 0, 2).tolist()
        )
    }

_, cf_outputs = intervenable(
    {
        "input_ids": step["input_ids"],
        "attention_mask": step["attention_mask"]
    },
    unit_locations=unit_locations,
    labels=step["labels"]
)

tokenizer.batch_decode(cf_outputs.logits.argmax(dim=-1))



unit_locations
step["input_ids"]

