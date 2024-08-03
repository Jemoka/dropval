import torch, transformers, pyreft
from transformers import AutoModelForMaskedLM, AutoTokenizer

from datasets import Dataset
from torch.utils.data import DataLoader


def prepare_training_data(x, y, model, tokenizer, num_layers, intervene_tokens=1):
    all_base_input_ids, all_intervention_locations, all_output_ids = [], [], []

    for i, j in zip(x, y):
        x_enc = tokenizer(i, max_length=512,
                        return_tensors="pt", truncation=True)
        y_enc = tokenizer(j, max_length=512,
                        return_tensors="pt", truncation=True)["input_ids"]

        y_enc[x_enc["input_ids"] == tokenizer.pad_token_id] = -100

        intervention_locations = pyreft.get_intervention_locations(
            last_position = x_enc["input_ids"].size(1),
            first_n = intervene_tokens, 
            last_n = intervene_tokens,
            pad_mode = "last",
            num_interventions = num_layers,
            share_weights = False,
        )

        all_base_input_ids.append(x_enc["input_ids"][0])
        all_output_ids.append(y_enc[0])
        all_intervention_locations.append(intervention_locations)

    dataset = Dataset.from_dict({
        "input_ids": all_base_input_ids,
        "intervention_locations": all_intervention_locations,
        "labels": all_output_ids,
    })

    data_collator_fn = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest"
    )
    data_collator_fn = pyreft.ReftDataCollator(data_collator_fn)

    return dataset, data_collator_fn

