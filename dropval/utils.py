from transformers import AutoModelForMaskedLM, AutoTokenizer

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datetime import timedelta

from pathlib import Path

import json

L = get_logger("dropfree", log_level="DEBUG")
process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=16200))  # 3 hours

def get_accelerator(config):
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[process_group_kwargs])
    accelerator.init_trackers(
        project_name="dropval", 
        config=vars(config),
        init_kwargs={"wandb": {"entity": "jemoka",
                               "mode": None if config.wandb else "disabled",
                               "name": config.experiment}},
    )

    return accelerator

def load_base(config):
    base = config.base

    model = AutoModelForMaskedLM.from_pretrained(base)
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = model.eval()

    # detect what model type it is (including size), and load the correct config
    model_name = type(model).__name__

    if model_name == "RobertaForMaskedLM" and len(model.roberta.encoder.layer) == 24:
        config = "roberta.json"
    elif model_name == "BertForMaskedLM" and len(model.bert.encoder.layer) == 12:
        config = "bert.json"
    else:
        raise FileNotFoundError(f"Missing config for a {model_name} or for the size of the model.")

    with open(Path(__file__).parent / "config" / config, 'r') as df:
        model_specific_config = json.load(df)

    return model, tokenizer, model_specific_config
