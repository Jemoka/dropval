"""
dataset.py
Data Collator, adatped from https://huggingface.co/learn/nlp-course/en/chapter7/7#training-loop
"""

import evaluate
metric = evaluate.load("squad_v2")

import collections
from collections import defaultdict
import numpy as np

import torch.nn.functional as F
import torch

def preprocess_training_examples(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs["offset_mapping"]
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    example_ids = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        sequence_ids = inputs.sequence_ids(i)
        example_ids.append(examples["id"][sample_idx])

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
            context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
            context_end = idx - 1

        if len(answer["answer_start"]) == 0:
            # if no answer, label is (0,0)
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                    start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                    end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs["id"] = example_ids
    return inputs

def hydrate_preprocessor(tokenizer):
    def hydrated(x):
        return preprocess_training_examples(x, tokenizer)
    return hydrated

def compute_metrics(start_logits, end_logits, features, examples):
    n_best = 20
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["id"]].append(idx)

    predicted_answers = []
    for example in examples:
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            try:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
            except:
                continue
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                            end_index < start_index
                            or end_index - start_index + 1 > 512
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

                    # Append the zero score
                    answer = {
                        "text": "",
                        "logit_score": start_logit[0] + end_logit[0],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            res = {"id": example_id, "prediction_text": best_answer["text"]}
        else:
            res = {"id": example_id, "prediction_text": ""}

        # if the span was 0,0, this means no answer was predicted
        res["no_answer_probability"] = int(res["prediction_text"] == "")
        predicted_answers.append(res)

    theoretical_answers = [{ "id": ex["id"], "answers": ex["answers"]} for ex in examples]

    return {
        f"squad/val/{k}": v
        for k,v in
        metric.compute(predictions=predicted_answers, references=theoretical_answers).items()
    }

