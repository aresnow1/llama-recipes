import copy

import datasets
import torch

PROMPT = f"{{sys_prompt}} Response:{{response}}"


def get_preprocessed_function_call(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("marclove/llama_functions", split="train")
    if split == "test":
        dataset = dataset.train_test_split(test_size=0.1)[split]

    def convert(sample):
        max_words = 4096
        full = tokenizer.encode(sample["input"] + sample["output"])
        prompt = torch.tensor(
            tokenizer.encode(sample["input"]), dtype=torch.int64
        )
        full = torch.tensor(full + [tokenizer.eos_token_id], dtype=torch.int64)
        padding = max_words - full.shape[0]
        if padding > 0:
            full = torch.cat((full, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            full = full[: max_words]
        labels = copy.deepcopy(full)
        labels[: len(prompt)] = -1
        full_mask = full.ge(0)
        label_mask = labels.ge(0)
        full[~full_mask] = 0
        labels[~label_mask] = 0
        full_mask = full_mask.float()
        label_mask = label_mask.float()
        return {
            "input_ids": full,
            "labels": labels,
            "attention_mask": full_mask,
        }

    dataset = dataset.map(
        convert,
        remove_columns=list(dataset.features),
    )
    return dataset
