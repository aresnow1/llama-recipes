# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/JanosAudran/financial-reports-sec

import datasets
from .utils import Concatenator


def get_preprocessed_financial(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("JanosAudran/financial-reports-sec", "large_full", split=split)

    def apply_prompt_template(sample):
        return {
            "text": sample
        }

    dataset = dataset.map(apply_prompt_template, input_columns=["sentence"])

    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    )
    dataset = dataset.map(Concatenator(chunk_size=1024), batched=True)
    return dataset
