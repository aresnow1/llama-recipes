
import datasets
from .utils import Concatenator

PROMPT = f"{{sys_prompt}} Response:{{response}}"


def get_preprocessed_function_call(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("marclove/llama_functions", split="train")
    dataset = dataset.train_test_split(test_size=0.05)[split]

    dataset = dataset.map(
        lambda sample: tokenizer(sample["input"] + sample["output"] + tokenizer.eos_token),
        batched=True,
        remove_columns=list(dataset.features),
    )
    dataset = dataset.map(Concatenator(chunk_size=512), batched=True)
    return dataset
