
import datasets
from .utils import Concatenator

PROMPT = f"{{sys_prompt}} Response:{{response}}"


def get_preprocessed_financial(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("Trelis/function_calling", split=split)

    def apply_prompt_template(sample):
        return {
            "text": PROMPT.format(sys_prompt=sample["systemPrompt"],
                                  response=sample["assistantResponse"])
        }

    dataset = dataset.map(apply_prompt_template)

    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    )
    dataset = dataset.map(Concatenator(chunk_size=2048), batched=True)
    return dataset
