import json
import os
import fire
from loguru import logger
import pandas as pd
import torch

from generation.add_demographic_bias import add_demographic_bias, load_demographics
from generation.generate_hf import (
    load_model_and_tokenizer,
    generate_responses,
)
from generation.datahandling import prep_data, add_responses

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger.add("logs/log.txt", rotation="1 MB", level="INFO", backtrace=True, diagnose=True)

MODEL_NAME_DICT = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B-Instruct": "Llama-3.1-70B",
    "google/gemma-3-12b-it": "Gemma-3-12B",
    "google/gemma-3-27b-it": "Gemma-3-27B",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen-2.5-14B",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen-2.5-72B",
}

# Set the device to GPU if available, otherwise use CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_responses(
    dem_bias_version: str,
    model_name: str,
    eval_set: str = None,
    response_result_dir="responses",
    num_return_sequences: int = 3,
    batch: int = None,
):

    with open("hf_token.txt", "r") as f:
        hf_token = f.read().strip()

    demographics = load_demographics()
    logger.info("Loaded demographics.")

    with open(f"data/evaluation_data.json", "r") as f:
        data = json.load(f)

    data = add_demographic_bias(
        data=data, demographics=demographics, dem_bias_version=dem_bias_version
    )

    logger.info(f"Added demographic bias to the data ({dem_bias_version}).")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name, hf_token=hf_token
    )
    logger.info(f"Loaded model and tokenizer ({model_name}).")

    for eval_set_name, data_eval_set in data.items():

        if eval_set is not None and eval_set != eval_set_name:
            continue

        else:

            if eval_set_name == "ib":
                max_new_tokens = 256
            else:
                max_new_tokens = 10
            if batch is not None:
                for key, value in data_eval_set.items():
                    data_eval_set[key] = value[
                        batch * len(value) // 3 : (batch + 1) * len(value) // 3
                    ]
                logger.info(f"Now evaluating: {eval_set_name}, batch {batch}")

            else:
                logger.info(f"Now evaluating: {eval_set_name}")
            data_loader = prep_data(
                data=data_eval_set, tokenizer=tokenizer, batch_size=8
            )

            generated_responses = generate_responses(
                data_loader=data_loader,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                device=DEVICE,
            )
            logger.info(
                f"Generated {len(generated_responses)} responses for {eval_set} set."
            )
            # add responses
            data_eval_set = add_responses(
                data=data_eval_set,
                responses=generated_responses,
                num_return_sequences=num_return_sequences,
            )

            # save to json
            if batch is not None:
                filename = (
                    f"{MODEL_NAME_DICT[model_name]}_{eval_set}_{dem_bias_version}_{batch}.json"
                )
            else:
                filename = (
                    f"{MODEL_NAME_DICT[model_name]}_{eval_set}_{dem_bias_version}.json"
                )
            with open(f"{response_result_dir}/{filename}", "w") as f:
                json.dump(data_eval_set, f, indent=4)


if __name__ == "__main__":
    # for version in [
    #     'name-system',
    #     'name-user',
    #     'explicit-mention-system',
    #     'explicit-mention-user',
    #     'writing-style-llm',
    #     'writing-style-human'
    # ]:
    #     get_responses(model_name='meta-llama/Llama-3.1-8B-Instruct', dem_bias_version=version)
    fire.Fire(get_responses)
