import json
import fire
from loguru import logger
import pandas as pd

from generation.add_demographic_bias import add_demographic_bias, load_demographics
from generation.datahandling import add_responses
from generation.generate_openai import OpenAILLM


def get_responses_openai(
    dem_bias_version: str,
    openai_key: str,
    model_name: str = "gpt-4o-mini",  # gpt-5-nano-2025-08-07 
    num_return_sequences: int = 3,
    max_new_tokens: int = 2000,
    response_result_dir: str = "responses",
):

    demographics = load_demographics()
    logger.info("Loaded demographics.")

    with open(f"data/evaluation_data.json", "r") as f:
        data = json.load(f)

    data = add_demographic_bias(
        data=data, demographics=demographics, dem_bias_version=dem_bias_version
    )

    logger.info(f"Added demographic bias to the data ({dem_bias_version}).")

    model = OpenAILLM(
        model=model_name,
    )
    logger.info(f"Loaded model and tokenizer ({model_name}).")

    for eval_set, data_eval_set in data.items():

        logger.info(f"Now evaluating: {eval_set}")

        generated_responses = model.generate(
            prompts=data_eval_set["prompt"],
            temperature=1.0,  
            max_tokens=max_new_tokens,
            n=num_return_sequences,
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

        # create df
        df_eval_data = pd.DataFrame(data_eval_set)
        # save to json
        filename = f"{model_name}_{eval_set}_{dem_bias_version}.json"
        df_eval_data.to_csv(f"{response_result_dir}/{filename}", index=False)


if __name__ == "__main__":

    fire.Fire(get_responses_openai)
