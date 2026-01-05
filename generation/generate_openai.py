import os
from typing import List

import backoff
from openai import OpenAI
import openai
from tqdm import tqdm


class OpenAILLM:

    def __init__(self, model: str):

        self.model_source = model
        self.model = OpenAI()

    def generate(self, prompts: List[str], **kwargs):

        # to save generated responses
        responses = []

        for prompt in tqdm(prompts):

            # Generate response
            output = self.completions_with_backoff(
                model=self.model_source, messages=prompt, **kwargs
            )
            response = [x.message.content for x in output.choices]

            # Save responses
            responses.extend(response)

        return responses

    # make sure no error occurs when passing rate limits, simply wait
    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def completions_with_backoff(self, **kwargs):
        return self.model.chat.completions.create(**kwargs)
