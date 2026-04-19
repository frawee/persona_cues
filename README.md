# One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization

This is the code for the paper ['One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization'](https://arxiv.org/abs/2601.18572).

To replicate our findings, please follow these steps.

1. Setup
    - install the requirements
    - for open-source LLMs: make sure you have access to a GPU, ideally multiple
    - for ChatGPT: make sure you have access to a paid OpenAI accound and API key.
2. Preprocess the data
    - the names from the [North Carolina voter registration file](https://www.ncsbe.gov/results-data/voter-pregistration-data) (prep_names.py)
    - the evaluation tasks (prep_evaluation_data.ipynb)
    - the persona demographics (prep_data_demographics.ipynb)
3. Generate all responses 
    - for open-source LLMs: get_responses.py
    - for ChatGPT: get_responses_openai.py
4. Postprocess responses
    - if applicable: Merge all batches of one model/persona cue combination
    - apply postprocess_responses.py 
    - apply evaluation/eval_responses.py (this will also perform the stance detection for IB responses)
    - run the first few cells of visualization.py
5. Evaluate
    - run the remaining cells of visualization.py
