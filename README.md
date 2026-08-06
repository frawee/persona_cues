# One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization

This is the code for the paper ['One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization'](https://aclanthology.org/2026.acl-long.2079/).

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

## Citation
If you use the code in this repository, please cite the following paper:
```bibtex
@inproceedings{weeber-etal-2026-one,
    title = "One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact {LLM} Personalization",
    author = "Weeber, Franziska  and
      Neplenbroek, Vera  and
      Batzner, Jan  and
      Pad{\'o}, Sebastian",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.2079/",
    doi = "10.18653/v1/2026.acl-long.2079",
    pages = "44892--44921",
    ISBN = "979-8-89176-390-6",
    abstract = "Personalization of LLMs by sociodemographic subgroup often improves user experience, but can also introduce or amplify biases and unfairoutcomes across groups. Prior work has employed so-called personas, sociodemographic user attributes conveyed to a model, to studybias in LLMs by relying on a single cue to prompt a persona, such as user names or explicit attribute mentions. This disregards LLM sensitivity to prompt variation and the rarity of some cues in real interactions (external validity). We compare six commonly used personacues across seven open and proprietary LLMs on four writing and advice tasks. While cues are overall highly correlated, they produce sub-stantial variance in responses across personas that can change findings on persona-induced differences and bias. We therefore cautionagainst claims based on single persona cues, especially when they are overly explicit and have low external validity."
}
```
