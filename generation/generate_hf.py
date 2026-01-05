from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig




def load_model_and_tokenizer(model_name: str, hf_token: str) -> tuple:
    """Load the model and tokenizer from Hugging Face.

    Args:
        model_name (str): Name of the model to load.
        hf_token (str): Hugging Face token for authentication.

    Returns:
        tuple: Loaded model and tokenizer.
    """
    # Load tokenizer with left padding
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=hf_token)

    # Set valid pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load in 8-bit quantization configuration
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model and automatically map to GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        quantization_config=bnb_config,
    )

    return model, tokenizer


def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data_loader: DataLoader,
    max_new_tokens: int = 256,
    num_return_sequences: int = 3,
    temperature: float = 1.0,
    device: str = "cuda",
) -> np.ndarray:
    """Generate responses using the model.

    Args:
        model (AutoModelForCausalLM): The model to use for generation.
        tokenizer (AutoTokenizer): Tokenizer to use for encoding.
        data_loader (DataLoader): DataLoader containing the input data.
        max_new_tokens (int): Maximum number of tokens to generate.
        num_return_sequences (int): Number of paraphrases to return per input.
        temperature (float): Sampling temperature for generation.
        device (str): Device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        np.ndarray: Array of generated paraphrases.
    """
    # Process inputs in batches
    all_responses = []
    for batch in tqdm(data_loader):
        # Send inputs to the specified device
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            # Generate outputs from the model
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=num_return_sequences,
            )

        # Remove input from output, decode, and add to results
        input_length = batch["input_ids"].shape[-1]
        generated_ids = outputs[:, input_length:]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_responses.extend(responses)

    return all_responses
