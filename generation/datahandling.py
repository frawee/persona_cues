from functools import partial
from typing import Dict, List, Union
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TDataset
from transformers import AutoTokenizer


class ChatPromptDataset(TDataset):
    """Custom dataset class for handling chat-format prompts."""

    def __init__(self, prompt: List[str]):
        """
        Initialize the dataset with a list of prompts.

        Args:
            prompts (List[str]): List of chat-format prompts.
        """
        self.prompt = prompt  # Store the list of prompts

    def __len__(self) -> int:
        """Return the number of prompts in the dataset."""
        return len(self.prompt)

    def __getitem__(self, idx: int) -> str:
        """Return a prompt at a specific index.

        Args:
            idx (int): Index of the prompt to retrieve.

        Returns:
            str: The prompt at the specified index.
        """
        return self.prompt[idx]


def chat_collate_fn(batch: List[str], tokenizer: AutoTokenizer) -> Dict:
    """Collate function to format batch inputs for the model.

    Args:
        batch (List[str]): List of prompts to collate.
        tokenizer (AutoTokenizer): Tokenizer to use for encoding.

    Returns:
        Dict: Encoded batch ready for model input.
    """
    encoding = tokenizer.apply_chat_template(
        batch,
        tokenize=True,  # Return numeric IDs, not text
        return_tensors="pt",  # Return as PyTorch tensors
        padding=True,  # Padding required when batching
        add_generation_prompt=True,
        return_dict=True,
    )
    return encoding


def prep_data(
    data: Dict[str, List[Union[str, int]]],
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
) -> DataLoader:
    """Prepare the data for the model by creating a DataLoader.

    Args:
        data (Dict[str, List[Union[str, int]]]): Dictionary containing the input data.
        tokenizer (AutoTokenizer): Tokenizer to use for encoding.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: DataLoader for the prepared dataset.
    """
    dataset = ChatPromptDataset(prompt=data["prompt"])
    collate_fn = partial(chat_collate_fn, tokenizer=tokenizer)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return data_loader


def add_responses(data, responses, num_return_sequences):

    target_length = len(responses)
    
    for key in data.keys():
        current_length = len(data[key])
        
        if current_length == target_length:
            pass
        elif current_length == target_length // num_return_sequences:
            data[key] =  [x for x in data[key] for _ in range(num_return_sequences)] 
        else:
            raise ValueError('Data length not compatible with number of return sequences.')
    
    data['responses'] = responses
    data['sample_number'] = [i % num_return_sequences for i in range(len(responses))]
     
    return data
