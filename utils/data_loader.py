import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset

import random
import itertools

from functools import partial
from utils.generation import (
    load_hf_dataset,
    check_input_lengths,
    tokenize_for_generation
)

class C4ValidationIterableDataset(IterableDataset):
    def __init__(self, data_source, split='validation', valid_split=0.5, total_size=1000, seed=None):
        self.data_source = data_source
        self.valid_split = valid_split
        self.split = split
        self.total_size = total_size
        self.seed = seed

    def __iter__(self):
        if self.seed is not None:
            random.seed(self.seed)
        data_iter = iter(self.data_source)
        valid_size = int(self.total_size * self.valid_split)
        if self.split == 'validation':
            return itertools.islice(data_iter, 0, valid_size)
        else:
            return itertools.islice(data_iter, valid_size, self.total_size)

class TrainValTestIterableDataset(IterableDataset):
    def __init__(self, data_source, total_size=6400, seed=None):
        self.data_source = data_source
        self.total_size = total_size
        self.seed = seed

    def __iter__(self):
        if self.seed is not None:
            random.seed(self.seed)
        data_iter = iter(self.data_source)
        return itertools.islice(data_iter, 0, self.total_size)
        
class CustomIterableDataset(IterableDataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __iter__(self):
        for sample in self.hf_dataset:
            yield sample

def filter_dataset(args, tokenizer):
    dataset = load_hf_dataset(args)

    token_kwargs = dict(
        hf_model_name=args.model_name_or_path,
        tokenizer=tokenizer,
        args=args,
    )
    token_kwargs.update(dict(max_new_tokens=args.max_new_tokens))
    tokenize_prompts = partial(tokenize_for_generation, **token_kwargs)

    input_check_kwargs = dict(
        max_input_len=args.max_input_len, 
        max_new_tokens=args.max_new_tokens,
    )
    input_check_kwargs.update(dict(min_prompt_len=args.min_prompt_tokens, min_completion_len=args.max_new_tokens))
    input_check = partial(check_input_lengths, **input_check_kwargs)

    ###########################################################################
    # Compose the partials to create the pipeline
    ###########################################################################

    # tokenize and truncate the row inputs to create prompts according to the strategy spec'd above
    dataset_w_prompts = dataset.map(tokenize_prompts, batched=False)

    # filter the rows of the dataset based on length checks for the tokenized prompts and baseline completions
    dataset_input_len_filtered = dataset_w_prompts.filter(input_check, batched=False)
    return dataset_input_len_filtered


def data_loader(args, dataset_input_len_filtered, tokenizer, model_short_name):
    # construct the collator
    def data_collator(batch):
        # Pad input_ids
        if model_short_name == 'opt':
            input_ids_batch = [torch.tensor(item['input_ids'], dtype=torch.long).view(-1) for item in batch]
        else:
            input_ids_batch = []
            for item in batch:
                input_ids = tokenizer(item['truncated_input'], return_tensors="pt")["input_ids"]
                input_ids_batch.append(input_ids.view(-1))
        
        # Reverse the sequences, pad them, and then reverse back
        input_ids_reversed = [torch.flip(tensor, [0]) for tensor in input_ids_batch]  # Reverse each sequence
        input_ids_padded_reversed = pad_sequence(input_ids_reversed, batch_first=True, padding_value=tokenizer.pad_token_id)
        input_ids_padded = torch.flip(input_ids_padded_reversed, [1])  # Reverse back to original order

        # Collate other data fields dynamically
        collated_batch = {'input_ids': input_ids_padded}
        for key in batch[0].keys():
            if key != 'input_ids':  # Assuming 'input_ids' is handled separately
                collated_batch[key] = [item[key] for item in batch]

        return collated_batch
    # Create instances for train, validation, and test sets
    if args.dataset_split == 'train':
        train_dataset = CustomIterableDataset(TrainValTestIterableDataset(dataset_input_len_filtered, seed=0))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collator)
        loader = train_loader
    elif args.dataset_split == 'validation':
        if args.split == 'valid':
            cur_dataset = CustomIterableDataset(C4ValidationIterableDataset(dataset_input_len_filtered, split='validation', seed=0))
        elif args.split == 'test':
            cur_dataset = CustomIterableDataset(C4ValidationIterableDataset(dataset_input_len_filtered, split='test', seed=0))
        loader = DataLoader(cur_dataset, batch_size=args.batch_size, collate_fn=data_collator)
    else:
        print("C4 split doesn't exist.")
        return None

    return loader