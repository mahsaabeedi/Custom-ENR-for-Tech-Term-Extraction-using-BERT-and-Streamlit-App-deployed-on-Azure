# -*- coding: utf-8 -*-

import torch
from torch.utils.data import TensorDataset,DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer 


BATCH_SIZE = 8
MODEL_CHECKPOINT = "bert-base-cased"


# this function:
# 1. create label_map
# 2. retokenizion by Bert
# 3. align lengths of tags and tokens after retokenization (BERT retokenizes the tokens we have, causing that the length of the tokens increased but the length of tags still remain the same. So, we will increase the length of tags to match the new token length.)  
# 4. transform format for tokens and tags from list to TensorDataset, then Dataloader

def preprocess_data_for_bert(texts,tags,label_all_tokens, train):
   #create label_map
    unique_labels = ['O','I','B']
    label_map = {}
    for (i, label) in enumerate(unique_labels):
        label_map[label] = i

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT) # load fast bert tokenizer
    tokenized_inputs = tokenizer(texts, padding = 'max_length',truncation=True, return_tensors = 'pt' ,is_split_into_words=True) #max_length = 30,

    labels = []
    labels_int = []
    for i, label in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        label_ids_int = [] 
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
                label_ids_int.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
                label_ids_int.append(label_map[label[word_idx]]) #get the integer label through mapping

            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
                label_ids_int.append(label_map[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx

        assert(len(tokenized_inputs['input_ids'][i]) == len(label_ids))
        assert(len(tokenized_inputs['input_ids'][i]) == len(label_ids_int))
       
        labels.append(label_ids)
        labels_int.append(label_ids_int)

    # Convert the lists into PyTorch tensors.
    # `input_ids` is a list of tensor arrays--stack them into a matrix with size: [num_examples  x  example_length].
    input_ids = torch.stack(tuple(tokenized_inputs['input_ids']), dim=0)
    attention_masks = torch.stack(tuple(tokenized_inputs['attention_mask']), dim=0)
    labels_int = torch.tensor(tuple(labels_int), dtype=torch.long)
    # Combine them into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels_int)
    # Create the DataLoader. We'll take training samples in random order. 
    if train is True:
        dataloader = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = BATCH_SIZE)
    else:
        dataloader = DataLoader(dataset, sampler = SequentialSampler(dataset), batch_size = BATCH_SIZE)
    
    return dataloader, tokenizer
