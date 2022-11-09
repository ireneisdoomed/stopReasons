#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 08:30:21 2021

@author: olesyar
"""
from typing import Tuple

import torch
from transformers import BertTokenizer

CLASS_TO_IDX = {
        'Another_Study': 0,
        'Business_Administrative': 1,
        'Covid19': 2,
        'Endpoint_Met': 3,
        'Ethical_Reason': 4,
        'Insufficient_Data': 5,
        'Insufficient_Enrollment': 6,
        'Interim_Analysis': 7,
        'Invalid_Reason': 8,
        'Logistics_Resources': 9,
        'Negative': 10,
        'No_Context': 11,
        'Regulatory': 12,
        'Safety_Sideeffects': 13,
        'Study_Design': 14,
        'Study_Staff_Moved': 15,
        'Success': 16,
    }

CLASS_TO_SUPER = {
        'Business_Administrative': "Possibly_Negative",
        'Another_Study': "Neutral",
        'Negative': "Negative",
        'Study_Design': "Neutral",
        'Invalid_Reason': "Invalid_Reason",
        'Ethical_Reason': "Neutral",
        'Insufficient_Data': "Neutral",
        'Insufficient_Enrollment': "Neutral",
        'Study_Staff_Moved': "Neutral",
        'Endpoint_Met': "Neutral",
        'Regulatory': "Neutral",
        'Logistics_Resources': "Neutral",
        'Safety_Sideeffects': "Safety_Sideeffects",
        'No_Context': "Invalid_Reason",
        'Success': "Success",
        'Interim_Analysis': "Neutral",
        'Covid19': "Neutral",
    }


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def text_preprocessing(text: str) -> str:
    """
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    return text.strip()


# Create a function to BERT tokenize a set of texts
def preprocessing_for_bert(text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform required preprocessing steps for pretrained BERT.
    @param    text (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    MAX_LEN = 177

    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []
    # For every sentence...
    for sent in text:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,  # Max length to truncate/pad
            pad_to_max_length=True,  # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,  # Return attention mask
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return input_ids, attention_masks


def bert_predict(model, dataloader):
    """
    Perform a forward pass on the trained BERT model to predict probabilities.
    """

    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time. The predictions also do not need these functions.
    # This is to make the processing quicker
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)

    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # sigmoid shows higher accuracy on multi class classification tasks
    return all_logits.sigmoid().cpu().numpy()

