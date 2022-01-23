#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 08:30:21 2021

@author: olesyar
"""
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer
import torch
import torch.nn.functional as F


if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 17

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits
    
def text_preprocessing(text):
    """
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    return text.strip()

# Create a function to BERT tokenize a set of texts
def preprocessing_for_bert(text: str):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    MAX_LEN=64
    """Perform required preprocessing steps for pretrained BERT.
    @param    text (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
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
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
            )
        
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return input_ids, attention_masks

def bert_predict(model, dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
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

    # Apply softmax to calculate probabilities. For more than one class classification, 
    # sigmoid should be normally applied. The tests showed that for this task, sigmoid
    # reduces the accuracy quite substantially, while softmax produces more accurate results
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs

def get_class(code):
    """
    @param: the position of the class with the maximum probability.
    Return the mapped class name
    """
    class_dict = {
        'Business_Administrative':0, 
        'Another_Study':1, 
        'Negative':2,
        'Study_Design':3,
        'Invalid_Reason':4,
        'Ethical_Reason':5, 
        'Insufficient_Data':6,
        'Insufficient_Enrollment':7, 
        'Study_Staff_Moved':8, 
        'Endpoint_Met':9,
        'Regulatory':10, 
        'Logistics_Resources':11,
        'Safety_Sideeffects':12, 
        'No_Context':13, 
        'Success':14,
        'Interim_Analysis':15, 
        'Covid19':16
        }
    key_list = list(class_dict.keys())
    val_list = list(class_dict.values())
    position = val_list.index(code)
    return(key_list[position])

def class_map(name):
    """
    @param: the name of the predicted class.
    Return the mapped parent class: Neutral, Negative, Positive, Success, Invalid Reason, or Safety and Side Effects
    """
    main_reasons_dict = {
        'Business_Administrative':"Possibly_Negative", 
        'Another_Study':"Neutral", 
        'Negative':"Negative",
        'Study_Design':"Neutral",
        'Invalid_Reason':"Invalid_Reason",
        'Ethical_Reason':"Neutral", 
        'Insufficient_Data':"Neutral",
        'Insufficient_Enrollment':"Neutral", 
        'Study_Staff_Moved':"Neutral", 
        'Endpoint_Met':"Neutral",
        'Regulatory':"Neutral", 
        'Logistics_Resources':"Neutral",
        'Safety_Sideeffects':"Safety_Sideeffects", 
        'No_Context':"Invalid_Reason", 
        'Success':"Success",
        'Interim_Analysis':"Neutral", 
        'Covid19':"Neutral"
        }
    return(main_reasons_dict[name])