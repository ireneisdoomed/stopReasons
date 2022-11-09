#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 15:52:01 2022

@author: lesya
"""
from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):
    """
        Bert Model for classification Tasks.
    """
    def __init__(self, freeze_bert=False, n_class=17):
        """
        @param   bert: a BertModel object
        @param   classifier: a torch.nn.Module classifier
        @param   freeze_bert (bool): Set `False` to fine_tune the Bert model
        @param   n_class (integer): number of classes to feed the classifier
        """
        super(BertClassifier,self).__init__()
        # Specify hidden size of Bert, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, n_class
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        self.classifier = nn.Sequential(
                            nn.Linear(D_in, H),
                            nn.ReLU(),
                            nn.Linear(H, D_out))
        self.sigmoid = nn.Sigmoid()
        # Freeze the Bert Model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self,input_ids,attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        outputs = self.bert(input_ids=input_ids,
                           attention_mask = attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:,0,:]
        
        # Feed input to classifier to compute logits
        return self.classifier(last_hidden_state_cls)
    
