#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 07:44:19 2021

@author: olesyar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 12:15:38 2021

@author: olesyar
"""
# Libraries
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel
from transformers import BertTokenizer
from common_classes import BertClassifier
from common_classes import text_preprocessing
from common_classes import preprocessing_for_bert
from numpy import argmax
from common_classes import get_class
import csv
import torch.nn as nn
import logging


if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
 
# modify the model path to load the model    
model=torch.load('/Users/olesyar/Documents/data/bert_trials')

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


logging.basicConfig(level=logging.ERROR)
names_all = ["nct_id","phase","name","mesh_term","why_stopped","pmid"]
names_studies = ['nct_id','nlm_download_date_description',
                 'study_first_submitted_date','results_first_submitted_date','disposition_first_submitted_date',
                 'last_update_submitted_date','study_first_submitted_qc_date','study_first_posted_date',
                 'study_first_posted_date_type','results_first_submitted_qc_date','results_first_posted_date',
                 'results_first_posted_date_type','disposition_first_submitted_qc_date',
                 'disposition_first_posted_date','disposition_first_posted_date_type',
                 'last_update_submitted_qc_date','last_update_posted_date','last_update_posted_date_type',
                 'start_month_year','start_date_type','start_date','verification_month_year',
                 'verification_date','completion_month_year','completion_date_type','completion_date',
                 'primary_completion_month_year','primary_completion_date_type','primary_completion_date',
                 'target_duration','study_type','acronym','baseline_population','brief_title','official_title',
                 'overall_status','last_known_status','phase','enrollment','enrollment_type','source',
                 'limitations_and_caveats','number_of_arms','number_of_groups','why_stopped','has_expanded_access',
                 'expanded_access_type_individual','expanded_access_type_intermediate',
                 'expanded_access_type_treatment','has_dmc','is_fda_regulated_drug','is_fda_regulated_device',
                 'is_unapproved_device','is_ppsd','is_us_export','biospec_retention','biospec_description',
                 'ipd_time_frame','ipd_access_criteria','ipd_url','plan_to_share_ipd','plan_to_share_ipd_description',
                 'created_at','updated_at']

def create_predictions(df):
    # Display 5 samples from the test data
    print('The test set is loaded')
    # Run `preprocessing_for_bert` on the test set
    test_inputs, test_masks = preprocessing_for_bert(df[df["why_stopped"].notnull()].why_stopped)
    # Create the DataLoader for our test set
    test_dataset = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)
    print('The test set is ready')
    return test_dataloader

# =============================================================================
# make predictions
# =============================================================================

if __name__ == '__main__':
    LARGE_FILE = "/Users/olesyar/Documents/data/studies.txt"
    reader = pd.read_csv(LARGE_FILE, skiprows=1, names=names_studies, delimiter='|')
    # reader2 = pd.read_csv(LARGE_FILE2, skiprows=1, names=names_all, delimiter='\t')

    # print(len(reader))
    # # print(len(reader2))
    reader=(reader[['why_stopped','phase','nct_id', 'start_date']]).drop_duplicates()
    # reader2=(reader2[['why_stopped','phase','nct_id', 'start_date']]).drop_duplicates()
    print(len(reader))
    # print(len(reader2))
    probs = BertClassifier(model, create_predictions(reader))
    csv_file1=open('/Users/olesyar/Documents/data/stopped_predictions2.tsv', "w")
    writer1 = csv.writer(csv_file1, delimiter='\t', lineterminator='\n')
    i=0
    not_stopped=reader[reader["why_stopped"].notnull()]
    for ind,dat in not_stopped.iterrows():
        writer1.writerow([dat['why_stopped'].replace('\r~', ''),dat['phase'],dat['nct_id'],dat['start_date'],get_class(argmax(probs[i]))])
        i=i+1
    csv_file2=open('/Users/olesyar/Documents/data/notstopped_predictions2.tsv', "w")
    writer2 = csv.writer(csv_file2, delimiter='\t')
    i=0
    stopped=reader[reader["why_stopped"].isnull()]
    for ind,dat in stopped.iterrows():
        writer2.writerow([dat['why_stopped'],dat['phase'],dat['nct_id'],dat['start_date'],''])
    
    
    
    
    
    
    
# probs = bert_predict(model, create_predictions('/Users/olesyar/Documents/data/studies.tsv'))
# preds=[]
# for pred in probs:
#     print(get_class(argmax(pred)))
# newData = pd.read_csv("/Users/olesyar/Documents/data/studies2.tsv",  skiprows=2,
#                           names=names_all, delimiter='\t')
# csv_file=open('/Users/olesyar/Documents/data/stopped_predictions.tsv', "w")
# writer = csv.writer(csv_file, delimiter='\t')

# newData=newData.drop_duplicates(subset='nct_id', keep="first")
# print('Tokenizing data...')
# test_inputs, test_masks = preprocessing_for_bert(newData["why_stopped"])
# test_dataset = TensorDataset(test_inputs, test_masks)
# test_sampler = SequentialSampler(test_dataset)
# test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)
# probs = bert_predict(model, test_dataloader)
# for data in probs:
#     print(get_class(argmax(data)))
# print(probs)



