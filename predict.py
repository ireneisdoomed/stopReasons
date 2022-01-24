#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:23:43 2021

@author: olesyar
"""

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
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertModel
from transformers import BertTokenizer
from src.common_classes import BertClassifier
from src.common_classes import text_preprocessing
from src.common_classes import preprocessing_for_bert
from numpy import argmax
from src.common_classes import get_class
from src.common_classes import class_map
from src.common_classes import bert_predict
import csv
import torch.nn as nn
import logging
import sys


 
# modify the model path to load the model    
model=torch.load('/Users/irene/Desktop/PRs/stopReasons/data/bert_trials.pth')

logging.basicConfig(level=logging.ERROR)
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

def prepare_data(df):
    # Display 5 samples from the data
    print('The data set is loaded')
    # Run `preprocessing_for_bert` on the data set
    data_inputs, data_masks = preprocessing_for_bert(df[df["why_stopped"].notnull()].why_stopped)
    # Create the DataLoader for our prediction set
    dataset = TensorDataset(data_inputs, data_masks)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=5)
    print('The data set is ready')
    return dataloader

def get_parser():
    """Get parser object for script predict.py"""
    parser = argparse.ArgumentParser(
        description='This script categorises why a clinical trial has stopped into several classes.'
    )

    parser.add_argument(
        '--input_file',
        help='Input TSV file containing the reasons to stop per clinical trial.',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--output_file',
        help='Output TSV file containing the reasons to stop of a clinical trial with their corresponding 2 levels of categories.',
        type=str,
        required=True,
    )

    return parser

# =============================================================================
# make predictions
# =============================================================================
    
def main(input_file, output_file):

    # load the imput file studies.tsv, and extract the columns needed
    studies_file = input_file
    reader = pd.read_csv(studies_file, sep='\t')
    reader = reader[reader['why_stopped'].notna()]
    reader = (reader[['why_stopped', 'nct_id']]).drop_duplicates()

    # generate probabilities
    model_data_loader = prepare_data(reader)
    probs = bert_predict(model, model_data_loader)

    # export predictions
    csv_file1=open(output_file, "w")
    writer1 = csv.writer(csv_file1, delimiter='\t', lineterminator='\n')
    i=0
    stopped=reader[reader["why_stopped"].notnull()]   
    for ind, row in stopped.iterrows():
        # get all the classes that have a probability more than a threshold of 0.01 and order them based on the likelihood
        # from bigger to smaller
        print(probs[i])
        class_indices=sorted([j for j in range(len(probs[i])) if probs[i][j] >= 0.01], reverse=True)
        class_indices=class_indices[0:3]
        print(row[['why_stopped', 'nct_id']])
        i=i+1
        # create a list of the classes assigned
        subclasses_all=[]
        superclasses_all=[]
        for class_index in class_indices:
            subclasses_all.append(get_class(class_index))
            superclasses_all.append(class_map(get_class(class_index)))
        writer1.writerow([row['why_stopped'].replace('\r~', ''), subclasses_all, superclasses_all])

if __name__ == '__main__':
    args = get_parser().parse_args()

    main(args.input_file, args.output_file)
    