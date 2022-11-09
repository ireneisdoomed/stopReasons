#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 16:39:49 2022
​
@author: lesya
"""
import torch
from transformers import BertTokenizer
import pandas as pd
import csv
from preprocess import preprocessing_for_bert
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from predict import bert_predict
import sys
import numpy as np
import logging
import torch.nn as nn
from BertClassifier import BertClassifier
​
​
def create_input(input_dataframe):
    # Run `preprocessing_for_bert` on the input data
    inputs, masks = preprocessing_for_bert(input_dataframe.why_stopped)
    # Create the DataLoader for the dataset
    dataset = TensorDataset(inputs, masks)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=16)
    print('The input data is ready')
    return dataloader
​
def predict(model, 
            input_file, 
            output_file_stopped,
            output_file_notstopped):
    # load the model    
    model=torch.load(model)
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
    # load the imput file studies.tsv, and extract the columns needed
    reader = pd.read_csv(input_file, skiprows=1, names=names_studies, delimiter='|')
    reader=(reader[['why_stopped','phase','nct_id', 'start_date', 'overall_status', 'last_update_posted_date', 'completion_date']]).drop_duplicates()
    # generate probabilities
    probs = bert_predict(model, create_input(reader[reader["why_stopped"].notnull()][0:50]))
    # get all the classes that have a probability more than a threshold of 0.3 (empirically derived best threshold)
    probs[np.where(probs < 0.3)] = 0
    probs[np.where(probs >= 0.3)] = 1
    probs=probs.astype(int)
    # map the predictions to high-level classes: Possibly Negative, Negative,  Safety/Side effects, Success, Invalid, Neutral 
    superclasses=np.zeros([len(probs),6])
    superclasses=superclasses.astype(int)
    superclasses[:,0]=probs[:,2]
    superclasses[:,1]=probs[:,10]
    superclasses[:,2]=probs[:,13]
    superclasses[:,3]=probs[:,16]
    superclasses[:,4]=probs[:,8]
    neutral_indices=[0,2,3,4,5,6,7,9,11,12,14,15]
​
    superclasses[:,5]=[1 if j>0 else 0 for j in np.sum(probs[:,neutral_indices], axis=1)]
                       
    # writing to file. The columns that start with prefix H_ indicate parent(high-level) classes
    csv_file1=open(output_file_stopped, 'w')
    writer1 = csv.writer(csv_file1, delimiter='\t', lineterminator='\n')
    writer1.writerow(["Text", 'Phase','NCT_ID', 'Start_Date', 'Overall_Status', 
           'Last_Update_Posted_Date', 'Completion_Date','Another_Study', 
           'Business_Administrative', 'Covid19',
           'Endpoint_Met', 'Ethical_Reason', 'Insufficient_Data',
           'Insufficient_Enrollment', 'Interim_Analysis', 'Invalid_Reason',
           'Logistics_Resources', 'Negative', 'No_Context', 'Regulatory',
           'Safety_Sideeffects', 'Study_Design', 'Study_Staff_Moved',
           'Success', 'H_Possibly_Negative', 'H_Negative', 'H_Safety_Sideeffects', 'H_Success', 'H_Invalid_Reason', 'H_Neutral'])
    i=0
    for ind, row in reader[reader["why_stopped"].notnull()][0:50].iterrows():
        child=probs[i]
        parent=superclasses[i]
        writer1.writerow([row['why_stopped'].replace('\r~', ''),row['phase'],
                          row['nct_id'],row['start_date'], row['overall_status'],
                          row['last_update_posted_date'],row['completion_date'],
                          child[0],child[1],child[2], child[3], child[4],
                          child[5], child[6], child[7], child[8], child[9], child[10], 
                          child[11], child[12], child[13], child[14], child[15], 
                          child[16], parent[0], parent[1], parent[2],
                          parent[3], parent[4], parent[5]])
        i=i+1
    # write to a separate file non-stopped trials
    csv_file2=open(output_file_notstopped, "w")
    writer2 = csv.writer(csv_file2, delimiter='\t')
    writer2.writerow(["Stop_Reason", 'Phase','NCT_ID', 'Start_Date', 'Overall_Status', 
           'Last_Update_Posted_Date', 'Completion_Date'])
    not_stopped=reader[reader["why_stopped"].isnull()]
    for ind,row in not_stopped.iterrows():
        writer2.writerow([row['why_stopped'],row['phase'],row['nct_id'],row['start_date'],
                          row['overall_status'],row['last_update_posted_date'],row['completion_date']])
​
    
if __name__ == "__main__":
    predict('/users/lesya/Documents/bert_stop_reasons_new_l', '/users/lesya/Documents/data/studies.txt', '/users/lesya/Documents/wh_st.csv', '/users/lesya/Documents/nwh_st.csv',)
     # predict(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3])