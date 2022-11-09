#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 21:46:27 2021

@author: olesyar
"""

# Libraries
import re
import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
import transformers
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
words = stopwords.words("english")
lemma = nltk.stem.WordNetLemmatizer()



if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Add the Data using pandas
names_all = ['text','label1', 'label2', 'label3']
# Define the list of classes:
classes=['Business_Administrative', 
         'Another_Study', 
         'Negative',
         'Study_Design', 
         'Invalid_Reason',
         'Ethical_Reason', 
         'Insufficient_Data',
         'Insufficient_Enrollment', 
         'Study_Staff_Moved', 
         'Endpoint_Met',
         'Regulatory', 
         'Logistics_Resources',
         'Safety_Sideeffects', 
         'No_Context', 
         'Success',
         'Interim_Analysis', 
         'Covid19']

# Download data
corpus = pd.read_csv("/Users/olesyar/Documents/new_data.txt", skiprows=0, 
                        encoding='unicode_escape', names=names_all, delimiter='\t')
# corpus['text'].dropna(inplace=True)
# merge columns into a new one to apply multibinarizer
corpus["why_stopped"]=corpus[['label1', 'label2', 'label3']].values.tolist()


# remove "nan" values from the list to avoid the extra class
for ind, row in corpus.iterrows():
        row["why_stopped"]=[x for x in row["why_stopped"] if str(x)!='nan']

# print(corpus['label'].unique())
# For multiclass learning task, we need to transform each label into a one-hot vector:
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
one_hot=mlb.fit_transform(corpus['why_stopped'])



# replace the values of the classes inthe list with one-hot vectors:
for ind, row in corpus.iterrows():
        row["why_stopped"]=one_hot[ind]
        

from transformers import BertTokenizer
# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
# Defining some key variables that will be used later on in the training
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 7
LEARNING_RATE = 5e-05
# if mode is 2015, the corpus is split into training and test. If mode is 2022, all the data is used for training
MODE="2015"

all_texts = corpus['text']
# Encode our concatenated data
encoded_texts = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_texts]
# Find the maximum length
max_len = max([len(sent) for sent in encoded_texts])
print('Max length: ', max_len)
# Specify `MAX_LEN`
MAX_LEN = max_len

n_class=len(mlb.classes_)




corpus = corpus.sample(frac=1).reset_index(drop=True)
test=corpus[1:360]
training=corpus[361:3599]


# ---new code starts here:
# split the training and validation set (90% and 10%)
X = training.text.values
y = training.why_stopped.values
X_train, X_val, y_train, y_val =train_test_split(X, y, test_size=0.1, random_state=2020)
y_train=np.vstack(y_train).astype(np.float)
y_val=np.vstack(y_val).astype(np.float)


def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"won't", "will not ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'\n", " ", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"\'\xa0", " ", text)
    text = re.sub('\s+', ' ', text)
    text = ''.join(c for c in text if not c.isnumeric())
    
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


#Load the Bert tokenizer
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base',do_lower_case=True)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)
# Create a funcition to tokenize a set of text

def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # create empty lists to store outputs
    input_ids = []
    attention_masks = []
    
    #for every sentence...
    
    for sent in data:
        # 'encode_plus will':
        # (1) Tokenize the sentence
        # (2) Add the `[CLS]` and `[SEP]` token to the start and end
        # (3) Truncate/Pad sentence to max length
        # (4) Map tokens to their IDs
        # (5) Create attention mask
        # (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text = text_preprocessing(sent),   #preprocess sentence
            add_special_tokens = True,         #Add `[CLS]` and `[SEP]`
            max_length= MAX_LEN  ,             #Max length to truncate/pad
            pad_to_max_length = True,          #pad sentence to max length 
            return_attention_mask= True        #Return attention mask 
        )
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
        
    #convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    
    return input_ids,attention_masks



# Run function 'preprocessing_for_bert' on the train set and validation set
print('Tokenizing data...')
train_inputs, train_masks = preprocessing_for_bert(X_train)
val_inputs, val_masks = preprocessing_for_bert(X_val)

# Convert other data types to torch.Tensor
train_labels = torch.tensor(y_train)
val_labels = torch.tensor(y_val)

## For fine-tuning Bert, the authors recommmend a batch size of 16 or 32
batch_size = 16

# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs,train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)


# Create the BertClassifier class

class BertClassifier(nn.Module):
    """
        Bert Model for classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param   bert: a BertModel object
        @param   classifier: a torch.nn.Module classifier
        @param   freeze_bert (bool): Set `False` to fine_tune the Bert model
        """
        super(BertClassifier,self).__init__()
        # Specify hidden size of Bert, hidden size of our classifier, and number of labels
        D_in, H,D_out = 768,50,n_class
        
#         self.bert = RobertaModel.from_pretrained('roberta-base')
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
        logit = self.classifier(last_hidden_state_cls)
        
#         logits = self.sigmoid(logit)
        
        return logit
    
    


def initialize_model(epochs=EPOCHS):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(freeze_bert=False)
    
    bert_classifier.to(device)
    
    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                     lr=5e-5, #Default learning rate
                     eps=1e-8 #Default epsilon value
                     )
    
    # Total number of training steps
    total_steps = len(train_dataloader) * epochs
    
    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              num_warmup_steps=0, # Default value
                                              num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler    



# Specify loss function
#loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.BCEWithLogitsLoss()

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, train_dataloader, val_dataloader=None, epochs=EPOCHS, evaluation=False):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels.float())
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20--50000 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    
    print("Training complete!")


def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels.float())
        val_loss.append(loss.item())

        # Get the predictions
        #preds = torch.argmax(logits, dim=1).flatten()
        
        # Calculate the accuracy rate
        #accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        accuracy = accuracy_thresh(logits.view(-1,17),b_labels.view(-1,17))
        
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

def accuracy_thresh(y_pred, y_true, thresh:float=0.5, sigmoid:bool=True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: 
        y_pred = y_pred.sigmoid()
    return ((y_pred>thresh)==y_true.byte()).float().mean().item()
    #return np.mean(((y_pred>thresh).float()==y_true.float()).float().cpu().numpy(), axis=1).sum()



set_seed(42)    # Set seed for reproducibility
bert_classifier, optimizer, scheduler = initialize_model(epochs=EPOCHS)
train(bert_classifier, train_dataloader, val_dataloader, epochs=EPOCHS, evaluation=True)


# Save the model and data it was trained and tested on
import csv
torch.save(bert_classifier,'/Users/olesyar/Documents/bert_stop_reasons_last_whole')

csv_file=open('/Users/olesyar/Documents/testset_last_whole.csv', 'w')
writer = csv.writer(csv_file)
for ind, row in test.iterrows():
    label=row.why_stopped
    writer.writerow([row["text"],label[0],label[1], label[2], label[3], label[4], label[5], label[6], label[7], 
                     label[8], label[9], label[10], label[11], label[12], label[13], label[14], 
                     label[15], label[16]])
    
csv_file=open('/Users/olesyar/Documents/trainset_last_whole.csv', 'w')
writer = csv.writer(csv_file)
for ind, row in training.iterrows():
    label=row.why_stopped
    writer.writerow([row["text"],label[0],label[1], label[2], label[3], label[4], label[5], label[6], label[7], 
                     label[8], label[9], label[10], label[11], label[12], label[13], label[14], 
                     label[15], label[16]])    



def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    #probs = F.softmax(all_logits, dim=1).cpu().numpy()
    probs = all_logits.sigmoid().cpu().numpy()
    

    return probs


def create_test(test_dataframe):
    # Run `preprocessing_for_bert` on the test set
    test_inputs, test_masks = preprocessing_for_bert(test_dataframe.text)
    labels=test_dataframe.why_stopped
    # Create the DataLoader for our test set
    test_dataset = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)
    print('The test set is ready')
    return test_dataloader



# bert_classifier=torch.load("/Users/olesyar/Documents/bert_rare_common")
probs = bert_predict(bert_classifier, create_test(test))

# where there is at least one element above threshold, decode all such probabilities as predictions
probs[np.where(probs < 0.3)] = 0
probs[np.where(probs >= 0.3)] = 1
probs=probs.astype(int)



csv_file=open('/Users/olesyar/Documents/test_eval_new.csv', 'w')

writer = csv.writer(csv_file)

for ind, cl in test.iterrows():
    gold=cl.why_stopped
    pred=probs[ind-1]
    writer.writerow([cl["text"],gold[0],gold[1], gold[2], gold[3], gold[4], gold[5], gold[6], gold[7], 
                     gold[8], gold[9], gold[10], gold[11], gold[12], gold[13], gold[14], 
                     gold[15], gold[16], pred[0],pred[1],pred[2], 
                     pred[3], pred[4],pred[5], pred[6], pred[7], 
                     pred[8], pred[9], pred[10], pred[11], pred[12], 
                     pred[13], pred[14], pred[15], pred[16]])



def test_eval(input_data, output_data):
    # input data in .tsv/txt format, output file in a csv format
    file=pd.read_csv(input_data, skiprows=1, encoding='unicode_escape', names=names_all, delimiter='\t')
    file["why_stopped"]=file[['label1', 'label2', 'label3']].values.tolist()
    for ind, row in file.iterrows():
            row["why_stopped"]=[x for x in row["why_stopped"] if str(x)!='nan']
    file.why_stopped=file.why_stopped.astype(str)
    file_one_hot=mlb.fit_transform(file['why_stopped'])
    for label in classes:
        if label not in mlb.classes_:
            file=file.append([{'text':"-----",'label1':label,'label2':'nan','label3':'nan','why_stopped':'nan'}], ignore_index=True)
            file["why_stopped"]=file[['label1', 'label2', 'label3']].values.tolist()
            for ind, row in file.iterrows():
                    row["why_stopped"]=[x for x in row["why_stopped"] if str(x)!='nan']
            file_one_hot=mlb.fit_transform(file['why_stopped'])
    for ind, row in file.iterrows():
            row["why_stopped"]=file_one_hot[ind]
    file = file[file.text != "-----"]
    probs = bert_predict(bert_classifier, create_test(file))
    probs[np.where(probs < 0.3)] = 0
    probs[np.where(probs >= 0.3)] = 1
    probs=probs.astype(int)
    csv_file=open(output_data, 'w')
    writer = csv.writer(csv_file)
    for ind, cl in file.iterrows():
        gold=cl.why_stopped
        pred=probs[ind]
        writer.writerow([cl["text"],gold[0],gold[1], gold[2], gold[3], gold[4], 
                         gold[5], gold[6], gold[7],gold[8], gold[9], gold[10], 
                         gold[11], gold[12], gold[13], gold[14],
                         gold[15], gold[16], pred[0],pred[1],pred[2],
                         pred[3], pred[4],pred[5], pred[6], pred[7],
                         pred[8], pred[9], pred[10], pred[11], pred[12],
                         pred[13], pred[14], pred[15], pred[16]])
    report=classification_report(np.vstack(file.why_stopped), probs, target_names=mlb.classes_)
    print(report)
    return report








# modify the model path to load the model    
model=torch.load('/Users/olesyar/Documents/bert_stop_reasons_new_l')

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

import logging
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


import numpy as np
studies_file = "/Users/olesyar/Documents/data/studies.txt"
reader = pd.read_csv(studies_file, skiprows=1, names=names_studies, delimiter='|')
reader=(reader[['why_stopped','phase','nct_id', 'start_date', 'overall_status', 'last_update_posted_date', 'completion_date']]).drop_duplicates()
csv_file1=open('/Users/olesyar/Documents/data/stopped_predictions_new.tsv', "w")
writer1 = csv.writer(csv_file1, delimiter='\t', lineterminator='\n')
csv_file2=open('/Users/olesyar/Documents/data/notstopped_predictions_new.tsv', "w")
writer2 = csv.writer(csv_file2, delimiter='\t')

def create_inp(input_dataframe):
    # Run `preprocessing_for_bert` on the test set
    test_inputs, test_masks = preprocessing_for_bert(input_dataframe.why_stopped)

    # Create the DataLoader for our test set
    test_dataset = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)
    print('The test set is ready')
    return test_dataloader



# bert_classifier=torch.load("/Users/olesyar/Documents/bert_rare_common")
probs = bert_predict(model, create_inp(reader[reader["why_stopped"].notnull()]))
probs[np.where(probs < 0.3)] = 0
probs[np.where(probs >= 0.3)] = 1
probs=probs.astype(int)
i=0

for ind, row in reader[reader["why_stopped"].notnull()].iterrows():
    pred=probs[i]
    writer1.writerow([row['why_stopped'].replace('\r~', ''),row['phase'],row['nct_id'],row['start_date'], row['overall_status'],row['last_update_posted_date'],row['completion_date'],
                      pred[0],pred[1],pred[2],
                      pred[3], pred[4],pred[5], pred[6], pred[7],
                      pred[8], pred[9], pred[10], pred[11], pred[12],
                      pred[13], pred[14], pred[15], pred[16]])
    i=i+1

#probs = all_logits.sigmoid().cpu().numpy()

## Compute predicted probabilities on the test set


# Evalueate the bert classifier

# evaluate_roc(probs, y_val)
























# def load_ckp(checkpoint_fpath, model, optimizer):
#     """
#     checkpoint_path: path to save checkpoint
#     model: model that we want to load checkpoint parameters into       
#     optimizer: optimizer we defined in previous training
#     """
#     # load check point
#     checkpoint = torch.load(checkpoint_fpath)
#     # initialize state_dict from checkpoint to model
#     model.load_state_dict(checkpoint['state_dict'])
#     # initialize optimizer from checkpoint to optimizer
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     # initialize valid_loss_min from checkpoint to valid_loss_min
#     valid_loss_min = checkpoint['valid_loss_min']
#     # return model, optimizer, epoch value, min validation loss 
#     return model, optimizer, checkpoint['epoch'], valid_loss_min.item()


# import shutil, sys   
# def save_ckp(state, is_best, checkpoint_path, best_model_path):
#     """
#     state: checkpoint we want to save
#     is_best: is this the best checkpoint; min validation loss
#     checkpoint_path: path to save checkpoint
#     best_model_path: path to save best model
#     """
#     f_path = checkpoint_path
#     # save checkpoint data to the path given, checkpoint_path
#     torch.save(state, f_path)
#     # if it is a best model, min validation loss
#     if is_best:
#         best_fpath = best_model_path
#         # copy that checkpoint file to best path given, best_model_path
#         shutil.copyfile(f_path, best_fpath)
        
        
# val_targets=[]
# val_outputs=[] 



# def train_model(start_epochs,  n_epochs, valid_loss_min_input, 
#                 training_loader, validation_loader, model, 
#                 optimizer, checkpoint_path, best_model_path):
   
#   # initialize tracker for minimum validation loss
#   valid_loss_min = valid_loss_min_input 
   
 
#   for epoch in range(start_epochs, n_epochs+1):
#     train_loss = 0
#     valid_loss = 0

#     model.train()
#     print('############# Epoch {}: Training Start   #############'.format(epoch))
#     for batch_idx, data in enumerate(training_loader):
#         # print( 'batch:', batch_idx)
#         ids = data['ids'].to(device, dtype = torch.long)
#         mask = data['mask'].to(device, dtype = torch.long)
#         token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
#         targets = data['targets'].to(device, dtype = torch.float)
#         outputs = model(ids, mask, token_type_ids, return_dict=False)
#         optimizer.zero_grad()
#         loss = loss_fn(outputs, targets)
#         #if batch_idx%5000==0:
#          #   print(f'Epoch: {epoch}, Training Loss:  {loss.item()}')
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # print('before loss data in training', loss.item(), train_loss)
#         train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
    
#     print('############# Epoch {}: Training End     #############'.format(epoch))
    
#     print('############# Epoch {}: Validation Start   #############'.format(epoch))
#     ######################    
#     # validate the model #
#     ######################
 
#     model.eval()
   
#     with torch.no_grad():
#       for batch_idx, data in enumerate(validation_loader, 0):
#             ids = data['ids'].to(device, dtype = torch.long)
#             mask = data['mask'].to(device, dtype = torch.long)
#             token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
#             targets = data['targets'].to(device, dtype = torch.float)
#             outputs = model(ids, mask, token_type_ids, return_dict=False)

#             loss = loss_fn(outputs, targets)
#             valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
#             val_targets.extend(targets.cpu().detach().numpy().tolist())
#             val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
#             evaluate(val_targets,val_outputs)
#       print('############# Epoch {}: Validation End     #############'.format(epoch))
#       # calculate average losses
#       #print('before cal avg train loss', train_loss)
#       train_loss = train_loss/len(training_loader)
#       valid_loss = valid_loss/len(validation_loader)
#       # print training/validation statistics 
#       print('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
#             epoch, 
#             train_loss,
#             valid_loss
#             ))
      
#       # create checkpoint variable and add important data
#       checkpoint = {
#             'epoch': epoch + 1,
#             'valid_loss_min': valid_loss,
#             'state_dict': model.state_dict(),
#             'optimizer': optimizer.state_dict()
#       }
        
#         # save checkpoint
#       save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        
#       ## TODO: save the model if validation loss has decreased
#       if valid_loss <= valid_loss_min:
#         print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
#         # save checkpoint as best model
#         save_ckp(checkpoint, True, checkpoint_path, best_model_path)
#         valid_loss_min = valid_loss

#     print('############# Epoch {}  Done   #############\n'.format(epoch))


#   return model




# checkpoint_path = '/Users/olesyar/Documents/checkpoints/checkpoint'
# best_model = '/Users/olesyar/Documents/checkpoints/models/best_model'
# trained_model = train_model(1, 15, np.Inf, training_loader, validation_loader, model, 
#                       optimizer,checkpoint_path,best_model)







# # corpus['label'] = corpus['label'].replace(['Business_Administrative', 'Another_Study', 'Negative',
# #         'Study_Design', 'Invalid_Reason',
# #        'Ethical_Reason', 'Insufficient_Data',
# #        'Insufficient_Enrollment', 'Study_Staff_Moved', 'Endpoint_Met',
# #         'Regulatory', 'Logistics_Resources',
# #        'Safety_Sideeffects', 'No_Context', 'Success',
# #        'Interim_Analysis', 'Covid19'], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]) 


# # Display 5 random samples
# corpus.sample(5)

# corpus = corpus.sample(frac=1).reset_index(drop=True)
# test=corpus[1:409]
# training=corpus[410:4096]

# X = training['text']
# y = training['label']

# # split into train and test sets
# # check unique y_val and y_train to make sure each class is represented
# X_train, X_val, y_train, y_val =\
#     train_test_split(X, y, test_size=0.1, random_state=2020)
    
# def text_preprocessing(text):
#     """
#     - Remove entity mentions (eg. '@united')
#     - Correct errors (eg. '&amp;' to '&')
#     @param    text (str): a string to be processed.
#     @return   text (Str): the processed string.
#     """
#     # Remove '@name'
#     text = re.sub(r'(@.*?)[\s]', ' ', text)

#     # Replace '&amp;' with '&'
#     text = re.sub(r'&amp;', '&', text)

#     # Remove trailing whitespace
#     text = re.sub(r'\s+', ' ', text).strip()

#     return text




# # Create a function to BERT tokenize a set of texts
# def preprocessing_for_bert(data):
#     """Perform required preprocessing steps for pretrained BERT.
#     @param    data (np.array): Array of texts to be processed.
#     @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
#     @return   attention_masks (torch.Tensor): Tensor of indices specifying which
#                   tokens should be attended to by the model.
#     """
#     # Create empty lists to store outputs
#     input_ids = []
#     attention_masks = []
#     # For every sentence...
#     for sent in data:
#         # `encode_plus` will:
#         #    (1) Tokenize the sentence
#         #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
#         #    (3) Truncate/Pad sentence to max length
#         #    (4) Map tokens to their IDs
#         #    (5) Create attention mask
#         #    (6) Return a dictionary of outputs
#         encoded_sent = tokenizer.encode_plus(
#             text=text_preprocessing(sent),  # Preprocess sentence
#             add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
#             max_length=MAX_LEN,                  # Max length to truncate/pad
#             pad_to_max_length=True,         # Pad sentence to max length
#             #return_tensors='pt',           # Return PyTorch tensor
#             return_attention_mask=True      # Return attention mask
#             )
        
#         # Add the outputs to the lists
#         input_ids.append(encoded_sent.get('input_ids'))
#         attention_masks.append(encoded_sent.get('attention_mask'))

#     # Convert lists to tensors
#     input_ids = torch.tensor(input_ids)
#     attention_masks = torch.tensor(attention_masks)
#     return input_ids, attention_masks


# # Concatenate train data and test data


# # Print sentence 0 and its encoded token ids
# token_ids = list(preprocessing_for_bert([X[0]])[0].squeeze().numpy())
# print('Original: ', X[0])
# print('Token IDs: ', token_ids)

# # Run function `preprocessing_for_bert` on the train set and the validation set
# print('Tokenizing data...')
# train_inputs, train_masks = preprocessing_for_bert(X_train)
# val_inputs, val_masks = preprocessing_for_bert(X_val)







# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# # Convert other data types to torch.Tensor
# val_labels = torch.tensor(y_val.values)
# train_labels = torch.tensor(y_train.values)


# # For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
# batch_size = 16

# # Create the DataLoader for our training set
# train_data = TensorDataset(train_inputs, train_masks, train_labels)
# train_sampler = RandomSampler(train_data)
# train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# # Create the DataLoader for our validation set
# val_data = TensorDataset(val_inputs, val_masks, val_labels)
# val_sampler = SequentialSampler(val_data)
# val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)






# # Create the BertClassfier class
# class BertClassifier(nn.Module):
#     """Bert Model for Classification Tasks.
#     """
#     def __init__(self, freeze_bert=False):
#         """
#         @param    bert: a BertModel object
#         @param    classifier: a torch.nn.Module classifier
#         @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
#         """
#         super(BertClassifier, self).__init__()
#         # Specify hidden size of BERT, hidden size of our classifier, and number of labels
#         D_in, H, D_out = 768, 50, 17

#         # Instantiate BERT model
#         self.bert = BertModel.from_pretrained('bert-base-uncased')

#         # Instantiate an one-layer feed-forward classifier
#         self.classifier = nn.Sequential(
#             nn.Linear(D_in, H),
#             nn.ReLU(),
#             #nn.Dropout(0.5),
#             nn.Linear(H, D_out)
#         )

#         # Freeze the BERT model
#         if freeze_bert:
#             for param in self.bert.parameters():
#                 param.requires_grad = False
        
#     def forward(self, input_ids, attention_mask):
#         """
#         Feed input to BERT and the classifier to compute logits.
#         @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
#                       max_length)
#         @param    attention_mask (torch.Tensor): a tensor that hold attention mask
#                       information with shape (batch_size, max_length)
#         @return   logits (torch.Tensor): an output tensor with shape (batch_size,
#                       num_labels)
#         """
#         # Feed input to BERT
#         outputs = self.bert(input_ids=input_ids,
#                             attention_mask=attention_mask)
        
#         # Extract the last hidden state of the token `[CLS]` for classification task
#         last_hidden_state_cls = outputs[0][:, 0, :]

#         # Feed input to classifier to compute logits
#         logits = self.classifier(last_hidden_state_cls)

#         return logits
    
    
    
    
    
# from transformers import AdamW, get_linear_schedule_with_warmup

# def initialize_model(epochs=15):
#     """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
#     """
#     # Instantiate Bert Classifier
#     bert_classifier = BertClassifier(freeze_bert=False)

#     # Tell PyTorch to run the model on GPU
#     bert_classifier.to(device)

#     # Create the optimizer
#     optimizer = AdamW(bert_classifier.parameters(),
#                       lr=5e-5,    # Default learning rate
#                       eps=1e-8    # Default epsilon value
#                       )

#     # Total number of training steps
#     total_steps = len(train_dataloader) * epochs

#     # Set up the learning rate scheduler
#     scheduler = get_linear_schedule_with_warmup(optimizer,
#                                                 num_warmup_steps=0, # Default value
#                                                 num_training_steps=total_steps)
#     return bert_classifier, optimizer, scheduler




# import random
# import time

# # Specify loss function
# loss_fn = nn.CrossEntropyLoss()

# def set_seed(seed_value):
#     """Set seed for reproducibility.
#     """
#     random.seed(seed_value)
#     np.random.seed(seed_value)
#     torch.manual_seed(seed_value)
#     torch.cuda.manual_seed_all(seed_value)

# def train(model, train_dataloader, val_dataloader=None, epochs=15, evaluation=False):
#     """Train the BertClassifier model.
#     """
#     # Start training loop
#     print("Start training...\n")
#     for epoch_i in range(epochs):
#         # =======================================
#         #               Training
#         # =======================================
#         # Print the header of the result table
#         print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
#         print("-"*70)

#         # Measure the elapsed time of each epoch
#         t0_epoch, t0_batch = time.time(), time.time()

#         # Reset tracking variables at the beginning of each epoch
#         total_loss, batch_loss, batch_counts = 0, 0, 0

#         # Put the model into the training mode
#         model.train()

#         # For each batch of training data...
#         for step, batch in enumerate(train_dataloader):
#             batch_counts +=1
#             # Load batch to GPU
#             b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

#             # Zero out any previously calculated gradients
#             model.zero_grad()

#             # Perform a forward pass. This will return logits.
#             logits = model(b_input_ids, b_attn_mask)

#             # Compute loss and accumulate the loss values
#             loss = loss_fn(logits, b_labels)
#             batch_loss += loss.item()
#             total_loss += loss.item()

#             # Perform a backward pass to calculate gradients
#             loss.backward()

#             # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

#             # Update parameters and the learning rate
#             optimizer.step()
#             scheduler.step()

#             # Print the loss values and time elapsed for every 20 batches
#             if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
#                 # Calculate time elapsed for 20 batches
#                 time_elapsed = time.time() - t0_batch

#                 # Print training results
#                 print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

#                 # Reset batch tracking variables
#                 batch_loss, batch_counts = 0, 0
#                 t0_batch = time.time()

#         # Calculate the average loss over the entire training data
#         avg_train_loss = total_loss / len(train_dataloader)

#         print("-"*70)
#         # =======================================
#         #               Evaluation
#         # =======================================
#         if evaluation == True:
#             # After the completion of each training epoch, measure the model's performance
#             # on our validation set.
#             val_loss, val_accuracy = evaluate(model, val_dataloader)

#             # Print performance over the entire training data
#             time_elapsed = time.time() - t0_epoch
            
#             print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
#             print("-"*70)
#         print("\n")
    
#     print("Training complete!")


# def evaluate(model, val_dataloader):
#     """After the completion of each training epoch, measure the model's performance
#     on our validation set.
#     """
#     # Put the model into the evaluation mode. The dropout layers are disabled during
#     # the test time.
#     model.eval()

#     # Tracking variables
#     val_accuracy = []
#     val_loss = []

#     # For each batch in our validation set...
#     for batch in val_dataloader:
#         # Load batch to GPU
#         b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        

#         # Compute logits
#         with torch.no_grad():
#             logits = model(b_input_ids, b_attn_mask)

#         # Compute loss
#         loss = loss_fn(logits,b_labels)
#         val_loss.append(loss.item())

#         # Get the predictions
#         preds = torch.argmax(logits, dim=1).flatten()

#         # Calculate the accuracy rate
#         accuracy = (preds == b_labels).cpu().numpy().mean() * 100
#         val_accuracy.append(accuracy)

#     # Compute the average accuracy and loss over the validation set.
#     val_loss = np.mean(val_loss)
#     val_accuracy = np.mean(val_accuracy)

#     return val_loss, val_accuracy


# print("training starts")


# set_seed(42)    # Set seed for reproducibility
# bert_classifier, optimizer, scheduler = initialize_model(epochs=15)
# train(bert_classifier, train_dataloader, val_dataloader, epochs=15, evaluation=True)



# # best epoch 15: 78.39

# torch.save(bert_classifier,'/Users/olesyar/Documents/bert_stop_reasons')



# import torch.nn.functional as F

# def bert_predict(model, test_dataloader):
#     """Perform a forward pass on the trained BERT model to predict probabilities
#     """
#     # Put the model into the evaluation mode. The dropout layers are disabled during
#     # the test time.
#     model.eval()

#     all_logits = []

#     # For each batch in our test set...
#     for batch in test_dataloader:
#         # Load batch to GPU
#         b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

#         # Compute logits
#         with torch.no_grad():
#             logits = model(b_input_ids, b_attn_mask)
#         all_logits.append(logits)
    
#     # Concatenate logits from each batch
#     all_logits = torch.cat(all_logits, dim=0)

#     # Apply softmax to calculate probabilities
#     probs = F.softmax(all_logits, dim=1).cpu().numpy()

#     return probs


        
#     def forward(self, input_ids, attention_mask):
#         """
#         Feed input to BERT and the classifier to compute logits.
#         @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
#                       max_length)
#         @param    attention_mask (torch.Tensor): a tensor that hold attention mask
#                       information with shape (batch_size, max_length)
#         @return   logits (torch.Tensor): an output tensor with shape (batch_size,
#                       num_labels)
#         """
#         # Feed input to BERT
#         outputs = self.bert(input_ids=input_ids,
#                             attention_mask=attention_mask)
        
#         # Extract the last hidden state of the token `[CLS]` for classification task
#         last_hidden_state_cls = outputs[0][:, 0, :]

#         # Feed input to classifier to compute logits
#         logits = self.classifier(last_hidden_state_cls)

#         return logits


# def create_test(test_dataframe):
#     # Run `preprocessing_for_bert` on the test set
#     test_inputs, test_masks = preprocessing_for_bert(test_dataframe.text)
#     labels=test_dataframe.label
#     # Create the DataLoader for our test set
#     test_dataset = TensorDataset(test_inputs, test_masks)
#     test_sampler = SequentialSampler(test_dataset)
#     test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)
#     print('The test set is ready')
#     return test_dataloader


# from numpy import argmax 
# import csv
# # bert_classifier=torch.load("/Users/olesyar/Documents/bert_rare_common")
# probs = bert_predict(bert_classifier, create_test(test))
# csv_file=open('/Users/olesyar/Documents/test_eval.csv', 'w')
# writer = csv.writer(csv_file)
# writer.writerow(['sent', 'gold','pred'])
# i=0
# labels=test['label']
# text=test['text']
# for ind, text in test.iterrows():
#     writer.writerow([text["text"],text["label"],argmax(probs[i])])
#     i=i+1




