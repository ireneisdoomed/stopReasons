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
import csv
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.common_classes import bert_predict, preprocessing_for_bert, CLASS_TO_IDX, CLASS_TO_SUPER
from src.BertClassifier import BertClassifier

def prepare_data(df: pd.DataFrame) -> DataLoader:
    """
    Creates the embeddings for the reasons to stop and loads vectors to DataLoader.
    """

    # Run `preprocessing_for_bert` on the data set
    data_inputs, data_masks = preprocessing_for_bert(df[df["why_stopped"].notnull()].why_stopped)

    # Create the DataLoader for our prediction set
    dataset = TensorDataset(data_inputs, data_masks)

    return DataLoader(dataset, batch_size=32, num_workers=5)


def get_parser():
    """
    Get parser object for script predict.py.
    """
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
        '--model',
        help='Input PyTorch model to be applied on the data.',
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


def main(input_file: str, model_path: str, output_file: str) -> None:
    """
    Module to apply a NLP model to categorise the reason why a clinical trial has stopped early.
    """

    # initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # load model
    model = torch.load(model_path)
    logging.info(f"{model_path} is ready to be used.")

    # load the input file studies.tsv, and extract the columns needed
    studies_file = input_file
    studies = (
        pd.read_csv(studies_file, sep='\t', lineterminator='\n').dropna(subset=['why_stopped'], axis=0)
        [['why_stopped', 'nct_id']].drop_duplicates()
    )

    # generate probabilities
    model_data_loader = prepare_data(studies)
    logging.info('Embeddings generated from input. \n Data is ready to be used. Making predictions...')
    probs = bert_predict(model, model_data_loader) # numpy matrix of shape (n_reasons, n_classes)
    idx_to_class_dict = {v: k for k, v in CLASS_TO_IDX.items()}
    probs_df = pd.DataFrame(probs).rename(columns=idx_to_class_dict)
    # Per row (reason to stop), get the labels of the classes with a probability that is higher than .3
    predictions_df = pd.DataFrame(
        probs_df.apply(lambda row: row.loc[row >= 0.3].index.values, axis=1)
    ).rename(columns={0: 'subclasses'})
    # Map each subclass to the corresponding superclass
    predictions_df['superclasses'] = predictions_df.subclasses.apply(lambda value: list(set(np.vectorize(CLASS_TO_SUPER.get)(value))) if len(value) > 0 else None)
    
    # merge the predictions with the studies dataframe to obtain a df with the nct_id, the reason to stop and the predicted classes
    studies_with_predictions = studies.merge(predictions_df, left_index=True, right_index=True)

    logging.info(f'Predictions are ready. Writing to {output_file}...')
    studies_with_predictions.to_json(output_file, orient='records', lines=True)


if __name__ == '__main__':
    args = get_parser().parse_args()

    main(args.input_file, args.model, args.output_file)


