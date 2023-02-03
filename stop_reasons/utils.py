import shutil
from typing import Tuple

from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import torch
from transformers import AutoTokenizer, EvalPrediction, TFAutoModelForSequenceClassification

def export_labels_to_model(model_name: str, model) -> None:
    """
    Reads from a model configuration to export the labels of the class target to a file in the model's assets folder.
    
    Args:
      model_name (str): The name of the model. This is used to create a directory for the model.
      model: The model to export.
    """
    labels = model.config.label2id
    labels = sorted(labels, key=labels.get)

    model_assets_path = f'models/{model_name}/saved_model/1/assets'

    with open(f'{model_assets_path}/labels.txt', 'w') as f:
        f.write('\n'.join(labels))

def save_model_from_hub(model_name: str) -> None:
    """
    We load the model and tokenizer from the HuggingFace hub, save them to the `models` directory, and then export
    the labels of the model to the directory that contains all the assets.
    
    Args:
      model_name (str): The name of the model you want to save.
    """

    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(f'models/{model_name}', from_tf=True, save_format='tf', saved_model=True)
    tokenizer.save_pretrained(f'models/{model_name}_tokenizer', from_tf=True, save_format='tf')
    export_labels_to_model(model_name, model)

    print(f"Model {model_name} saved.")

def copy_tokenizer_vocab_to_model(model_name):
    """
    We copy the tokenizer's vocabulary to the model's directory, so that we can use the model for
    predictions.

    Args:
        model_name (str): The name of the model you want to use.
    """

    tokenizer_vocab_path = f'models/{model_name}_tokenizer/vocab.txt'
    model_assets_path = f'models/{model_name}/saved_model/1/assets'

    shutil.copyfile(tokenizer_vocab_path, f'{model_assets_path}/vocab.txt')
    

def prepare_model_from_hub(model_name: str, model_dir:str) -> None:
    """
    If the model directory doesn't exist, download the model from the HuggingFace Hub, and copy the tokenizer
    vocab to the model directory so that the format can be digested by Spark NLP.
    
    Args:
      model_name (str): The name of the model you want to use.
      model_dir (str): The directory where the model will be saved.
    """

    model_path = f'{model_dir}/{model_name}'

    if not Path(model_path).is_dir():
        save_model_from_hub(model_name)
        copy_tokenizer_vocab_to_model(model_name)

def get_label_metadata(dataset):
    labels = [label for label in dataset['train'].features.keys() if label not in ['text', 'label_descriptions']]
    id2label = dict(enumerate(labels))
    label2id = {label:idx for idx, label in enumerate(labels)}
    return labels, id2label, label2id

def multi_label_metrics(predictions, labels, threshold=0.5):
  # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
  sigmoid = torch.nn.Sigmoid()
  probs = sigmoid(torch.Tensor(predictions))
  # next, use threshold to turn them into integer predictions
  y_pred = np.zeros(probs.shape)
  y_pred[np.where(probs >= threshold)] = 1
  # finally, compute metrics
  y_true = labels
  f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
  roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
  accuracy = accuracy_score(y_true, y_pred)
  return {'f1': f1_micro_average,
              'roc_auc': roc_auc,
              'accuracy': accuracy}


def compute_metrics(p: EvalPrediction):
  preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
  return multi_label_metrics(
    predictions=preds, 
    labels=p.label_ids
  )

def prepare_splits_for_training(dataset, subset_data):
  """Splits and shuffles the dataset into train and test splits.

  Args:
      dataset (DatasetDict): The dataset to split. 
      subset_data (bool, optional): Flag to use a subset of the data.

  Returns:
      Tuple[Dataset]: One dataset object per train, test split.
  """
  fraction = 0.1 if subset_data else 1
  splits = [dataset["train"], dataset["test"]]

  return [
    split.shuffle(seed=42).select(range(int(len(split) * fraction)))
    for split in splits
  ]

def convert_to_tf_dataset(dataset, data_collator, shuffle_flag, batch_size):
    return (
        dataset.to_tf_dataset(
            columns=["attention_mask", "input_ids", "token_type_ids"],
            label_cols=["labels"],
            shuffle=shuffle_flag,
            collate_fn=data_collator,
            batch_size=batch_size
        )
    )
