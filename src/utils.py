import shutil

from pathlib import Path
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

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
    > We load the model and tokenizer from the HuggingFace hub, save them to the `models` directory, and then export
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
    > We copy the tokenizer's vocabulary to the model's directory, so that we can use the model for
    predictions.

    Args:
        model_name (str): The name of the model you want to use.
    """

    tokenizer_vocab_path = f'models/{model_name}_tokenizer/vocab.txt'
    model_assets_path = f'models/{model_name}/saved_model/1/assets'

    shutil.copyfile(tokenizer_vocab_path, f'{model_assets_path}/vocab.txt')
    

def prepare_model_from_hub(model_name: str, model_dir:str) -> None:
    """
    > If the model directory doesn't exist, download the model from the HuggingFace Hub, and copy the tokenizer
    vocab to the model directory so that the format can be digested by Spark NLP.
    
    Args:
      model_name (str): The name of the model you want to use.
      model_dir (str): The directory where the model will be saved.
    """

    model_path = f'{model_dir}/{model_name}'

    if not Path(model_path).is_dir():
        save_model_from_hub(model_name)
        copy_tokenizer_vocab_to_model(model_name)
