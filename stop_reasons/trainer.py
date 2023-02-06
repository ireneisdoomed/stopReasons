import logging
from typing import Optional

from datasets import load_dataset
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
import typer

from stop_reasons.utils import get_label_metadata, prepare_splits_for_training

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        # compute custom loss
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {
        'accuracy_thresh': accuracy_thresh(predictions, labels),
    }

def accuracy_thresh(y_pred, y_true, thresh=0.5): 
    y_pred = torch.from_numpy(y_pred).sigmoid()
    y_true = torch.from_numpy(y_true)
    return ((y_pred>thresh)==y_true.bool()).float().mean().item()

def tokenize(batch):
    """Tokenises the text and creates a numpy array with its assigned labels."""
    encoding = tokenizer(batch["text"], max_length=177, padding="max_length", truncation=True)

    labels_batch = {k: batch[k] for k in batch.keys() if k in labels}
    labels_matrix = np.zeros((len(batch["text"]), len(labels)))
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]
    encoding["labels"] = labels_matrix.tolist()
    return encoding


def preprocess(batch):
    NotImplementedError

def main(
    epochs: int = typer.Argument(...),
    output_model_name: str = typer.Argument(...),
    subset_data: bool = typer.Option(False),
    push_to_hub: bool = typer.Option(False),
    personal_token: Optional[str] = typer.Option(None),
) -> Trainer:

    logging.basicConfig(level=logging.INFO)

    dataset = load_dataset("opentargets/clinical_trial_reason_to_stop", split='train').train_test_split(test_size=0.1, seed=42)
    global labels
    labels, id2label, label2id = get_label_metadata(dataset)

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    dataset_cols = [col for col in dataset["train"].column_names if col not in ["text", "input_ids", "attention_mask", "labels"]]
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset_cols)

    train_dataset, test_dataset = prepare_splits_for_training(tokenized_dataset, subset_data)
    logging.info(f"Train dataset length: {len(train_dataset)}")
    logging.info(f"Test dataset length: {len(test_dataset)}")

    train_dataset = (
        tokenized_dataset["train"].shuffle(seed=42).select(range(500)).with_format("torch")
    )
    test_dataset = (
        tokenized_dataset["test"].shuffle(seed=42).select(range(50)).with_format("torch")
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    args = TrainingArguments(
        output_dir=output_model_name,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        data_seed=42,
        num_train_epochs=epochs,
        metric_for_best_model="f1",
        save_total_limit=2,
        save_strategy="no",
        load_best_model_at_end=False,
        push_to_hub=push_to_hub,
        hub_model_id=output_model_name,
        hub_token=personal_token,
        hub_private_repo=False,
    )

    trainer = MultilabelTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)
    predictions = trainer.predict(test_dataset)
    print(predictions)
    trainer.save_model(output_model_name)
    if push_to_hub:
        trainer.push_to_hub()

    return trainer

if __name__ == '__main__':
    main()