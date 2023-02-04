import logging

from datasets import load_dataset
from huggingface_hub import login
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, DefaultDataCollator, PushToHubCallback, TFAutoModelForSequenceClassification, TFTrainer, TFTrainingArguments

from stop_reasons.utils import (compute_metrics, convert_to_tf_dataset, get_label_metadata, prepare_splits_for_training)

def main(subset_data: bool, epochs: int, output_path: str, env: str, push_to_hub, personal_token: str) -> None:

    logging.basicConfig(level=logging.INFO)

    logging.info("Loading dataset")
    dataset = load_dataset("opentargets/clinical_trial_reason_to_stop", split='train').train_test_split(test_size=0.1)
    global labels
    labels, id2label, label2id = get_label_metadata(dataset)

    logging.info("Tokenizing dataset")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    encoded_dataset = dataset.map(encode_text, batched=True)

    train_dataset, eval_dataset = prepare_splits_for_training(encoded_dataset, subset_data)
    data_collator = DefaultDataCollator(return_tensors="tf")
    tf_train_dataset = convert_to_tf_dataset(train_dataset, data_collator, shuffle_flag=True, batch_size=32)
    tf_validation_dataset = convert_to_tf_dataset(eval_dataset, data_collator, shuffle_flag=False, batch_size=32)

    logging.info("Loading model")
    model = TFAutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(labels),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    # args = TFTrainingArguments(
    #     "stop_reasons_classificator_multi_label",
    #     evaluation_strategy = "epoch",
    #     save_strategy = "epoch",
    #     learning_rate=5e-5,
    #     # optimizer is adam by default
    #     per_device_train_batch_size=32,
    #     per_device_eval_batch_size=32,
    #     num_train_epochs=epochs,
    #     load_best_model_at_end=True,
    #     metric_for_best_model="f1",
    # )

    # trainer = TFTrainer(
    #     model,
    #     args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     # tokenizer=tokenizer,
    #     compute_metrics=compute_metrics
    # )

    # trainer.train()
    # trainer.evaluate()
    # trainer.push_to_hub(repo_name="stop_reasons_classificator_multi_label")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=tf.keras.metrics.BinaryAccuracy(),
    )

    logging.info("Training model")
    callback = None
    if push_to_hub:
        login(token=personal_token)
        callback = PushToHubCallback(
            output_dir=output_path, tokenizer=tokenizer, hub_model_id="opentargets/stop_reasons_multi_direct"
        )
    model.fit(tf_train_dataset, epochs=epochs, validation_data=tf_validation_dataset, callbacks=callback)

    output_path = output_path if env == "local" else f"gs://ot-team/irene/bert/{output_path}"
    model.save_pretrained(output_path, saved_model=True, save_format="tf")
    tokenizer.save_pretrained(f"{output_path}_tokenizer")
    logging.info("Model and tokenizer saved. Exiting.")

def encode_text(dataset_split):
    """Tokenize texts and align labels with them."""
    text = dataset_split["text"]
    encoding = tokenizer(text, padding="max_length", truncation=True)
    labels_batch = {k: dataset_split[k] for k in dataset_split.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()
    return encoding

if __name__ == "__main__":
    main(
        subset_data=True,
        epochs=1,
        output_path="stop_reasons_classificator_multi_label_test",
        env="local",
        push_to_hub=False,
        personal_token="",
    )