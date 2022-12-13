import logging

from datasets import load_dataset
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, DefaultDataCollator, TFAutoModelForSequenceClassification, TFTrainer, TFTrainingArguments

from utils import (compute_metrics, convert_to_tf_dataset, get_label_metadata, subset_split)

def main():

    logging.basicConfig(level=logging.INFO)

    logging.info("Loading dataset")
    dataset = load_dataset("opentargets/clinical_trial_reason_to_stop", split='train').train_test_split(test_size=0.1)
    global labels
    labels, id2label, label2id = get_label_metadata(dataset)

    logging.info("Tokenizing dataset")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    encoded_dataset = dataset.map(preprocess_data, batched=True)

    small_train_dataset = subset_split(encoded_dataset, "train", n_samples=200) # .with_format("tf")
    small_eval_dataset = subset_split(encoded_dataset, "test", n_samples=200) # .with_format("tf")
    # train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(500))
    # eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(500))

    logging.info("Loading model")
    # data_collator = DefaultDataCollator(return_tensors="tf")
    # tf_train_dataset = convert_to_tf_dataset(small_train_dataset, data_collator, shuffle_flag=True, batch_size=32)
    # tf_validation_dataset = convert_to_tf_dataset(small_eval_dataset, data_collator, shuffle_flag=False, batch_size=32)

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
    #     learning_rate=5e-5, # optimizer is adam by default
    #     per_device_train_batch_size=32,
    #     per_device_eval_batch_size=32,
    #     num_train_epochs=1,
    #     load_best_model_at_end=True,
    #     metric_for_best_model="f1",
    # )

    # trainer = TFTrainer(
    #     model,
    #     args,
    #     train_dataset=small_train_dataset,
    #     eval_dataset=small_eval_dataset,
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
    model.fit(tf_train_dataset, epochs=1, validation_data=tf_validation_dataset)

    model.save_pretrained('models/model_all_10_epochs_classificator_tf_multi_label', saved_model=True, save_format='tf')
    logging.info("Model saved. Exiting.")

def preprocess_data(examples):
    text = examples["text"]
    encoding = tokenizer(text, padding="max_length", truncation=True)
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    # add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()
    
    return encoding

if __name__ == "__main__":
    main()