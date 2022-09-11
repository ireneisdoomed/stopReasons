import logging

from datasets import load_dataset
import tensorflow as tf
from transformers import AutoTokenizer, DefaultDataCollator, TFAutoModelForSequenceClassification

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def subset_split(dataset, split, n_samples):
    return dataset[split].shuffle(seed=42).select(range(n_samples))

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

def main():

    logging.basicConfig(level=logging.INFO)

    logging.info("Loading dataset")
    dataset = load_dataset("opentargets/clinical_trial_reason_to_stop", split='train').train_test_split(test_size=0.1)

    logging.info("Tokenizing dataset")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    small_train_dataset = subset_split(tokenized_datasets, "train", n_samples=500)
    small_eval_dataset = subset_split(tokenized_datasets, "test", n_samples=500)

    logging.info("Loading model")
    data_collator = DefaultDataCollator(return_tensors="tf")
    tf_train_dataset = convert_to_tf_dataset(small_train_dataset, data_collator, shuffle_flag=True, batch_size=32)
    tf_validation_dataset = convert_to_tf_dataset(small_eval_dataset, data_collator, shuffle_flag=False, batch_size=32)

    model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=17)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy(),
    )

    logging.info("Training model")
    model.fit(tf_train_dataset, epochs=3, validation_data=tf_validation_dataset)

    model.save_pretrained('models/model_500n_3_epochs_classificator_tf', saved_model=True, save_format='tf')
    logging.info("Model saved. Exiting.")

if __name__ == "__main__":
    main()