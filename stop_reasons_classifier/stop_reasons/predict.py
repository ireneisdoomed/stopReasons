from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


if __name__ == '__main__':
    model_name = 'opentargets/stop_reasons_classificator'
    # model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    tokenizer.push_to_hub(model_name, organization="opentargets", commit_message="feat: add bert-base-uncased tokenizer")