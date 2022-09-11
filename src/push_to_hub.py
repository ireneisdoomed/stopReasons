from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

LABEL2ID = {'Another_Study': 0,
 'Business_Administrative': 1,
 'Covid19': 2,
 'Endpoint_Met': 3,
 'Ethical_Reason': 4,
 'Insufficient_Data': 5,
 'Insufficient_Enrollment': 6,
 'Interim_Analysis': 7,
 'Invalid_Reason': 8,
 'Logistics_Resources': 9,
 'Negative': 10,
 'No_Context': 11,
 'Regulatory': 12,
 'Safety_Sideeffects': 13,
 'Study_Design': 14,
 'Study_Staff_Moved': 15,
 'Success': 16}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

if __name__ == '__main__':

    model = TFAutoModelForSequenceClassification.from_pretrained('./models/model_500n_3_epochs_classificator_tf', local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    model.config.label2id = LABEL2ID
    model.config.id2label = ID2LABEL

    repo_url = 'https://huggingface.co/opentargets/stop_reasons_classificator/tree/main'
    commit_message = 'feat: add bert-base-uncased tokenizer'
    model.push_to_hub(repo_url=repo_url, organization="opentargets", commit_message=commit_message)
    tokenizer.push_to_hub(repo_url=repo_url, organization="opentargets", commit_message=commit_message)
