# stopReasons
Implementation of a NLP classifier based on [BERT](https://huggingface.co/bert-base-uncased) that assigns one or multiple classes to each of the clinical trial stop reasons.

This provides a structured representation of why a clinical trial has stopped early.

## Usage

1. Create environment and install dependencies.

```bash
poetry env use 3.10.8
poetry install
```

2. Bundle package.

```bash

VERSION_NO=0.1.0

rm -rf ./dist
poetry build
cp ./stop_reasons/*.py ./dist
gsutil cp ./dist/stopreasons-${VERSION_NO}-py3-none-any.whl gs://ot-team/irene/bert/initialisation/
gsutil cp ./utils/initialise_cluster.sh gs://ot-team/irene/bert/initialisation/
```

3. Set up cluster.

```bash
gcloud dataproc clusters create il-stop-reasons \
    --image-version=2.1 \
    --project=open-targets-eu-dev \
    --region=europe-west1 \
    --master-machine-type=n1-highmem-32 \
    --metadata="PACKAGE=gs://ot-team/irene/bert/initialisation/stopreasons-${VERSION_NO}-py3-none-any.whl" \
    --initialization-actions=gs://ot-team/irene/bert/initialisation/initialise_cluster.sh \
    --enable-component-gateway \
    --single-node \
    --max-idle=10m
```

4. Submit job.

```bash
gcloud dataproc jobs submit pyspark ./dist/train.py \
    --cluster=il-stop-reasons \
    --py-files=gs://ot-team/irene/bert/initialisation/stopreasons-${VERSION_NO}-py3-none-any.whl \
    --project=open-targets-eu-dev  \
    --region=europe-west1
```