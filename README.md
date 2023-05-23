# stopReasons
Implementation of a NLP classifier based on [BERT](https://huggingface.co/bert-base-uncased) that assigns one or multiple classes to each of the clinical trial stop reasons.

This provides a structured representation of why a clinical trial has stopped early.

## Usage

1. Create environment and install dependencies.

```bash
conda create -n stopReasons  python=3.7 -y -c conda-forge --file requirements.txt
conda activate stopReasons
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

2. Prepare input data.

```python
# from OT ChEMBL evidence
evd = 'http://ftp.ebi.ac.uk/pub/databases/opentargets/platform/latest/output/etl/parquet/evidence/sourceId%3Dchembl/'

studies = (
    evd
    .filter(F.col('studyStopReason').isNotNull())
    .withColumn('urls', F.explode('urls'))
    .filter(F.col('urls.niceName').contains('ClinicalTrials'))
    .withColumn('nct_id', F.element_at(F.split(F.col('urls.url'), '/'), -1))
    .select('nct_id', F.col('studyStopReason').alias('why_stopped'))
    .distinct()
)

studies.coalesce(1).write.csv('data/studies.tsv', sep='\t', header=True)

# schema of the input data
studies.printSchema()
>> root
    |-- nct_id: string (nullable = true)
    |-- why_stopped: string (nullable = true)
```
3. Categorize text.

```bash
export MODEL=gs://ot-team/olesya/bert_trials

python predict.py \
    --input_file studies.tsv \
    --model $MODEL \
    --output_file trials_predictions.tsv

# schema of the output data
predictions.printSchema()
>> root
    |-- nct_id: string (nullable = true)
    |-- subclasses: array (nullable = true)
    |    |-- element: string (containsNull = true)
    |-- superclasses: array (nullable = true)
    |    |-- element: string (containsNull = true)
    |-- why_stopped: string (nullable = true)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
