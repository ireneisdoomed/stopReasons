
from pathlib import Path
from pyspark.ml import Pipeline
import pyspark.sql.functions as F
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *

from utils import prepare_model_from_hub

def save_spark_nlp_classifier(classifier, model_name:str) -> None:
    classifier.write().overwrite().save(f"models/{model_name}_spark_nlp")

def load_spark_nlp_classifier(model_name:str, spark_instance):
    return (
        BertForTokenClassification
        .loadSavedModel(
            f"models/{model_name}/saved_model/1",
            spark_instance
        )
        .setInputCols(["document", "token"])
        .setOutputCol("prediction")
        .setCaseSensitive(False)
        .setMaxSentenceLength(512)
    )

def prepare_spark_nlp_classifier(model_name:str, model_dir: str, spark_instance):

    classifier = load_spark_nlp_classifier(model_name, spark_instance)
    
    # export the model if it has not been exported yet
    spark_model_path = f"{model_dir}/{model_name}_spark_nlp"
    if not Path(spark_model_path).is_dir():
        save_spark_nlp_classifier(classifier, model_name)
    
    return classifier

def main(model_name, model_dir, evidence_path, spark_instance):
    
    # Save and format model from HuggingFace Hub to be ingested by Spark NLP
    prepare_model_from_hub(model_name, model_dir)
    spark_classifier = prepare_spark_nlp_classifier(model_name, model_dir, spark_instance)

    # Load evidence
    chembl_evd = spark_instance.read.json(evidence_path).filter(F.col("studyStopReason").isNotNull()).limit(100)

    # Prepare the pipeline
    document_assembler = (
        DocumentAssembler()
        .setInputCol('studyStopReason')
        .setOutputCol('document')
    )
    tokenizer = (
        Tokenizer()
        .setInputCols(['document'])
        .setOutputCol('token')
    )
    pipeline = Pipeline(stages=[
        document_assembler,
        tokenizer,
        spark_classifier
    ])

    # Run the pipeline
    pipeline_model = pipeline.fit(chembl_evd)
    annotated_df = pipeline_model.transform(chembl_evd)
    print(annotated_df.printSchema())
    annotated_df.write.parquet('data/annotated_df.parquet')


if __name__ == "__main__":

    spark = sparknlp.start()

    main(model_name='opentargets/stop_reasons_classificator', model_dir='models', evidence_path='data/cttv008-25-08-2022.json.gz', spark_instance=spark)


