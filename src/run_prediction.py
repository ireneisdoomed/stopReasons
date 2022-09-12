import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from utils import prepare_model_from_hub

def save_spark_nlp_classifier(classifier, model_name:str) -> None:
    classifier.write().overwrite().save(f"models/{model_name}_spark_nlp")

def main(spark_instance, model_name):
    
    # Save and format model from HuggingFace Hub to be ingested by Spark NLP
    # prepare_model_from_hub(model_name)
    
    classifier = (
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
    save_spark_nlp_classifier(classifier, model_name)

    # df = spark_instance.read.json('data/cttv008-25-08-2022.json.gz')

    

if __name__ == "__main__":

    spark = sparknlp.start()

    main(spark_instance=spark, model_name='opentargets/stop_reasons_classificator')


