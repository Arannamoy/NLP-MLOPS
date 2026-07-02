from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime


## Define task 1

def preprocess_data():
    print("Preprocess data")

## Define task 2

def train_model():
    print("training model")


## define task 3

def evaluate_model():
    print("evaluate model")


## define dag

with DAG("ml_pipeline",start_date=datetime(2024,1,1),schedule="@weekly") as dag:
    # define the task
    preprocess=PythonOperator(task_id="preprocess_task",python_callable=preprocess_data)
    train=PythonOperator(task_id="train_task",python_callable=train_model)
    evaluate=PythonOperator(task_id="evaluate",python_callable=evaluate_model)


    ## set dependencies
    preprocess >> train >> evaluate