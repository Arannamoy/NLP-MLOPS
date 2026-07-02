from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime


def hello():
    print("helloooo............")


with DAG(f"just_for_practice_1",start_date=datetime(2025,10,10),schedule="@daily") as dag:
    hello=PythonOperator(task_id=f"hello",python_callable=hello)
    hello