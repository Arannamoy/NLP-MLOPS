from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime 


def initial_number(**context):
    context["ti"].xcom_push(key="current_value",value=10)
    print("Initial Number")

def add(**context):
    current_value=context["ti"].xcom_pull(key="current_value",task_ids="initial_number")
    new_value=current_value+5
    context["ti"].xcom_push(key="new_value",value=new_value)
    print("add 5")

def multiply(**context):
    current_value=context["ti"].xcom_pull(key="current_value",task_ids="add")
    new_value=current_value*3
    context["ti"].xcom_push(key="new_value",value=new_value)
    print("multiple 5")

def subtract(**context):
    current_value=context["ti"].xcom_pull(key="current_value",task_ids="multiply")
    new_value=current_value-5
    context["ti"].xcom_push(key="new_value",value=new_value)
    print("subtract 3")

def compute(**context):
    current_value=context["ti"].xcom_pull(key="current_value",task_ids="substract")
    new_value=current_value+5
    print("compute")

with DAG("math_operations",start_date=datetime(2025,10,10),schedule="@daily",catchup=False) as dag:
    initial=PythonOperator(task_id="initial_number",python_callable=initial_number)
    ad=PythonOperator(task_id="add",python_callable=add)
    mul=PythonOperator(task_id="multiply",python_callable=multiply)
    sub=PythonOperator(task_id="subtract",python_callable=subtract)
    com=PythonOperator(task_id="compute",python_callable=compute)
    initial>>ad>>mul>>sub>>com