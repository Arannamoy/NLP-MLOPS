from airflow import DAG
from airflow.decorators import task
from datetime import datetime

with DAG(dag_id="math_sequence_dag_with_taskflow",start_date=datetime(2026,10,10),schedule="@once",catchup=False)as dag:
    # Task 1: Start with initial number

    @task
    def start_number():
        initial_value=10
        print(f"Starting number: {initial_value}")
        return initial_value

    @task
    def add_five(number):
        new_value=number+5
        print(f"Add 5:{number}*2={new_value}")
        return new_value
    
    @task
    def multiply(number):
        new_value=number*5
        print(f"Multiply 5:{number}*5={new_value}")
        return new_value

    @task
    def subtract(number):
        print(f"Subtract 5:{number}-5={number-5}")
        return number-5
    
    @task
    def compute(number):
        print(f"Compute {number}")

    # set task dependencies
    init=start_number()
    add=add_five(init)
    mul=multiply(add)
    sub=subtract(mul)
    com=compute(sub)
    
