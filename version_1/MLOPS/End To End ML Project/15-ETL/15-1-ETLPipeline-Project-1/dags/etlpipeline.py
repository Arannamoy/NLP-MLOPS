from airflow import DAG
from airflow.providers.http.operators.http import HttpOperator
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
import datetime
import json

## Define the dag

with DAG(dag_id="nasa_apod_postgres",start_date=datetime.datetime(2026,10,10),schedule="@daily",catchup=False)as dag:
    # step 1: create a table if it does not exist
    @task
    def create_table():
        # initialize postgres hook
        postgres_hook=PostgresHook(postgres_connection_id="my_postgres_connection")
        # sql query to create the table
        create_table_query="""
        CREATE TABLE IF NOT EXIST apod_data(
            id SERIAL PRIMARY KEY,
            title VARCHAR(150),
            explanation TEXT,
            url TEXT,
            date DATE,
            media_type VARCHAR(150)
        )
        """
        postgres_hook.run(create_table_query)

    # step 2: extract the nasa api data(APOD -> Astronomy Picture Of the Day)
    extract_apod=HttpOperator(task_id="extract",
                                    http_conn_id="nasa_api", ## connection ID defined In Airflow For NASA API
                                    endpoint='planetary/apod', ## NASA API endpoint for APOD
                                    method="GET",
                                    data={"api_key":"{{conn.nasa_api.extra_dejson.api_key}}"},## Use the api key from connection
                                    response_filter=lambda response:response.json(), ## convert response to json
                                    ) 
    ### api -> api.nasa.gov/planetary/apod?api_key=api key
    
    # setp 3: tranform the data (Pick the information that need to save)
    @task
    def transform_apod_data(response):
        apod_data={
            'title':response.get("title",),
            'explanation':response.get("explanation",""),
            'url':response.get("url",""),
            "data":response.get("date",""),
            "media_type":response.get("media_type","")
        }
        return apod_data
    # step 4: load the data into postgres
    @task
    def load_data_to_postgres(apod_data):
        postgres_hook=PostgresHook(postgres_connection_id="my_postgres_connection")
        insert_query="""
        INSERT INTO apod_data(title,explanation,url,data,media_type)
        VALUES (%s,%s,%s,%s,%s)
        """ 
        ## execute the query
        postgres_hook.run(insert_query,parameters=(
            apod_data['title'],
            apod_data['explanation'],
            apod_data['url'],
            apod_data['date'],
            apod_data['media_type']
        ))
    # step 5: verify the data DBViewer
    # step 6: define the task dependencies
    # extract
    create_table () >> extract_apod 
    api_response=extract_apod.output
    # transform
    transform_data=transform_apod_data(api_response)
    # load
    load_data_to_postgres(transform_data)



