from flask import Flask
import os

app=Flask(__name__)

@app.route("/",methods=["Get"])
def home():
    return f"You successflully run this by docker"


if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)