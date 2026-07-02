from flask import Flask,jsonify

app=Flask(__name__)

@app.route("/")
def home():
    return jsonify({"name":"Hello"},200)

app.run()