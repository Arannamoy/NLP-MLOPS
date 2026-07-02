from flask import Flask,request,render_template_string

app=Flask(__name__)


@app.route("/") # url: http://127.0.0.1:5000/?name=hello&age=20
def home():
    name=request.args.get("name",default="Invalid")
    age=request.args.get("age",default=30)
    return render_template_string(f'Query params: [{name}] [{age}]')

app.run()