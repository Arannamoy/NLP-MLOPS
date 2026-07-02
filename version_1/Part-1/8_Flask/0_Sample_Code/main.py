from flask import Flask,render_template

app = Flask(__name__,static_folder="assets",static_url_path="/files")

@app.route("/")
def hello_world():
    return "<p>Hello, World4!</p>"


@app.route("/about")
def about():
    return "<h>About Us</h1>"


@app.route("/index")
def index():
    return render_template("index.html")

app.run(debug=True)