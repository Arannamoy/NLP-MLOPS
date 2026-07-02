from flask import Flask,render_template_string,flash,render_template

app=Flask(__name__)

app.secret_key="secret_key"


@app.route("/")
def home():
    flash("Thank you")
    return render_template("home.html")


app.run()