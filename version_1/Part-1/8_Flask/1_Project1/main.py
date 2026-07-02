from flask import Flask,render_template,request,render_template_string

app = Flask(__name__,static_folder="assets",static_url_path="/files")

@app.route("/")
def root():
    return render_template("base.html")




@app.route("/form",methods=["GET","POST"])
def form():
    if request.method=="POST":
        email=request.form["email"]
        variant=request.form["variant"]
        return render_template_string(f'<h1>Your email {email} and a4000 variant {variant}<h1>')
    return render_template("form.html")


app.run(debug=True)