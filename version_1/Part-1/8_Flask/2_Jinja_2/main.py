from flask import Flask,render_template

app=Flask(__name__)


@app.route("/")
def root():
    word1="Hello"
    word2="World"
    html_tag=f'<form action="\form" method="post"><input type="email" name="email" placeholder="Email"><br>Select  A4000 Variant<br><select name="variant" id=""><option value="us-variant">US Variant</option><option value="non-us-variant">Non US Variant</option></select><br><input type="submit" value="Submit"></form>'
    return render_template("base.html",word1=word1,word2=word2,html_tag=html_tag)


app.run()