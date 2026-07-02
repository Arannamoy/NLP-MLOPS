### Flask with minimal requirement



### API Base: http://127.0.0.1
### API endpoints:
- /home
- /about
- /index
- /static/script.js


## Use JS,CSS file in flask.

- 1: Create file in `static` folder.
- 2: Link it from `templates` folder.  

## Changing the Static Folder and URL Path

```py
app = Flask(
    __name__,
    static_folder='assets',           # Physical folder on disk
    static_url_path='/files'          # URL path to access those files
)
```

```css
<link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
```

## GET, POST

```py
@app.route("/form",methods=["GET","POST"])
def form():
    if request.method=="POST":
        email=request.form["email"]
        variant=request.form["variant"]
        return render_template_string(f'<h1>Your email {email} and a4000 variant {variant}<h1>')
    return render_template("form.html")
```


```html
<form action="\form" method="post">
        <input type="email" name="email" placeholder="Email">
        <br>
        Select  A4000 Variant
        <br>
        <select name="variant" id="">
        <option value="us-variant">US Variant</option>
        <option value="non-us-variant">Non US Variant</option>
        </select>
        <br>
        <input type="submit" value="Submit">
    </form> 
```



## Dynamic Value, HTML Tag rendering

```py
@app.route("/")
def root():
    word1="Hello"
    word2="World"
    html_tag=f'<form action="\form" method="post"><input type="email" name="email" placeholder="Email"><br>Select  A4000 Variant<br><select name="variant" id=""><option value="us-variant">US Variant</option><option value="non-us-variant">Non US Variant</option></select><br><input type="submit" value="Submit"></form>'
    return render_template("base.html",word1=word1,word2=word2,html_tag=html_tag)
```

```html
<body>
    {{word1}} {{word2}}
    <br>
    {{html_tag | safe}}     <!-- For HTML Tag Rendering  -->

</body>
</html>
```

## Template inheritance

```html
{% extends 'base.html' %}
```