import pandas as pd

data=pd.read_json("data.json")

print(data.query("Age<30 and Age>28"))

print(data.query("Age<30 and City=='Delhi'")) # for string must be in single or double quote

print(data.query("`Full Stack`=='MERN+Django+SpringBoot'")) # backtick for special character like as space

age=30

print(data.query("Age>= @age")) # @ for variable comparison

# use and,or,not instead of &,|,~ in data.query()