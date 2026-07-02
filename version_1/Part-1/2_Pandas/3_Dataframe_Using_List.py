import pandas as pd

data=[["Hello",5],["Hi",2],["Bye",3]]


df=pd.DataFrame(data,columns=["String","String Length"])


print(df)


dictionary_of_list={
    "Hello":["H e l l o".split(" ")],
    "Hi":["H i".split(" ")],
    "Bye":["B y e".split(" ")]
}

df1=pd.DataFrame(dictionary_of_list)
print(df1)


list_2d=[[1,2],[3,4],[5,6]]

print(pd.DataFrame(list_2d,columns=["A","B"]))