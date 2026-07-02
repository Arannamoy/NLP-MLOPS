import pandas as pd



# Dataframe is basically dictonary of series.

data={
    "Airlines":["Delta","Singapore","Thai","Qatar","Etihad","Indigo","Eastern China","Air France"],
    "Destination":["New York","Sydney","Tokyo","New Jersey","San Jose","California","San Francisco","Los Angles"],
    "Flight-Models":["B-787","A-350","A-350","A-380","A-350","A-350","A-350","A-350"]
}

df=pd.DataFrame(data) # All arrays must be of the same length

print(df)



print(df.index) # Print all index

print(df.columns) # Print column names



