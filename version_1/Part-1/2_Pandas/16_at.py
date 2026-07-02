# at & iat faster than loc & iloc

import pandas as pd

data=pd.read_json("data.json")

print(data.at[1,"Name"])