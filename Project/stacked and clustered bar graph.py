import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import pandas as pd 
import pycountry

def alpha3code(column):
    CODE=[]
    for country in column:
        try:
            # print(country)
            if(country == "USSR"):
                
                country == "USSR, Union of Soviet Socialist Republics"
                code=pycountry.countries.get(alpha_3='RUS').alpha_3
            elif(country == "St.Vincent & Grenadines"):
                code = pycountry.countries.get(name="Saint Vincent and the Grenadines").alpha_3
            elif(country == "Korea"):
                code = pycountry.countries.get(name='Korea, Republic of').alpha_3
            elif(country == "Russia"):
                 code = pycountry.countries.get(name='Russian Federation').alpha_3
            elif(country == "USA"):
               code = pycountry.countries.get(alpha_3='USA').alpha_3
            else:
                code=pycountry.countries.get(name=country).alpha_3

            CODE.append(code)
        except:
            CODE.append('None')
    return CODE





original_df = pd.read_csv("Project\\WhalingData.csv").dropna()

# create a column for code 
original_df['CODE']=alpha3code(original_df['Nation'])


# # grouped_data = []

# # for row in data:
# #     whale_counts = row[2:11]
# #     for i, count in enumerate(whale_counts):
# #         if count != 0:
# #             whale_type = data[0][i+2]
# #             grouped_data.append([row[1], row[0], whale_type, count])

# # for group in grouped_data:
# #     print(group[0], ",", group[1], ",", group[2], ",", group[3])

# # yearStack = pd.DataFrame(data, columns=['Nation', 'Type', 'Whale', 'Number'])
# # print(df)