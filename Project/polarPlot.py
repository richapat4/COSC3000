
import pandas as pd 
import matplotlib.pyplot as plt
import pycountry 
import geopandas as gpd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import pandas as pd

# load the csv data
original_df_2 = pd.read_csv("Project\\WhalingData.csv").dropna()
original_df_2['Nation'].replace({"USSR": "Russia"}, inplace=True)
original_df_2['Type'] = original_df_2['Type'].str.title()

# long_df = px.data.medals_long()

# # print(long_df)
# # fig = px.bar(long_df, x="nation", y="count", color="medal", title="Long-Form Input")
# # fig.show()

# Create a dictionary to hold the data
decades_dict = {}

# Iterate through each decade (1980s, 1990s, 2000s)
for decade in [1980, 1990, 2000, 2008]:

    # Filter the dataframe to only include rows from the current decade
    if decade == 2000:
        decade_df = original_df_2[(original_df_2['Year'] >= decade) & (original_df_2['Year'] < decade+9)] # Years 2000 to 2008
    elif decade == 2008:
         decade_df = original_df_2[original_df_2['Year'] ==decade] #Year 2008 alone
    else:
        decade_df = original_df_2[(original_df_2['Year'] >= decade) & (original_df_2['Year'] < decade+10)] #1980s and 1990s

    # Create a dictionary to hold the country-specific data for this decade
    country_dict = {}

    # Iterate through each country in the current decade
    for country in decade_df['Nation'].unique():
        # Filter the dataframe to only include rows for the current country
        country_df = decade_df[decade_df['Nation'] == country]
        # Calculate the total number of each whale species for the current country
        whale_counts = {}
        for species in ['Fin', 'Sperm', 'Humpback', 'Sei', "Bryde's", 'Minke', 'Gray', 'Bowhead']:
            whale_counts[species] = country_df[species].sum()
        # Add the whale counts to the dictionary for the current country
        country_dict[country] = whale_counts
    # Add the country-specific data to the dictionary for the current decade
    decades_dict[f"{decade}s"] = country_dict

print(list(decades_dict['2000s'].keys()))


haline_colors = px.colors.sequential.haline

red_cols = px.colors.sequential.YlOrRd[2:]

['#29186B', '#2A23A0', '#0F4799', '#12618E', '#26748A', '#358888', '#419D85', '#51B27C', '#6FC66B', '#A0D65B', '#D4E172', '#FDEE99']

print(haline_colors)
print(red_cols)