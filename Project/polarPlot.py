
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

# long_df = px.data.medals_long()

# # print(long_df)
# # fig = px.bar(long_df, x="nation", y="count", color="medal", title="Long-Form Input")
# # fig.show()

# Create a dictionary to hold the data
decades_dict = {}

# Iterate through each decade (1980s, 1990s, 2000s)
for decade in range(1980, 2020, 10):

    # Filter the dataframe to only include rows from the current decade
    decade_df = original_df_2[(original_df_2['Year'] >= decade) & (original_df_2['Year'] < decade+10)]

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

print(list(decades_dict['1980s'].keys()))