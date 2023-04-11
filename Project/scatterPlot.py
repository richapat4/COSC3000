import dash
from dash import Dash, html, dcc
from dash.dependencies import Input, Output,State
import plotly.graph_objs as go
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pycountry 
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Polygon, MultiPolygon
import json
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder


import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

original_df = pd.read_csv("Project\\WhalingData.csv").dropna()

original_df['Nation'].replace({"USSR": "Russia"}, inplace=True)


df = original_df.groupby(['Year','Nation'])['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead'].sum().reset_index()

species = df.iloc[:, 2:10]
print(species.columns.to_list)
# list_colors = px.colors.sequential.haline

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F40
label_array = df['Nation'].unique()
print(label_array)

le = LabelEncoder()

# Fit and transform the labels
numeric_labels = le.fit_transform(df['Nation'])


fig2 = plt.figure(figsize=(14,9))

ax2 = fig2.add_subplot(111, 
                     projection='3d')

array = np.array(df['Gray'])
constant = 0.000001
array = np.clip(array, constant, np.inf)

# Apply the log transform to the array
array = np.log(array)

ax2.scatter(numeric_labels, 
                df['Year'] ,
                array, 
                s=60)

plt.title("3D PCA plot")
plt.show()


def PCA_test(x,y,ax):
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    print(principalComponents)
    principalDf = pd.DataFrame(data = principalComponents
                , columns = ['principal component 1', 'principal component 2','principal component 3'])
    print(principalDf)
    finalDf = pd.concat([principalDf, y], axis = 1)
    finalDf.columns = ['principal component 1', 'principal component 2', 'principal component 3', 'Nation']
    Xax = np.array(principalDf['principal component 1'])
    Yax = np.array(principalDf['principal component 2'])
    Zax = np.array(principalDf['principal component 3'])
    k = 0
    for l in np.unique(y):

        ix=np.where(y==l)
        ax.scatter(Xax[ix], 
                Yax[ix], 
                Zax[ix], 
                s=60,label =label_array[k])
        k+=1
    ax.set_xlabel("PC1", 
                fontsize=12)
    ax.set_ylabel("PC2", 
                fontsize=12)
    ax.set_zlabel("PC3", 
                fontsize=12)

    ax.view_init(30, 125)
    ax.legend()
    plt.title("3D PCA plot")
    plt.show()
    # g = sns.FacetGrid(finalDf, hue = 'Nation', height = 8).map(sns.scatterplot, 'principal component 1', 'principal component 2','principal component 3').add_legend()
    # # g.axes[0,1].set_title("PCA")

fig = plt.figure(figsize=(14,9))
ax = fig.add_subplot(111, 
                     projection='3d')
 

# PCA_test(species, df['Nation'],ax)





# species_colors = {
#     'Fin': 'blue',
#     'Sperm': 'green',
#     'Humpback': 'red',
#     'Sei': 'purple',
#     'Bryde\'s': 'orange',
#     'Minke': 'brown',
#     'Gray': 'gray',
#     'Bowhead': 'black'
# }

# for species, color in species_colors.items():
#     x = df['Nation']
#     y = df[species]
#     plt.scatter(x, y, color=color, label=species)

# # # set plot title and axis labels
# plt.title('Whale Populations by Year')
# plt.xlabel('Year')
# plt.ylabel('Population')

# # add a legend
# plt.legend()
# plt.show()
# # fig = plt.scatter()