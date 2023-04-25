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
import matplotlib as mpl

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

# plt.title("3D PCA plot")
# plt.show()


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
    plt.title("PCA Plot of Catch Distribution by Nation")
    plt.show()
    # g = sns.FacetGrid(finalDf, hue = 'Nation', height = 8).map(sns.scatterplot, 'principal component 1', 'principal component 2','principal component 3').add_legend()
    # # g.axes[0,1].set_title("PCA")

# fig = plt.figure(figsize=(14,9))
# ax = fig.add_subplot(111, 
#                      projection='3d')
 

# PCA_test(species, df['Nation'],ax)




box = original_df.groupby(['Year'])['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead'].sum().reset_index()

box_new = box.melt(id_vars=['Year'], var_name='Species', value_name='Number')


# fig = plt.figure(figsize =(10, 7))
# ax = fig.add_subplot(111)
# ax.set_yscale('symlog')



# sns.set_theme(style="ticks")
sns.set(style="ticks", context="talk")
plt.style.use("ggplot")

plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['grid.color'] = 'white'
plt.rcParams['figure.facecolor'] = 'black'

plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5})
plt.rcParams.update({'font.size': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11})


plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

qualitative_colors = sns.color_palette("RdBu", 10)

t_color = 'white'

f, ax = plt.subplots(figsize=(12, 10))
# ax.set_ylim([0, 10000])
ax.set_yscale("symlog",linthresh=1)
ax.yaxis.grid(True)
ax.spines['left'].set_linewidth(0.5)


sns.boxplot(
    data=box_new, x="Species", y="Number",
    flierprops={"marker": "o", "markerfacecolor": (0.2, 0.4, 0.6, 0.6), "markeredgecolor": t_color,"linewidth":0.5},
    boxprops={"facecolor": (0.2, 0.4, 0.6, 0.5), "edgecolor":t_color, "linewidth":0.5},
    medianprops={"color": t_color, "linewidth":0.5},
    whiskerprops={"color": t_color, "linewidth":0.5},
    capprops={"color": t_color, "linewidth":0.5},
)

sns.despine(trim=True, bottom=True)

ax.set_xlabel("Species", color= t_color)
ax.set_ylabel("Number of Whales",color= t_color)

plt.show()



# sns.boxplot(x="Species", y="Number", data=box_new)
# sns.despine(left=True)



# whales = [box['Fin'], box['Sperm'],box['Humpback'],box['Sei'], box['Bryde\'s'], box['Minke'], box['Gray'], box['Bowhead']]
# # Creating axes instance
# bp = ax.boxplot(whales,patch_artist=True)
# sty = 'seaborn-v0_8'
# mpl.style.use(sty)

# for patch in (bp['boxes']):
#     patch.set_facecolor('blue')
#     patch.set_alpha(0.5)

# for median in bp['medians']:
#     median.set_color('black')

# labels = box_new['Species'].unique()
# ax.set_xticklabels(labels)



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



# # Generate some sample data
# x = np.linspace(-10, 10, 101)
# y = np.sin(x)

# # Create the figure object
# fig = go.Figure(go.Scatter(x=x, y=y))

# # Set the y-axis to have both logarithmic and linear scales
# fig.update_layout(yaxis_type='log',
#                   yaxis_range=[-3, 1],
#                   yaxis_tickformat='.2e')

# # Set the y-axis scale for values less than or equal to zero to 'linear'
# fig.update_yaxes(type='linear', range=[-1, 0])

# # Set the axis labels and title
# fig.update_layout(title='Dual Scale Y-Axis Example',
#                   xaxis_title='X-axis',
#                   yaxis_title='Y-axis')

# fig.show()




# # define the linear and log scales
# linear_scale = go.layout.Linear(scaleanchor='y', scaleratio=1)
# log_scale = go.layout.Log(scaleanchor='y', scaleratio=0.2)



data = np.array([
    [0, 0, 0, 0, 0, 1941, 0, 0],
    [0, 0, 0, 0, 0, 3028, 0, 0],
    [9, 0, 0, 0, 0, 145, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0],
    [76, 0, 0, 40, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 379, 0, 0]
])

domFrac = 1.0/3.0 # distance from 0 to 1 same as 10 to 100


fig = go.Figure()
fig.add_trace(
    go.Box(y=data[2],yaxis ='y1', x = [1])
)
fig.add_trace(
    go.Box(y=data[2], yaxis ='y2',x = [1])
)


# create the figure layout
fig.update_layout(
    yaxis=dict(
        domain= [0,0.5],
        range=[-1.0,0.999],
        scaleanchor="x"
    ),
    yaxis2=dict(
        domain= [0.5,1],
        type='log',
        range=[0, 4],
        dtick=1,
        showticklabels=True,      
        scaleanchor="x"
    ),
     width=800,
    height=500,
    margin=dict(l=50, r=50, t=50, b=50)
)

# Adjust the positions of the box plots
fig.update_layout(boxmode='group')

fig.show()