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
import numpy as np
import seaborn as sns
import itertools 
import base64
import dash_bootstrap_components as dbc
import statsmodels.api as sm
import pylab as py
import time
import scipy.stats as stats
import polarPlot
from dash.exceptions import PreventUpdate
from plotly.tools import mpl_to_plotly

from statsmodels.graphics.gofplots import qqplot_2samples

original_df = pd.read_csv("Project\\WhalingData.csv").dropna()
original_df['Nation'].replace({"USSR": "Russia"}, inplace=True)

original_df['Type'] = original_df['Type'].str.title()

locations = pd.read_csv("Project\\Whaling Locations1.csv")


# Group the dataframe by Year and Nation and sum the Total column

#World Map Plot
df = original_df.groupby(['Year','Nation'])['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead', 'Total'].sum().reset_index()

df_world_comb = original_df.groupby(['Nation'])['Total'].sum().reset_index()

#Scatter Plot 
df_1 = original_df.groupby(['Year','Nation','Area'])['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead', 'Total'].sum().reset_index()

df_1_world_comb = original_df.groupby(['Nation','Area'])['Total'].sum().reset_index()


df_3 = original_df.groupby(['Nation'])['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead', 'Total'].sum().reset_index()

#Correlation Map
df_4 = original_df.groupby(['Year','Nation', 'Type'])['Total'].sum().reset_index()

df_heatmap_total = original_df.groupby(['Nation', 'Type'])['Total'].sum().reset_index()

#Horizontal Bar Graph
df_area = original_df.groupby(['Year','Area'])['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead'].sum().reset_index()

df_area_total = original_df.groupby(['Area'])['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead'].sum().reset_index()

#Box plot
box = original_df.groupby(['Year'])['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead'].sum().reset_index()

print(box)

# pivot = box.transpose()
# print(pivot)



# df = px.data.tips()
# print(df)
# fig = px.box(df, y="total_bill")
# fig.show()

# fig = px.box(df, x='S', y='Total', color='Nation')


x = df.groupby(df['Year'],group_keys=True)['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead'].sum().reset_index()


# japan_trend = df.groupby(df['Year','Nation'],group_keys=True)['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead'].sum().reset_index()


df_melted = x.melt(id_vars=['Year'], var_name='Species', value_name='Count')

df3_melted = df_3.melt(id_vars=['Nation'], var_name='Species', value_name='Total')


# fig = px.bar_polar(df, r="frequency", theta="direction",
#                     color="Species", template="ggplot2",
#                     color_discrete_sequence= px.colors.sequential.Plasma_r)
# fig.show()


def create_legend_mapping(df,key1,key2):
    # get unique values in the 'Whale' and 'RedList' columns
    values1 = df[key1].unique()
    values2 = df[key2].unique()

    # create the legend mapping dictionary
    legendmapping = {}
    for val in values1:
        legendmapping[val] = {}
        for val2 in values2:
            legendmapping[val][val2] = f"{val}: {val2}"
    return legendmapping

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

# create a column for code 
df['CODE']=alpha3code(df['Nation'])
df_world_comb['CODE'] = alpha3code(df_world_comb['Nation'])

# create a column for code 
df_1['CODE']=alpha3code(df_1['Area'])
df_1_world_comb['CODE'] = alpha3code(df_1_world_comb['Area'])

original_df['CODE']=alpha3code(original_df['Nation'])

stacked = original_df.groupby(['Year','CODE', 'Type'])['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead'].sum().reset_index()

stacked_total = original_df.groupby(['CODE', 'Type'])['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead'].sum().reset_index()


grouped = df.groupby('Year').agg({
    'Nation': list,
    'Fin': list,
    'Sperm': list,
    'Humpback': list,
    'Sei': list,
    "Bryde's": list,
    'Minke': list,
    'Gray': list,
    'Bowhead': list,
    'Total': list,
    'CODE':list
})

# Convert the result to a dictionary
df_nation = grouped.to_dict(orient='index')

grouped_1 = df_area.groupby('Year').agg({
    'Area': list,
    'Fin': list,
    'Sperm': list,
    'Humpback': list,
    'Sei': list,
    "Bryde's": list,
    'Minke': list,
    'Gray': list,
    'Bowhead': list,
})


df_area = grouped_1.to_dict(orient='index')

# Convert the result to a dictionary


grouped_2 = df_4.groupby('Year').agg({
    'Nation': list,
    'Type': list,
    'Total':list,
})

df_heatmap = grouped_2.to_dict(orient='index')

df_heatmap2 = grouped_1.to_dict(orient='index')


latitudes = []
longitudes = []

latitudes_1 = []
longitudes_1 = []

for d in df_1['Area']:
    for a,b,c in zip(locations['Area'],locations['Latitude'],locations['Longitude']):
        if(a==d):
            latitudes.append(b)
            longitudes.append(c)


for d in df_1_world_comb['Area']:
    for a,b,c in zip(locations['Area'],locations['Latitude'],locations['Longitude']):
        if(a==d):
            latitudes_1.append(b)
            longitudes_1.append(c)


df_1['latitudes'] = latitudes
df_1['longitudes'] = longitudes
df_1['size'] = np.ones(len(df_1['longitudes'])) * 5


df_1_world_comb['latitudes'] = latitudes_1
df_1_world_comb['longitudes'] = longitudes_1
df_1_world_comb['size'] = np.ones(len(df_1_world_comb['longitudes'])) * 5

df_1.to_csv('Project\\GeoMap.csv', index=False)

# check for duplicates in area and year
duplicates = df_1[['Year', 'Area']].duplicated()

# add a small offset to the latitude for duplicates
df_1.loc[duplicates, 'latitudes'] += 10

# check for duplicates in 
duplicates_1 = df_1_world_comb[['Area']].duplicated()
# add a small offset to the latitude for duplicates
df_1_world_comb.loc[duplicates_1, 'latitudes'] += 10

area_to_num = {area: i for i, area in enumerate(df_1['Area'].unique())}

df_1['Whaling Location N.o'] = df_1['Area'].map(area_to_num)

colour_scale = ["blue","green","red","yellow","purple", "turquoise","orange","brown","darkblue","magenta","cyan","teal","olive","lavender","lightblue","darkgreen","pink"]

totals = [str(num) for num in df['Total']]
totals_1 = [str(num) for num in df_1['Total']]
totals_2 = [str(num) for num in df_1_world_comb['Total']]

df['Total_str'] = totals
df_1['Total_str'] = totals_1
df_1_world_comb['Total_str'] = totals_2

# scatter_fig = px.scatter_geo(df_area[year], 
# 
# 
# 
# lat=df_area.latitudes, lon=df_area.longitudes,size = "size",hover_name = "Area", hover_data=["Area_num", "Total"],
#                             color=df_1['Area_num'],color_continuous_scale=colour_scale ,range_color=[0,16], 
#                             projection = 'natural earth',text= "Total_str")

# # scatter_fig.update_layout(mapbox_style="open-street-map")

# scatter_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
# scatter_fig.update_traces(textfont_size=8)
# scatter_fig.update_coloraxes(showscale=False)


area_color_map = {
    "Antarctic": "red",
    "Alaska": "green",
    "Chukotka": "red",
    "E. Greenland": "yellow",
    "Indonesia": "purple",
    "Japan": "pink",
    "NE Atlantic": "orange",
    "Korea": "#f95d6a",
    "W. Iceland": "gray",
    "W.Indies": "magenta",
    "NW Canada": "cyan",
    "NE Canada": "teal",
    "Azores": "olive",
    "NW Pacific": "lavender",
    "Iceland": "lightblue",
    "USA W coast": "#aaff80",
    "W. Greenland": "turquoise"
}

CODE_Nation = df['CODE'].unique()
# print(CODE_Nation)


haline_colors = px.colors.sequential.haline

print(haline_colors[2])
# Define a dictionary mapping country codes to colors from the haline color map
colors = {
    'JPN': haline_colors[0],
    'RUS': haline_colors[1],
    'DNK': haline_colors[2],
    'ISL': haline_colors[3],
    'IDN': haline_colors[4],
    'KOR': '#4482FF',
    'NOR': '#890089',
    'VCT':  'gray',
    'USA': haline_colors[8],
    'PRT': haline_colors[9],
    'CAN': haline_colors[10]
}

# colors = {
#     'JPN': '#0073cf',
#     'RUS': 'red',
#     'DNK': 'green',
#     'ISL': 'orange',
#     'IDN': 'purple',
#     'KOR': '#f95d6a',
#     'NOR': 'pink',
#     'VCT': 'gray',
#     'USA': '#aaff80',
#     'PRT': 'cyan',
#     'CAN': 'magenta'
# }

# japan_trend = df[df['Nation'] == 'Japan'].reset_index()

# print(japan_trend)

# japan_trend = df[df['Nation'] == 'Japan'].reset_index()
# test= px.scatter(japan_trend, x='Year', y='Total')
# test.show()

images_array = ['Project\\640x427-gray-whale.png', 'Project\\640x427-sei-whale.png','Project\\640x4270-fin-whale-v21.png', 'Project\\640x427-Whale-Humpback-markedDW_1.png', 'Project\\640x427-whale_dwarf_sperm02.png', 'Project\\640x427-Whale-Bowhead-markedDW.png', ]

image_1st = 'Project\\free-medal-icon-1369-thumb.png'

image_base64= []

for i in range(len(images_array)):
    image_base64.append(base64.b64encode(open(images_array[i], 'rb').read()))

data = pd.read_csv('Project\\EndangeredLists_2.csv')

df_endanger= pd.DataFrame(data)

# print(df_endanger['1980s'])


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# df_area = grouped_1.to_dict(orient='index')



app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])

tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'backgroundColor': 'black',
    'height': '30px',
    'width': '300px',
    'position':'relative'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#0F4799',
    'height': '30px',
    'width': '300px',
    'color': 'white',
    'padding': '6px',
    'color': 'white',
}

tab_content_style = {
    'margin': '0px',
    'padding': '20px',
}

check_list_style = {
    # 'color': 'white',
    'inputStyle': {
        'margin-right': '5px',
        'background-color': 'white',
        'border': '1px solid white',
        'border-radius': '3px',
        'box-shadow': 'none'
    }
}


options = [{'label': i, 'value': i} for i in CODE_Nation]
options.append({'label': 'Clear Selection', 'value': None})

bg_color = 'black'
font_color = 'white'


# Define the callback function to update the style properties
@app.callback(
    Output('dashboard', 'style'),
    Output('heading_1', 'style'),
    Output('heading_2', 'style'),
    Output('heading_3', 'style'),
    Output('my-checkbox','style'),
    Output('tab-1','style'),
    Output('tab-2','style'),
    Output('tab-3','style'),
    Output('tab-4','style'),
    Output('dropDown1','style'),
    Output('dropDown2','style'),
    Output('toggle-state', 'data'),
    State('toggle-state', 'data'),
    Input('theme-button', 'n_clicks'))
   

def update_style(n_clicks, data):
    # print(n_clicks%2 ==1)
    # print(data)
    # Define the updated colors
    global bg_color, font_color
    h1_style={'color': 'black'}
    h2_style={'color': 'black'}
    checkbox_style = {
        'color': 'white',
        'labelStyle': {
            'color': 'white'
        }
    }

    tab_style = {
        'backgroundColor': 'black',
        'color' : 'black',
        'borderBottom': '1px solid #d6d6d6',
        'padding': '6px',
        'height': '30px',
        'width': '300px',
    }

    drop_style = {
        'backgroundColor': 'black',
        'color' : 'black',
    }

    if n_clicks is None:
        bg_color = 'black'
        font_color = 'white'
        return {'backgroundColor': 'black', 'color': 'white'},h1_style,h2_style,checkbox_style,data
    else:
        if data % 2 == 0:
            bg_color = 'white'
            drop_bg = 'white'
            font_color = 'black'
        else:
            bg_color = 'black'
            drop_bg = 'black'
            font_color = 'white'

        h1_style={'color': font_color}
        h2_style={'color': font_color}
        tab_style= {'backgroundColor': bg_color, 
                    'color':font_color,
                    'borderBottom': '1px solid #d6d6d6',
                    'padding': '6px',
                    'height': '30px',
                    'width': '300px',
        }
        checkbox_style = {
        'color': font_color,
        'labelStyle': {
            'color': font_color
            }
        }
        
        drop_style = {
        'backgroundColor':drop_bg,
        'color':'black',
        'option': {'color': 'black'},
         'single': {'color':'black'},
        'control': {'backgroundColor': bg_color},
        'valueContainer': {
            'color': font_color,
            'backgroundColor': bg_color
        },
        }
        
        return {'backgroundColor': bg_color, 'color': font_color}, h1_style,h2_style,h2_style,checkbox_style,tab_style, tab_style, tab_style, tab_style,drop_style,drop_style,data

@app.callback(
    Output('world-graph', 'figure'),
    Output('bar-graph', 'figure'),
    Output('heat-map', 'figure'),
    Input('slider', 'value'),
    Input('all-box', 'value'),
    Input('animate-box', 'value'),
    Input('toggle-state', 'data'),
    Input('theme-button', 'n_clicks')
)

# def update_graphs_1(selected_year,all, data, n_clicks):
def update_graphs_1(selected_year,all,animate, data, n_clicks):
    # print(data)
    # print(all)

    width_h = 500
    height_h = 500
    point_colors = px.colors.sequential.haline
    point_colors[10] = 'gray'
    titleStr = f'in {selected_year}'
    prev_year = 1985

    if((len(animate) == 1) and (len(all) == 0)):
        filtered_df = df[df['Year'] == selected_year]
        filtered_df_1 = df_1[df_1['Year'] == selected_year]
        df_heat_year = pd.DataFrame(df_heatmap[prev_year])
        df_area_breakdown = pd.DataFrame(df_area[prev_year])
    elif((len(all) == 1) and (len(animate) == 0)):
    # if((len(all) == 1) and (len(animate) == 0)):
        filtered_df = df_world_comb
        filtered_df_1 = df_1_world_comb
        # filtered_df_2 = df_melted[df_melted['Year'] == selected_year]
        df_heat_year = df_heatmap_total
        df_area_breakdown = df_area_total
        width_h = 780
        height_h = 800
        titleStr = f'- Total Across All Years'
        # print('HERE')         
    else:
        filtered_df = df[df['Year'] == selected_year]
        filtered_df = df[df['Year'] == selected_year]
        filtered_df_1 = df_1[df_1['Year'] == selected_year]
        df_heat_year = pd.DataFrame(df_heatmap[selected_year])
        df_area_breakdown = pd.DataFrame(df_area[selected_year])
        prev_year = selected_year

    if(data % 2 ==0):
        ex_template ='plotly_white'
    else:
        ex_template = 'plotly_dark'



    df_heat_year = df_heat_year.pivot(index='Nation', columns='Type', values='Total').fillna(0)
    df_heat_year.columns.name = None
    df_heat_year.index.name = None
    
    df_area_breakdown = df_area_breakdown.melt(id_vars=['Area'], var_name='Whale', value_name='Number')
    df_area_breakdown['Log_scale'] = np.where(df_area_breakdown['Number'] == 0, 0, np.log10(df_area_breakdown['Number']))

    df_area_breakdown.to_csv("Area.csv")
    bar_chart2 = go.Figure()
    # point_colors = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600','#488f31','#569e7a','#87b68d']

    bar_chart2 = px.bar(df_area_breakdown, x="Number", y="Area", color='Whale', orientation = 'h',title=f'Whaling Location Breakdown by Area {titleStr}',barmode='stack',template = ex_template,color_discrete_sequence = point_colors,log_x=True)
    bar_chart2.update_xaxes(title='Number of Whales',
                        showgrid=True,  # show the vertical grid lines
                        gridcolor='darkgray',  # set the color of the vertical grid lines
                        gridwidth=1)

    if(selected_year == 1985 and (len(all) == 0)):
        bar_chart2.update_traces(width=0.1)
    

    # if(len(animate) == 1): #animate world map over time
    #     fig1 = px.choropleth(df, geojson="https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json",
    #                     color_continuous_scale=px.colors.sequential.YlOrRd,                      
    #                     color = df.Total,  template=ex_template,
    #                     locations=df.CODE, featureidkey="id", hover_name = "Nation",
    #                     center={"lat": 0, "lon": 0},projection = 'natural earth',animation_frame=df.Year,
    #                         animation_group=df.Total)


    #     scatter_fig = px.scatter_geo(df_1, lat=df_1.latitudes, lon=df_1.longitudes,size = "size",hover_name = "Area", hover_data=["Total", "longitudes", "latitudes"], 
    #                             hovertemplate =                  
    #                             "<b>%{hover_name}</b><br>" +
    #                            "Total %{Total}<br>" +
    #                            "Longitude: %{longitudes}<br>" +
    #                            "Latitude: %{latitudes}<br>",
    #                     projection = 'natural earth',color="Nation", color_discrete_sequence = point_colors , animation_group=df_1.Area,text="Total_str", animation_frame=df_1.Year)
    #     scatter_fig.update_traces(marker=dict(size=30,opacity = 1.0, line=dict(width=1, color='lightgray')))
    #     scatter_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    #     scatter_fig.update_layout(legend=dict(title='Whaling Locations',font=dict(family='sans-serif',size=16)))
    #     scatter_fig.update_traces(marker=dict(size=20),selector=dict(mode='legend'))
                                
    #     scatter_fig.update_traces(textfont=dict(family='sans-serif', size=16, color='lightgray'))
    #     scatter_fig.update_coloraxes(showscale=True)



    #     for fe, fne in zip(fig1.frames, scatter_fig.frames):
    #         # for t in fne.data:
    #         #     t.update(marker_color= "colors")
    #         fe.update(data = fe.data + fne.data)

    # else:

    fig1 = px.choropleth(filtered_df, geojson="https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json",
                            color_continuous_scale=px.colors.sequential.YlOrRd,                      
                            color = filtered_df.Total,  template=ex_template,
                            locations=filtered_df.CODE, featureidkey="id", hover_name = "Nation",
                            center={"lat": 0, "lon": 0},projection = 'natural earth')

    scatter_fig = go.Figure()
    scatter_fig = px.scatter_geo(filtered_df_1, lat=filtered_df_1.latitudes, lon=filtered_df_1.longitudes,hover_name = "Area", hover_data=["Total", "longitudes", "latitudes"], 
                            projection = 'natural earth',color="Nation", color_discrete_sequence = point_colors , animation_group=filtered_df_1.Area,text="Total_str", custom_data=["Total_str"])

    scatter_fig.update_traces(marker=dict(size=35,opacity = 0.95, line=dict(width=1, color='white')))
    scatter_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    scatter_fig.update_layout(legend=dict(title='Whaling Locations',font=dict(family='sans-serif',size=13)))
    scatter_fig.update_traces(marker=dict(size=30),selector=dict(mode='legend'))
                            
    scatter_fig.update_traces(textfont=dict(family='sans-serif', size=9, color='white'))
    scatter_fig.update_coloraxes(showscale=True)

    for i in scatter_fig.data:
        fig1.add_trace(i)

    fig1.update_layout(title=f'Nations Involved and Whaling Locations {titleStr}',font=dict(family='sans-serif',size=12))
    fig1.data[0].legendgroup= scatter_fig.data[0].legendgroup
    fig1.update_layout(coloraxis_colorbar_x=-0.15)
    fig1.update_layout(coloraxis_colorbar=dict(title=dict(text='Total by Nation')))
    fig1.update_layout(legend=dict(title='Nation Involved in Area',font=dict(family='sans-serif',size=12)))
    
    heat_map = px.imshow(df_heat_year, color_continuous_scale=px.colors.sequential.YlOrRd, origin='lower',text_auto = True,template=ex_template, width = width_h, height =height_h)
    heat_map.update_layout(title=f'Type of Whale and Nation responsible {titleStr}',font=dict(family='sans-serif',size=12))
    heat_map.update_layout(coloraxis_colorbar=dict(x=1.5))
    df_heat_year.to_csv("heatmap.csv")
    if(df_heat_year.shape[1] > 5):
        heat_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        heat_map.update_layout(coloraxis_colorbar=dict(x=1.5, len= 0.5))
        heat_map.update_layout(width = 900, height = 800)
        # heat_map.update_layout(coloraxis_colorbar=dict(x=1.5))
        # heat_map.update_layout(coloraxis_colorbar_x=-0.05)
    return fig1,bar_chart2,heat_map



@app.callback(
    Output('aggregate-graph', 'figure'),
    Output('box-graph','children'),
    Input('my-checkbox', 'value'),
    Input('toggle-state', 'data'),
    Input('theme-button', 'n_clicks')
)

def update_graphs_2(selected_nation,n_clicks,data):
    print(selected_nation)
    fig = go.Figure()
    if(data % 2 == 0):
        ex_template = 'plotly_white'
        img = "Project\\images\\seabornLightBox.png"
    else:
        ex_template = 'plotly_dark'
        img = "Project\\images\\seabornDarkBox.png"

    with open(img, 'rb') as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode('ascii')
        html_image = html.Img(src=f'data:image/png;base64,{image_base64}')

    fig.update_layout(template = ex_template)
    # polyFig = go.Figure()
    traces = []
    for value in selected_nation:
        trend = df[df['CODE'] == value].reset_index()
        polyFig=go.Figure()
        if(value == 'JPN'):
            polyFig = px.scatter(trend, x='Year', y='Total',color= trend.CODE,color_discrete_map=colors,log_y=True,trendline="lowess",trendline_options=dict(frac=0.46))
        elif(value == 'NOR'):
            polyFig = px.scatter(trend, x='Year', y='Total',color= trend.CODE,color_discrete_map=colors,log_y=True,trendline="lowess",trendline_options=dict(frac=0.28))
        elif(value == 'RUS' ):
            polyFig = px.scatter(trend, x='Year', y='Total',color= trend.CODE,color_discrete_map=colors,log_y=True,trendline="lowess",trendline_options=dict(frac=0.23))
        else:
            polyFig = px.scatter(trend, x='Year', y='Total',color= trend.CODE,color_discrete_map=colors,log_y=True,trendline="lowess")
        
        polyFig.data = [t for t in polyFig.data if t.mode == "lines"]
        
        polyFig.add_traces(px.line(trend, x='Year', y='Total',color= trend.CODE,color_discrete_map=colors,log_y=True, symbol = 'CODE',symbol_sequence=['circle'],line_dash_sequence=['dash']).data[0])
       
        traces.append(polyFig)
        # print(traces)
    for trace in traces:
        
        fig.add_traces(trace.data)
        fig.update_layout(template = ex_template)
        fig.update_yaxes(type='log',title="Number of Whales")
        fig.update_layout(title='Whaling over Time by Nation')
        fig.update_xaxes(title= 'Year')

    list_colors = px.colors.sequential.haline
    first_eight_colors = list_colors[2:10]
    # list_colors = px.colors.sequential.haline
    color_t = list_colors[4]
    print(color_t)
    box_new = box.melt(id_vars=['Year'], var_name='Species', value_name='Number')
    # box_new.to_csv("BoxData.csv")
    # box_new['Log_scale'] = np.where(box_new['Number'] == 0, 0, np.log10(box_new['Number']))

    # boxFig = plt.figure(figsize =(10, 7))
    # ax = boxFig.add_subplot(111)
    # ax.set_yscale('symlog')
    # whales = [box['Fin'], box['Sperm'],box['Humpback'],box['Sei'], box['Bryde\'s'], box['Minke'], box['Gray'], box['Bowhead']]
    # # Creating axes instance
    # bp = ax.boxplot(whales)
    # labels = box_new['Species'].unique()
    # ax.set_xticklabels(labels)

    # boxF = mpl_to_plotly(boxFig)
    # boxF.update_layout(template=ex_template,title = f'Species Overall Catch Distribution')
    # boxF.update_traces(marker=dict(color='#0F4799'))

    # boxF = px.box(box_new, x='Species', y='Number', boxmode='group',
    #          title='Whale Species Population Loss 1985-2018',template = ex_template,color_discrete_sequence=['#0F4799'],  log_y=True, width = 900, height = 600)
    # boxF.update_yaxes(title=dict(text = 'Number of Whales'),type='log')

   # set log_y to False to use symlog scale
             # set range_y to control the scale range
    
    return fig,html_image


@app.callback(
    Output('stacked-map', 'figure'),
    Input('slider_2', 'value'),
    Input('all-box-1', 'value'),
    Input('toggle-state', 'data'),
    Input('theme-button', 'n_clicks')
)

def update_graphs_3(selected_year_2,all, data,n_clicks,):

    stackedCluster = stacked[stacked['Year'] == selected_year_2]
    stackedCluster = stackedCluster.drop(columns = ['Year'])
    titleStr = f'in {selected_year_2}'
    if(len(all) == 1):
        stackedCluster = stacked_total
        titleStr = '- Total Across All Years'

    stackedCluster_melted= pd.melt(stackedCluster, id_vars=['Type', 'CODE'], var_name='Whale', value_name='Number')
    df_filtered_stack = stackedCluster_melted[stackedCluster_melted['Number'] != 0]
    df_filtered_stack['log_scale'] = np.log10(df_filtered_stack['Number'])

    if(data % 2 == 0):
        ex_template = 'plotly_white'
    else:
        ex_template = 'plotly_dark'
        
    stackedBar = go.Figure()

    stackedBar.update_layout(
        template="simple_white",
        xaxis=dict(title_text="Nation"),
        yaxis=dict(title_text="Number"),
        barmode="stack",
        bargap = 0.5,
    )
    
    list_colors = px.colors.sequential.haline
    first_six_colors = list_colors[3:9]
    
    colors = ["#2A66DE", "#FFC32B","blue", "green", "yellow", "orange", "magenta", "red"]

    for r, c in zip(df_filtered_stack.Type.unique(), first_six_colors):
        plot_df = df_filtered_stack[df_filtered_stack.Type == r]
        stackedBar.add_trace(
            go.Bar(x=[plot_df.CODE, plot_df.Whale], y=plot_df.Number, name=r, 
                   marker_color=c,hovertext = r),
        )
    
    if(selected_year_2 == 1985 and (len(all) == 0)):
        stackedBar.update_traces(width=0.1)
    else:
        stackedBar.update_yaxes(type='log')

    stackedBar.update_layout(title=f'Type and Species of Whales, by Nation {titleStr}',template=ex_template)
    stackedBar.update_yaxes(title='Number of Whales')

    return stackedBar


@app.callback(
   Output('polar-map1', 'figure'),
   Output('polar-map2', 'figure'),
   Output('polar-map3', 'figure'),
    Output('polar-map4', 'figure'),
    Input('toggle-state', 'data'),
    # Input('slider_3', 'value'),
    Input('theme-button', 'n_clicks')
)

def update_graphs_4(n_clicks, data):

    if(data % 2 == 0):
        ex_template = 'plotly_white'
        line_colorr = 'black'
        grid_color = 'lightgray'
    else:
        ex_template = 'plotly_dark'
        line_colorr = 'lightgray'
        grid_color = 'darkgray'

    figures = []
    
    for decade_num in [1980, 1990, 2000,2008]:
        decade = f'{decade_num}s'

        decade_dataFrame = pd.DataFrame.from_dict(polarPlot.decades_dict[decade], orient='index')
        
        keys = list(polarPlot.decades_dict[decade].keys())
        decade_dataFrame['Nation'] = keys
        decadeMelted = pd.melt(decade_dataFrame, id_vars=['Nation'], var_name='Whale')
        decadeMelted = decadeMelted.sort_values('Nation')
        decadeMelted['RedList'] = decadeMelted['Whale'].map(df_endanger.set_index('Species')[decade])

        # decade_dataFrame['Endangered'] = 
        # Join the datasets based on the common column

        
        # print(decadeMelted)
        # decadeMelted['log_scale']= np.where(decadeMelted['value'] == 0, 0, np.log10(decadeMelted['value']))

        if(decade_num == 2000):
            titleStr = 'Years: 2000-2008'
        elif(decade_num == 2008):
            titleStr = 'Year: 2008'
        else:
            titleStr = f'Decade: {decade}'

        list_colors = px.colors.sequential.YlOrRd[2:]
        categories = ['UN', 'NE', 'LC', 'NT*', 'VU', 'EN']

      
        polarBar = go.Figure()
        polarBar = px.bar_polar(decadeMelted, r='value', theta='Nation', color='RedList',
                        # color_discrete_sequence=color_map,
                        template=ex_template,log_r = True,
                        labels={'value': 'Number of Whales'},width=600, height=320, hover_data=["Whale", "value"],color_discrete_sequence = list_colors,category_orders={'RedList': categories})
        polarBar.update_traces(text=decadeMelted['Whale'],marker=dict(line=dict(width=1, color='black')))
        polarBar.update_polars(radialaxis_gridcolor=grid_color)
        polarBar.update_polars(radialaxis=dict(dtick=1))
        polarBar.update_polars(angularaxis_gridcolor=grid_color,radialaxis_tickangle=-70,angularaxis_tickfont=dict(size=10),radialaxis_tickfont=dict(size=8))
        # update the layout
        # title=dict(text='N.o of Whales', font=dict(size=9))
        polarBar.update_layout(polar=dict(radialaxis=dict(side = 'clockwise'),
                                    angularaxis=dict(direction='clockwise',
                                                    rotation=90,
                                                    ticks='')
                                            
                                ),
                        legend=dict(orientation='v',font=dict(size=10)),
                                    # yanchor='bottom',
                                    # y=-0.2,
                                    # xanchor='right',
                                    # x=-0.15, font=dict(size=10)),
                        title=dict(text=f'{titleStr}',
                                    font=dict(size=14),
                                    y=0.75,
                                    x=0.005),
                        margin=dict(t=40, b=40, l=42, r=30))
    
        figures.append(polarBar)
    return figures[0], figures[1], figures[2], figures[3]





@app.callback(
   Output('qq-plot', 'figure'),
    Input('toggle-state', 'data'),
    Input('dropDown1', 'value'),
    Input('dropDown2', 'value'),
    Input('theme-button', 'n_clicks')
)

def update_graphs_5(n_clicks,species1, species2,data):

    if(data % 2 == 0):
        ex_template = 'plotly_white'
        colour = 'black'
    else:
        ex_template = 'plotly_dark'
        colour = 'white'

    qq_data = original_df.groupby(['Year'])['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead'].sum().reset_index()

    q1 = qq_data[species1]
    q2 = qq_data[species2]

    quantiles1 = sm.ProbPlot(q1)
    print(quantiles1)
    quantiles2 = sm.ProbPlot(q2)


    fig, ax1 = plt.subplots()
    qqplot_2samples(quantiles1, quantiles2, line='45',ax=ax1)
    ax1.set_xlabel(f'{species1} Catch Distribution', color=colour)
    ax1.set_ylabel(f'{species2} Catch Distribution', color=colour)
    # Convert Matplotlib figure to Plotly figure
    plotly_fig = mpl_to_plotly(fig)
    plotly_fig.update_layout(template=ex_template,title = f'Comparison of Catch Data Distribution {species1} and {species2}')
    plotly_fig.update_traces(line=dict(color=px.colors.sequential.haline[4]))

    return plotly_fig


# Define the callback for the animate button
@app.callback(Output('slider', 'value'),
              Input('interval-component', 'n_intervals'))
def animate_slider(n):
    slider_values = [1985, 1986, 1987, 1988, 1989, 1990, 
                     1991, 1992, 1993, 1994, 1995, 1996, 1997, 
                     1998, 1999, 2000, 2001, 2002, 2003, 2004, 
                     2005, 2006, 2007, 2008, 2009, 2010, 2011, 
                     2012, 2013, 2014, 2015, 2016, 2017]
    # Return the slider value at the current index
    return slider_values[n % len(slider_values)]


# Define the callback for starting the animation
@app.callback(Output('interval-component', 'n_intervals'),
              Output('interval-component', 'disabled'),
            #   Input('animate-button', 'n_clicks'),
              Input('all-box', 'value'),
              Input('animate-box', 'value'),
              State('interval-component', 'disabled'),
              State('interval-component', 'n_intervals'))
              
            #   State('animate-state', 'data'))

def start_animation(all,animate,n_intervals,disabled):
    print(all)
    print(animate)
    print(disabled)

    if (len(animate) == 0):
        n_intervals = 0
        disabled= True
        return n_intervals, disabled
    elif (len(all) == 1):
        n_intervals = 0
        disabled= True
        return n_intervals, disabled
    else:
        disabled = False
        return n_intervals + 1, disabled

# # Define the callback for starting the animation
# @app.callback(
#               Output('interval-component', 'disabled'),
#               Input('animate-button', 'n_clicks'),
#               State('interval-component', 'disabled'),
#               State('animate-state', 'data'))

# def stop_animation(data, disabled, n_clicks):
#     print(data)
#     print(n_clicks)
#     if data % 2 == 0:
#         disabled =True
#     else:
#         disabled = False
#         return disabled


app.layout = html.Div(id = 'dashboard', children = [ 
    html.Button('Change Theme', id='theme-button', n_clicks= 0, style={'position': 'absolute', 'top': '0', 'right': '0'}),
    dcc.Store(id='toggle-state', data=False),
     html.Br(),
    html.H1('International Whaling 1985-2018', id = 'heading_1'),
     html.Br(),
    #Put these two divisions into a panel
    dcc.Tabs(id='tabs', value='tab-1',  children=[
        
        dcc.Tab(id = 'tab-1', label='World Map', value='tab-1', selected_style = tab_selected_style,
children=[
         html.Br(), 
            # html.Div([
                html.H4('Geographic Visualisation', id = 'heading_2'),
                 html.Br(), 
                html.Div(children = [
                    html.Div([
                      html.Div(  
                    dcc.Slider(
                        id='slider',
                        min=df['Year'].min(),
                        max=df['Year'].max(),
                        value=df['Year'].min(),
                        step = None,
                        marks={i: '{}'.format(i) for i in range(1985, 2018,1)},
                        ),style={'width': '100%', 'display': 'inline-block', 'fontSize': '12px'}),
                    html.Div(dcc.Checklist(
                        id='all-box',
                        options=
                        [{'label': 'ALL', 'value': 'ALL'}],
                        value=[]
                    ), style={'width': '10%', 'display': 'inline-block','fontSize': '12px'}),
                    #  html.Button('Animate', id='animate-button', n_clicks=0),
                     #  dcc.Store(id='animate-state', data=False),
                    html.Div(dcc.Checklist(
                        id='animate-box',
                        options=
                        [{'label': 'ANIMATE', 'value': 'animate'}],
                        value=[]
                    ), style={'width': '10%', 'display': 'inline-block','fontSize': '12px'}),

                   
                    dcc.Interval(
                        id='interval-component',
                        interval=1500,  # Milliseconds
                        n_intervals=0,
                        disabled=False
                    ),


                    dcc.Graph(id='world-graph'),

                    #  style={'width': '40%', 'display': 'inline-block'}),
                    html.Div(dcc.Graph(id='heat-map'),
                              style={
                                'width': '100%',
                                'margin': 'auto',
                                'display': 'flex',
                                'align-items': 'center',
                                'justify-content': 'center'
                                }),
                    dcc.Graph(id='bar-graph'),
                    # style={'width': '60%', 'display': 'inline-block'}),
                ]),
            ]),
        ]),
        dcc.Tab(id = 'tab-2', label='Aggregate Data', value='tab-2', selected_style = tab_selected_style, children=[
            html.Br(),
            html.Div([
                
                dcc.Checklist(
                    id='my-checkbox',
                    options=
                    [
                    {'label': i, 'value': i} for i in  CODE_Nation],
                    value=['JPN','RUS','KOR'], style = check_list_style
                ),
                dcc.Graph(id='aggregate-graph'),
            ]),
                html.Div([
                    html.Div(dcc.Slider(
                        id='slider_2',
                        min=df['Year'].min(),
                        max=df['Year'].max(),
                        value=df['Year'].min(),
                        step = None,
                        marks={i: '{}'.format(i) for i in range(1985, 2018,1)},
                    ),style={'width': '100%', 'display': 'inline-block','fontSize': '2px'}), 
                    
                    html.Div(dcc.Checklist(
                        id='all-box-1',
                        options=
                        [{'label': 'ALL', 'value': 'ALL'}],
                        value=[]
                    ), style={'width': '10%', 'display': 'inline-block','fontSize':'12px'}),

                    dcc.Graph(id='stacked-map')
                    ]),
        ]),

            dcc.Tab(id = 'tab-3', label='Endangered Species', value='tab-3', selected_style = tab_selected_style, children=[
                html.Br(), 
                html.Div([
                    html.H4('Whaling by Decade, Species Red List and Nation', id = 'heading_3'),
                    html.Br(),    
                #     dcc.Slider(
                #         id='slider_3',
                #         min= 1980,
                #         max= 2000,
                #         value= 1980,
                #         step = None,
                #         marks={k: '{}'.format(k) for k in range(1970, 2010,10)},
                #     ),
                        html.Div(
                            dcc.Graph(id='polar-map1'),
                            style={
                                'width': '50%',
                                 'display': 'inline-block',
                                 'align-items': 'center',
                                'justify-content': 'center'
                                # 'margin': 'auto',
                                # 'display': 'flex',
                                # 'align-items': 'center',
                                # 'justify-content': 'center'
                                }
                        ),
                          html.Div(
                            dcc.Graph(id='polar-map2'),
                            style={
                                'width': '50%', 'display': 'inline-block',
                                 'align-items': 'center',
                                'justify-content': 'center'
                                # 'margin': 'auto',
                                # 'display': 'flex',
                                # 'align-items': 'center',
                                # 'justify-content': 'center'
                                }
                        ),
                         html.Div(
                            dcc.Graph(id='polar-map3'),
                            style={
                                 'width': '50%', 'display': 'inline-block',
                                # 'margin': 'auto',
                                # 'display': 'flex',
                                'align-items': 'center',
                                'justify-content': 'center'
                                }
                        ),
                            html.Div(
                            dcc.Graph(id='polar-map4'),
                            style={
                                'width': '50%', 'display': 'inline-block',
                                # 'margin': 'auto',
                                # 'display': 'flex',
                                'align-items': 'center',
                                'justify-content': 'center'
                                }
                        ),

                    ]),
                 ]),
            
            dcc.Tab(id = 'tab-4', label='Quantile Distribution', value='tab-4', selected_style = tab_selected_style, children=[
                html.Br(),
                html.Div([

                        html.Div(dcc.Dropdown(
                        id ='dropDown1',
                        options=['Fin','Sperm','Humpback','Sei','Bryde\'s','Minke','Gray','Bowhead'],
                        value='Fin'
                        ),style={'width': '50%', 'display': 'inline-block','fontSize': '15px'}),
                        
                        html.Div(dcc.Dropdown(
                        id ='dropDown2',
                        options=['Fin','Sperm','Humpback','Sei','Bryde\'s','Minke','Gray','Bowhead'],
                        value='Sei'
                        ),style={'width': '50%', 'display': 'inline-block','fontSize': '15px'}),

                        html.Div(
                            dcc.Graph(id='qq-plot'),
                            style={
                                'width': '50%',
                                'margin': 'auto',
                                'display': 'flex',
                                'align-items': 'center',
                                'justify-content': 'center'
                                }
                        ),
                        html.Div(id='box-graph',
                                 style={
                                'width': '50%',
                                'margin': 'auto',
                                'display': 'flex',
                                'align-items': 'center',
                                'justify-content': 'center'
                                }),
                    ]),
                 ]),
             ]),
        ])

if __name__ == '__main__':
    app.run_server(debug=True)