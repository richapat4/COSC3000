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
import polarPlot



original_df = pd.read_csv("Project\\WhalingData.csv").dropna()
original_df['Nation'].replace({"USSR": "Russia"}, inplace=True)

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
df_1_world_comb.loc[duplicates, 'latitudes'] += 10

area_to_num = {area: i for i, area in enumerate(df_1['Area'].unique())}

df_1['Whaling Location N.o'] = df_1['Area'].map(area_to_num)

colour_scale = ["blue","green","red","yellow","purple", "turquoise","orange","brown","darkblue","magenta","cyan","teal","olive","lavender","lightblue","darkgreen","pink"]

totals = [str(num) for num in df['Total']]
totals_1 = [str(num) for num in df_1['Total']]
totals_2 = [str(num) for num in df_1_world_comb['Total']]

df['Total_str'] = totals
df_1['Total_str'] = totals_1
df_1_world_comb['Total_str'] = totals_2

# scatter_fig = px.scatter_geo(df_area[year], lat=df_area.latitudes, lon=df_area.longitudes,size = "size",hover_name = "Area", hover_data=["Area_num", "Total"],
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

colors = {
    'JPN': '#0073cf',
    'RUS': 'red',
    'DNK': 'green',
    'ISL': 'orange',
    'IDN': 'purple',
    'KOR': '#f95d6a',
    'NOR': 'pink',
    'VCT': 'gray',
    'USA': '#aaff80',
    'PRT': 'cyan',
    'CAN': 'magenta'
}

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



app = Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'backgroundColor': 'black',
    'height': '30px',
    'width': '400px',
    'position':'relative'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'height': '30px',
    'width': '400px',
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
    Output('my-checkbox','style'),
    Output('tab-1','style'),
    Output('tab-2','style'),
    Output('tab-3','style'),
    Output('tab-4','style'),
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
        'width': '400px',
    }

    if n_clicks is None:
        bg_color = 'black'
        font_color = 'white'
        return {'backgroundColor': 'black', 'color': 'white'},h1_style,h2_style,checkbox_style,data
    else:
        if data % 2 == 0:
            bg_color = 'white'
            font_color = 'black'
        else:
            bg_color = 'black'
            font_color = 'white'

        h1_style={'color': font_color}
        h2_style={'color': font_color}
        tab_style= {'backgroundColor': bg_color, 
                    'color':font_color,
                    'borderBottom': '1px solid #d6d6d6',
                    'padding': '6px',
                    'height': '30px',
                    'width': '400px',
        }
        checkbox_style = {
        'color': font_color,
        'labelStyle': {
            'color': font_color
            }
        }
        
        return {'backgroundColor': bg_color, 'color': font_color}, h1_style,h2_style,checkbox_style,tab_style, tab_style, tab_style, tab_style, data

@app.callback(
    Output('world-graph', 'figure'),
    Output('bar-graph', 'figure'),
    Output('heat-map', 'figure'),
    Input('slider', 'value'),
    Input('all-box', 'value'),
    Input('toggle-state', 'data'),
    Input('theme-button', 'n_clicks')
)

def update_graphs_1(selected_year,all, data, n_clicks):
    print(data)
    print(all)
    filtered_df = df[df['Year'] == selected_year]
    filtered_df_1 = df_1[df_1['Year'] == selected_year]
    df_heat_year = pd.DataFrame(df_heatmap[selected_year])
    df_area_breakdown = pd.DataFrame(df_area[selected_year])
    width_h = 500
    height_h = 500
    
    if(data % 2 ==0):
        ex_template ='plotly_white'
    else:
        ex_template = 'plotly_dark'

    if(len(all) == 1):
        filtered_df = df_world_comb
        filtered_df_1 = df_1_world_comb
        filtered_df_2 = df_melted[df_melted['Year'] == selected_year]
        df_heat_year = df_heatmap_total
        df_area_breakdown = df_area_total
        width_h = 800
        height_h = 800
        print('HERE')

    df_heat_year = df_heat_year.pivot(index='Nation', columns='Type', values='Total').fillna(0)
    df_heat_year.columns.name = None
    df_heat_year.index.name = None
    
    df_area_breakdown = df_area_breakdown.melt(id_vars=['Area'], var_name='Whale', value_name='Number')
    df_area_breakdown['Log_scale'] = np.where(df_area_breakdown['Number'] == 0, 0, np.log10(df_area_breakdown['Number']))

    bar_chart2 = go.Figure()
    bar_chart2 = px.bar(df_area_breakdown, x="Log_scale", y="Area", color='Whale', orientation = 'h',title='Whaling Location Breakdown by Area in Year {0}'.format(selected_year),barmode='stack',template = ex_template,color_discrete_sequence = px.colors.sequential.haline)
    if(selected_year == 1985 and (len(all) == 0)):
        bar_chart2.update_traces(width=0.1)
    
    

    # # df_heatmap2 = [df['Whale'] != 'Total'] # remove the Total row
    # df_heat_area = df_heatmap2.pivot(index='Area', columns='Whale', values='Number').fillna(0)
    # df_heat_area.columns.name = None
    # df_heat_area.index.name = None

    point_colors = px.colors.sequential.haline

    fig1 = px.choropleth(filtered_df, geojson="https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json",
                            color_continuous_scale=px.colors.sequential.YlOrRd,                      
                            color = filtered_df.Total,  template=ex_template,
                            locations=filtered_df.CODE, featureidkey="id", hover_name = "Nation",
                             center={"lat": 0, "lon": 0},projection = 'natural earth')
    # fig1.update_layout(
    #     geo=dict(
    #         lonaxis=dict(
    #             range=[-180, 180],
    #             tick0=[-180, -90, 0, 90, 180]
    #         ),
    #         lataxis=dict(
    #             range=[-90, 90],
    #             tick0=[-90, -60, -30, 0, 30, 60, 90]
    #         )
    #     )
    # )
    
    # fig1.update_geos(showocean = True, oceancolor='#f9f9f9')
    
    scatter_fig = px.scatter_geo(filtered_df_1, lat=filtered_df_1.latitudes, lon=filtered_df_1.longitudes,size = "size",hover_name = "Area", hover_data=["Total"], 
                            projection = 'natural earth',color="Nation", color_discrete_sequence = point_colors , animation_group=filtered_df_1.Area,text="Total_str")
    
    scatter_fig.update_traces(marker=dict(size=25,opacity = 0.9, line=dict(width=1, color='white')))
    scatter_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    scatter_fig.update_layout(legend=dict(title='Whaling Locations',font=dict(family='sans-serif',size=14)))
    scatter_fig.update_traces(marker=dict(size=10),selector=dict(mode='legend'))
                              
    scatter_fig.update_traces(textfont=dict(family='sans-serif', size=8, color='white'))
    scatter_fig.update_coloraxes(showscale=True)

    for i in scatter_fig.data:
        fig1.add_trace(i)

    fig1.update_layout(title=f'Countries Involved and Whaling Locations in Year {selected_year}')
    fig1.data[0].legendgroup= scatter_fig.data[0].legendgroup
    fig1.update_layout(coloraxis_colorbar_x=-0.15)
    
    # bar_chart = px.bar(filtered_df_2,x="Species",y="Count",barmode='stack', template=ex_template)
    # bar_chart.update_layout(title=f'Species Lost in Year {selected_year}')
    # bar_chart.update_traces(marker_color='#D34336')
    # # df = px.data.medals_wide(indexed=True)
    heat_map = px.imshow(df_heat_year, color_continuous_scale=px.colors.sequential.YlOrRd, origin='lower',text_auto = True,template=ex_template, width = width_h, height =height_h)
    # heat_map2 = px.imshow(df_heat_area, color_continuous_scale='RdBu_r', origin='lower',text_auto = True,template=ex_template)
    return fig1,bar_chart2,heat_map



@app.callback(
    Output('aggregate-graph', 'figure'),
    Output('box-graph','figure'),
    Input('my-checkbox', 'value'),
    Input('toggle-state', 'data'),
    Input('theme-button', 'n_clicks')
)

def update_graphs_2(selected_nation,n_clicks,data):
    print(selected_nation)
    fig = go.Figure()
    if(data % 2 == 0):
        ex_template = 'plotly_white'
    else:
        ex_template = 'plotly_dark'

    fig.update_layout(template = ex_template)
    # polyFig = go.Figure()
    traces = []
    for value in selected_nation:
        trend = df[df['CODE'] == value].reset_index()
        poly = np.polyfit(trend['Year'], trend['Total'], 5)
        trend['polyFit'] = np.polyval(poly,trend['Year'])
        polyFig = px.scatter(trend, x='Year', y='Total',template = ex_template,color= trend.CODE,color_discrete_map=colors)
        polyFig.add_traces(px.line(trend, x='Year', y='polyFit',template = ex_template,color= trend.CODE,color_discrete_map=colors).data[0])
        traces.append(polyFig)
        # print(traces)
    for trace in traces:
        
        fig.add_traces(trace.data)
        fig.update_layout(template = ex_template)
    list_colors = px.colors.sequential.thermal
    first_eight_colors = list_colors[2:10]
    box_new = box.melt(id_vars=['Year'], var_name='Species', value_name='Number')
    box_new['Log_scale'] = np.where(box_new['Number'] == 0, 0, np.log10(box_new['Number']))
    boxF = px.box(box_new, x='Species', y='Log_scale', color='Species',
             title='Whale Species Population 1985-2018',template = ex_template,color_discrete_sequence=first_eight_colors)

    return fig,boxF


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

    if(len(all) == 1):
        stackedCluster = stacked_total
    # print(stackedCluster)

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
            go.Bar(x=[plot_df.CODE, plot_df.Whale], y=plot_df.log_scale, name=r, marker_color=c,hovertext = r),
        )
        
    stackedBar.update_layout(title=f'Type and Species of Whales,by Country {selected_year_2}',template=ex_template)

    return stackedBar


@app.callback(
   Output('polar-map', 'figure'),
    Input('toggle-state', 'data'),
    Input('slider_3', 'value'),
    Input('theme-button', 'n_clicks')
)

def update_graphs_4(n_clicks,decade_num, data):

    if(data % 2 == 0):
        ex_template = 'plotly_white'
        line_colorr = 'black'
        grid_color = 'lightgray'
    else:
        ex_template = 'plotly_dark'
        line_colorr = 'lightgray'
        grid_color = 'darkgray'

    decade = f'{decade_num}s'

    decade_dataFrame = pd.DataFrame.from_dict(polarPlot.decades_dict[decade], orient='index')
    
    index = list(polarPlot.decades_dict[decade].keys())
    decade_dataFrame['Country'] = index

    # decade_dataFrame['Endangered'] = 
    # Join the datasets based on the common column

    decadeMelted = pd.melt(decade_dataFrame, id_vars=['Country'], var_name='Whale')
    # print(decadeMelted)
    decadeMelted['log_scale']= np.where(decadeMelted['value'] == 0, 0, np.log10(decadeMelted['value']))
    polarBar = go.Figure()

    decadeMelted['RedList'] = decadeMelted['Whale'].map(df_endanger.set_index('Species')[decade])

    list_colors = px.colors.sequential.YlOrRd[2:]
    # categories = ['UN', 'LC', 'NT*', 'VU', 'LC', 'NE', 'EN']
    # colors = ['rgb(254,217,118)', 'rgb(254,178,76)', 'rgb(253,141,60)', 'rgb(252,78,42)', 'rgb(227,26,28)', 'rgb(189,0,38)', 'rgb(128,0,38)']
    # color_map = {category: color for category, color in zip(categories, colors)}
    # color_map = {'UN': 'rgb(254,217,118)', 'LC': 'rgb(227,26,28)', 'NT*': 'rgb(253,141,60)', 'VU': 'rgb(252,78,42)', 'NE': 'rgb(189,0,38)', 'EN': 'rgb(128,0,38)'}

    polarBar = px.bar_polar(decadeMelted, r='log_scale', theta='Country', color='RedList',
                    # color_discrete_sequence=color_map,
                    template=ex_template,
                    labels={'log_scale': 'Number of Whales'},width=800, height=600, hover_data=["Whale", "log_scale"],color_discrete_sequence = list_colors)
    polarBar.update_traces(text=decadeMelted['Whale'],marker=dict(line=dict(width=1, color='black')))
    polarBar.update_polars(radialaxis_gridcolor=grid_color)
    polarBar.update_polars(angularaxis_gridcolor=grid_color)

    # update the layout
    polarBar.update_layout(polar=dict(radialaxis=dict(title='Number of Whales'),
                                angularaxis=dict(direction='clockwise',
                                                rotation=90,
                                                ticks=''),
                                          
                               ),
                    legend=dict(orientation='h',
                                yanchor='bottom',
                                y=-0.2,
                                xanchor='right',
                                x=1),
                    title=dict(text='Whaling over decade {0}, by Species and Country'.format(decade),
                                font=dict(size=24)),
                    margin=dict(t=100, b=100, l=100, r=100))
    
    return polarBar


app.layout = html.Div(id = 'dashboard', children = [ 
    html.Button('Change Theme', id='theme-button', n_clicks= 0),
    dcc.Store(id='toggle-state', data=False),
    html.H1('Whaling Global Data and Statistics', id = 'heading_1'),
    #Put these two divisions into a panel
    dcc.Tabs(id='tabs', value='tab-1',  children=[
        dcc.Tab(id = 'tab-1', label='Tab 1', value='tab-1', selected_style = tab_selected_style,
children=[
            # html.Div([
                html.H3('Aggregate Data', id = 'heading_2'),
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
                        ),style={'width': '80%', 'display': 'inline-block'}),
                    html.Div(dcc.Checklist(
                        id='all-box',
                        options=
                        [{'label': 'ALL', 'value': 'ALL'}],
                        value=[]
                    ), style={'width': '20%', 'display': 'inline-block'}),
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
        dcc.Tab(id = 'tab-2', label='Tab 2', value='tab-2', selected_style = tab_selected_style, children=[
            html.Div([
                
                dcc.Checklist(
                    id='my-checkbox',
                    options=
                    [
                    {'label': i, 'value': i} for i in  CODE_Nation],
                    value=['JPN','RUS','KOR'], style = check_list_style
                ),
                dcc.Graph(id='aggregate-graph'),
                dcc.Graph(id='box-graph')
            ]),
        ]),
        dcc.Tab(id = 'tab-3',label='Tab 3', value='tab-3', selected_style = tab_selected_style, children=[
            html.Div(children = [
                html.Div([
                    html.Div(dcc.Slider(
                        id='slider_2',
                        min=df['Year'].min(),
                        max=df['Year'].max(),
                        value=df['Year'].min(),
                        step = None,
                        marks={i: '{}'.format(i) for i in range(1985, 2018,1)},
                    ),style={'width': '80%', 'display': 'inline-block'}), 
                    
                    html.Div(dcc.Checklist(
                        id='all-box-1',
                        options=
                        [{'label': 'ALL', 'value': 'ALL'}],
                        value=[]
                    ), style={'width': '20%', 'display': 'inline-block'}),

                    dcc.Graph(id='stacked-map')
                    ]),
                ]),
            ]),
            
            dcc.Tab(id = 'tab-4', label='Tab 4', value='tab-4', selected_style = tab_selected_style, children=[
                html.Div([
                    dcc.Slider(
                        id='slider_3',
                        min= 1980,
                        max= 2000,
                        value= 1980,
                        step = None,
                        marks={k: '{}'.format(k) for k in range(1970, 2010,10)},
                    ),
                        html.Div(
                            dcc.Graph(id='polar-map'),
                            style={
                                'width': '50%',
                                'margin': 'auto',
                                'display': 'flex',
                                'align-items': 'center',
                                'justify-content': 'center'
                                }
                        ),
                    ]),
                 ]),
             ]),
        ])

if __name__ == '__main__':
    app.run_server(debug=True)