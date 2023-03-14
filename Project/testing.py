from dash import Dash, html, dcc
from dash.dependencies import Input, Output
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



original_df = pd.read_csv("Project\\WhalingData.csv").dropna()
locations = pd.read_csv("Project\\Whaling Locations1.csv")




# Group the dataframe by Year and Nation and sum the Total column
df = original_df.groupby(['Year','Nation'])['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead', 'Total'].sum().reset_index()

df_1 = original_df.groupby(['Year','Area'])['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead', 'Total'].sum().reset_index()

df_3 = original_df.groupby(['Nation'])['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead', 'Total'].sum().reset_index()

df_4 = original_df.groupby(['Year','Nation', 'Type'])['Total'].sum().reset_index()

print(df_4)


test_1 = px.data.wind()

print(test_1)

x = df.groupby(df['Year'],group_keys=True)['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead'].sum().reset_index()


# japan_trend = df.groupby(df['Year','Nation'],group_keys=True)['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead'].sum().reset_index()


df_melted = x.melt(id_vars=['Year'], var_name='Species', value_name='Count')

df3_melted = df_3.melt(id_vars=['Nation'], var_name='Species', value_name='Total')


# fig = px.bar_polar(df, r="frequency", theta="direction",
#                     color="Species", template="ggplot2",
#                     color_discrete_sequence= px.colors.sequential.Plasma_r)
# fig.show()


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

# create a column for code 
df_1['CODE']=alpha3code(df_1['Area'])


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

grouped_1 = df_1.groupby('Year').agg({
    'Area': list,
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
df_area = grouped_1.to_dict(orient='index')


grouped_2 = df_4.groupby('Year').agg({
    'Nation': list,
    'Type': list,
    'Total':list,
})

df_heatmap = grouped_2.to_dict(orient='index')



latitudes = []
longitudes = []

for d in df_1['Area']:
    for a,b,c in zip(locations['Area'],locations['Latitude'],locations['Longitude']):
        if(a==d):
            latitudes.append(b)
            print(b)
            longitudes.append(c)


df_1['latitudes'] = latitudes
df_1['longitudes'] = longitudes
df_1['size'] = np.ones(len(df_1['longitudes'])) * 5

area_to_num = {area: i for i, area in enumerate(df_1['Area'].unique())}

df_1['Whaling Location N.o'] = df_1['Area'].map(area_to_num)

colour_scale = ["blue","green","red","yellow","purple", "turquoise","orange","brown","darkblue","magenta","cyan","teal","olive","lavender","lightblue","darkgreen","pink"]

totals = [str(num) for num in df['Total']]
totals_1 = [str(num) for num in df_1['Total']]

df['Total_str'] = totals
df_1['Total_str'] = totals_1


# scatter_fig = px.scatter_geo(df_area[year], lat=df_area.latitudes, lon=df_area.longitudes,size = "size",hover_name = "Area", hover_data=["Area_num", "Total"],
#                             color=df_1['Area_num'],color_continuous_scale=colour_scale ,range_color=[0,16], 
#                             projection = 'natural earth',text= "Total_str")

# # scatter_fig.update_layout(mapbox_style="open-street-map")

# scatter_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
# scatter_fig.update_traces(textfont_size=8)
# scatter_fig.update_coloraxes(showscale=False)


area_color_map = {
    "Antarctic": "blue",
    "Alaska": "green",
    "Chukotka": "red",
    "E. Greenland": "yellow",
    "Indonesia": "purple",
    "Japan": "pink",
    "NE Atlantic": "orange",
    "Korea": "brown",
    "W. Iceland": "gray",
    "W.Indies": "magenta",
    "NW Canada": "cyan",
    "NE Canada": "teal",
    "Azores": "olive",
    "NW Pacific": "lavender",
    "Iceland": "lightblue",
    "USA W coast": "darkgreen",
    "W. Greenland": "turquoise"
}

CODE_Nation = df['CODE'].unique()
print(CODE_Nation)

colors = {
    'JPN': 'blue',
    'RUS': 'red',
    'DNK': 'green',
    'ISL': 'orange',
    'IDN': 'purple',
    'KOR': 'brown',
    'NOR': 'pink',
    'VCT': 'gray',
    'USA': 'black',
    'PRT': 'cyan',
    'CAN': 'magenta'
}

# japan_trend = df[df['Nation'] == 'Japan'].reset_index()

# print(japan_trend)

# japan_trend = df[df['Nation'] == 'Japan'].reset_index()
# test= px.scatter(japan_trend, x='Year', y='Total')
# test.show()

images_array = ['Project\\640x427-gray-whale.png', 'Project\\640x427-sei-whale.png','Project\\640x427-Whale-Bowhead-markedDW.png']

image_1st = 'Project\\free-medal-icon-1369-thumb.png'

image_base64= []

for i in range(len(images_array)):
    image_base64.append(base64.b64encode(open(images_array[i], 'rb').read()))


data = {'team': {1: 'Sales team 1', 2: 'Sales team 2', 3: 'Sales team 3'},
        'award': {1: '', 2: '', 3: ''},
        'performance': {1: '67.00%', 2: '45.00%', 3: '35.00%'}}

test = pd.DataFrame(data)

table = go.Figure(data=[go.Table(

    columnwidth=[20, 40, 40],

    header=dict(
        values=list(test.columns),
        height=35),

    cells=dict(
        values=[test.team,
                test.award,
                test.performance],
        align=['center', 'center', 'center'],
        font=dict(color='black', size=18),
        height=45)
)])

heightRow = table.data[0].cells.height
numberRow = table.data[0].cells.values[0].__len__()

step_y = 1 / numberRow 
coordinate_y = 0

for index, eachRow in enumerate(test.iterrows()):
    table.add_layout_image(
        source='data:image/png;base64,{}'.format(image_base64[index].decode()),
        x=0.3,
        y=0.55 - coordinate_y,
        xref="x domain",
        yref="y domain",
        xanchor="center",
        yanchor="bottom",
        sizex=.55,
        sizey=.55,
    )

    coordinate_y = coordinate_y + step_y

table.update_layout(
    height=500,
    width=900,
    template="plotly_white"
    
)

table.update_traces(domain_x=[0.5,1], domain_y=[0,1])

table.show()

app = Dash(__name__)

@app.callback(
    Output('world-graph', 'figure'),
    Output('bar-graph', 'figure'),
    Output('heat-map', 'figure'),
    Input('slider', 'value')
)

def update_graphs_1(selected_year):

    filtered_df = df[df['Year'] == selected_year]
    filtered_df_1 = df_1[df_1['Year'] == selected_year]
    filtered_df_2 = df_melted[df_melted['Year'] == selected_year]

    df_heat_year = pd.DataFrame(df_heatmap[selected_year])
    df_heat_year = df_heat_year.pivot(index='Nation', columns='Type', values='Total').fillna(0)
    df_heat_year.columns.name = None
    df_heat_year.index.name = None


    fig1 = px.choropleth(filtered_df, geojson="https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json",
                            color_continuous_scale=px.colors.sequential.YlOrRd,                      
                            color = filtered_df.Total,
                            locations=filtered_df.CODE, featureidkey="id", hover_name = "Nation",
                             center={"lat": 0, "lon": 0},projection = 'natural earth')

    scatter_fig = px.scatter_geo(filtered_df_1, lat=filtered_df_1.latitudes, lon=filtered_df_1.longitudes,size = "size",hover_name = "Area", hover_data=["Total"], 
                            projection = 'natural earth',color=filtered_df_1.Area,color_discrete_sequence= px.colors.qualitative.Light24, animation_group=filtered_df_1.Area,text="Total_str")
    
    scatter_fig.update_traces(marker=dict(size=25,opacity = 0.9))
    scatter_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    scatter_fig.update_traces(textfont=dict(family='sans-serif', size=10, color='black'))
    scatter_fig.update_coloraxes(showscale=True)

    for i in scatter_fig.data:
        fig1.add_trace(i)

    fig1.update_layout(title=f'Countries Involved and Whaling Locations in Year {selected_year}')
    fig1.data[0].legendgroup= scatter_fig.data[0].legendgroup
  
    

    # fig1['data'][1]['marker'] = {
    # 'color': scatter_fig.data[0].marker.color,
    # 'size': 20,
    # # 'coloraxis': "coloraxis",
    # 'opacity': 0.5,
    # 'sizemode': 'area',
    # 'sizeref': 0.001,
    # 'autocolorscale': False
    # }

    # fig.update_layout( autosize=False,width=800, height=700)
    fig1.update_layout(coloraxis_colorbar_x=-0.15)
    
    bar_chart = px.bar(filtered_df_2,x="Species",y="Count",barmode='stack')
    bar_chart.update_layout(title=f'Species Lost in Year {selected_year}')
    bar_chart.update_traces(marker_color='#D34336')

    # df = px.data.medals_wide(indexed=True)
    heat_map = px.imshow(df_heat_year, color_continuous_scale='RdBu_r', origin='lower',text_auto = True)

    # fig2 = go.Figure(data=[go.Scatter(x=x, y=y2)])
    # fig2.update_layout(title='Cosine Graph')
    return fig1,bar_chart,heat_map



@app.callback(
    Output('aggregate-graph', 'figure'),
    Input('my-checkbox', 'value')
)

def update_graphs(selected_nation):
    print(selected_nation)
    fig = go.Figure()
    traces = []
    for value in selected_nation:
        trend = df[df['CODE'] == value].reset_index()
        traces.append(px.scatter(trend, x='Year', y='Total',color= trend.CODE,color_discrete_map=colors))

    for trace in traces:
        fig.add_traces(trace.data)

    return fig


app.layout = html.Div([
    html.H1('Whaling Global Data and Statistics'),
    dcc.Slider(
        id='slider',
        min=df['Year'].min(),
        max=df['Year'].max(),
        value=df['Year'].min(),
        step=None,
        marks= {str(year): str(year) if year % 5 == 0 else '' for year in df['Year'].unique()},
    ),
      dcc.Graph(id='world-graph'),
      dcc.Graph(id='bar-graph'),
      dcc.Graph(id='heat-map'),

    dcc.Checklist(
        id='my-checkbox',
        options=[
             {'label': i, 'value': i} for i in  CODE_Nation],
        value=['JPN','RUS','KOR'],
    ),
    dcc.Graph(id='aggregate-graph'),
    dcc.Graph(figure=table)
])


# if __name__ == '__main__':
#     app.run_server(debug=True)