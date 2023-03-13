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



original_df = pd.read_csv("Project\\WhalingData.csv").dropna()
locations = pd.read_csv("Project\\Whaling Locations1.csv")




# Group the dataframe by Year and Nation and sum the Total column
df = original_df.groupby(['Year','Nation'])['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead', 'Total'].sum().reset_index()

df_1 = original_df.groupby(['Year','Area'])['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead', 'Total'].sum().reset_index()


x = df.groupby(df['Year'],group_keys=True)['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead'].sum().reset_index()

df_melted = x.melt(id_vars=['Year'], var_name='Species', value_name='Count')

print(df_melted)

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
            # elif(country == "Antarctic"):
            #    #print("HERE\n")
            #   #country =pycountry.countries.search_fuzzy('Antarctica')
            #    #code = pycountry.countries.get(name = 'Antarctica').alpha_3
            # elif(country == "Alaska"):
                
            #     #country =pycountry.countries.search_fuzzy('Antarctica')
            #     #code = pycountry.countries.search_fuzzy('Alaska')
            #     print(code)
            else:
                code=pycountry.countries.get(name=country).alpha_3
            
           # .alpha_3 means 3-letter country code 
           # .alpha_2 means 2-letter country code
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

df_1['Area_num'] = df_1['Area'].map(area_to_num)

colour_scale = ["blue","green","red","yellow","purple", "turquoise","orange","brown","darkblue","magenta","cyan","teal","olive","lavender","lightblue","darkgreen","pink"]




# scatter_fig = px.scatter_geo(df_area[year], lat=df_area.latitudes, lon=df_area.longitudes,size = "size",hover_name = "Area", hover_data=["Area_num", "Total"],
#                             color=df_1['Area_num'],color_continuous_scale=colour_scale ,range_color=[0,16], 
#                             projection = 'natural earth',text= "Total_str")

# # scatter_fig.update_layout(mapbox_style="open-street-map")

# scatter_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
# scatter_fig.update_traces(textfont_size=8)
# scatter_fig.update_coloraxes(showscale=False)





app = Dash(__name__)

@app.callback(
    Output('world-graph', 'figure'),
    Output('bar-graph', 'figure'),
    Input('slider', 'value')
)


def update_graphs(selected_year):
    # x = np.linspace(-slider_value, slider_value, 100)
    filtered_df = df[df['Year'] == selected_year]
    filtered_df_1 = df_1[df_1['Year'] == selected_year]
    filtered_df_2 = df_melted[df_melted['Year'] == selected_year]

    fig1 = px.choropleth(filtered_df, geojson="https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json",
                            color_continuous_scale='RdYlGn_r',                      
                            color = filtered_df.Total,
                            locations=filtered_df.CODE, featureidkey="id",
                             center={"lat": 0, "lon": 0},projection = 'natural earth')

    scatter_fig = px.scatter_geo(filtered_df_1, lat=filtered_df_1.latitudes, lon=filtered_df_1.longitudes,size = "size",hover_name = "Area", hover_data=["Total"], 
                            projection = 'natural earth',color='Area_num',color_continuous_scale=px.colors.sequential.Viridis, range_color= [0,16])

    scatter_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    scatter_fig.update_traces(textfont_size=8)
    scatter_fig.update_coloraxes(showscale=True)

    fig1.add_trace(scatter_fig.data[0])

    fig1.update_layout(title='Whaling Global Data')

    fig1.layout.coloraxis2 = scatter_fig.layout.coloraxis
    
    fig1['data'][1]['marker'] = {
    'color': scatter_fig.data[0].marker.color,
    'size': 20,
    'coloraxis': "coloraxis2",
    'opacity': 0.5,
    'sizemode': 'area',
    'sizeref': 0.001,
    'autocolorscale': False
    }

    # fig.update_layout( autosize=False,width=800, height=700)
    fig1.update_layout(coloraxis2_colorbar_x=-0.15)
    
    bar_chart = px.bar(filtered_df_2,x="Species",y="Count",barmode='stack')

    # fig2 = go.Figure(data=[go.Scatter(x=x, y=y2)])
    # fig2.update_layout(title='Cosine Graph')
    return fig1,bar_chart

app.layout = html.Div([
    html.H1('Two Graphs with a Slider'),
    dcc.Slider(
        id='slider',
        min=df['Year'].min(),
        max=df['Year'].max(),
        value=df['Year'].min(),
        step=None,
        marks= {str(year): str(year) if year % 5 == 0 else '' for year in df['Year'].unique()},
    ),
      dcc.Graph(id='world-graph'),
      dcc.Graph(id='bar-graph')
])


if __name__ == '__main__':
    app.run_server(debug=True)