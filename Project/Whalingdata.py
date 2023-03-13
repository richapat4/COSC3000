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
from dash import Dash, html, dcc

original_df = pd.read_csv("Project\\WhalingData.csv").dropna()
locations = pd.read_csv("Project\\Whaling Locations1.csv")




# Group the dataframe by Year and Nation and sum the Total column
df = original_df.groupby(['Year','Nation'])['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead', 'Total'].sum().reset_index()

df_1 = original_df.groupby(['Year','Area'])['Fin', 'Sperm', 'Humpback', 'Sei', 'Bryde\'s', 'Minke', 'Gray', 'Bowhead', 'Total'].sum().reset_index()

# Save the new dataset to a CSV file
years = df['Year']

#Total number of whales killed per year 
x = df.groupby(df['Year'],group_keys=True)['Total'].sum()

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
print(world.columns.to_list)

world.columns=['pop_est', 'continent', 'name', 'CODE', 'gdp_md_est', 'geometry']

world.to_csv('Project\\world.csv', index=False)


x = gpd.read_file("Project\\former-soviet-union-ussr_1371.geojson")



# def groupby_multipoly(df, by, aggfunc="first"):
#     data = df.drop(labels=df.geometry.name, axis=1)
#     aggregated_data = data.groupby(by=by).agg(aggfunc)

#     # Process spatial component
#     def merge_geometries(block):
#         return MultiPolygon(block.values)

#     g = df.groupby(by=by, group_keys=False)[df.geometry.name].agg(
#         merge_geometries
#     )

#     # Aggregate
#     aggregated_geometry = gpd.GeoDataFrame(g, geometry=df.geometry.name, crs=df.crs)
#     # Recombine
#     aggregated = aggregated_geometry.join(aggregated_data)
#     return aggregated

# grouped = groupby_multipoly(x, by=None)




# Show the plot
plt.show()

# https://cartographyvectors.com/map/1371-former-soviet-union-ussr

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
print(df_1['Area'].unique())
print(df_1['CODE'].unique())



df.to_csv('Project\\new_dataset.csv', index=False)


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
result = grouped.to_dict(orient='index')

df['Year'] = df['Year'].apply(pd.to_numeric)
df = df.sort_values('Year')

df_1['Year'] = df_1['Year'].apply(pd.to_numeric)

# data=pd.merge(world,df,on='CODE')
geometries = []
for a in df['CODE']:
    if(a=="SUN"):
       geometries.append(x)
    elif(a=="VCT"):
        geometries.append(x)
    for c,b in zip(world['CODE'],world['geometry']):
        if(a == c):
            geometries.append(b)


df['geometry'] = geometries


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
df_1['size'] = np.ones(len(df_1['longitudes'])) * 2

# df_1.to_csv('Project\\new_dataset_2.csv', index=False)


df_1.to_csv('Project\\new_dataset_2.csv', index=False)

with open('Project\\former-soviet-union-ussr_1371.geojson') as f:
    data = json.load(f)

##PLOTTING USSR 
    


# locations = alpha 3 code for each country

# data = {
#   "x": [58.1769992],
#   "y": [-167.136665]
# }

# locations = pd.DataFrame(data)

# # fig = px.choropleth(df, locations=df.CODE, color=df.Total,
# #                     color_continuous_scale='RdYlGn_r',
# #                       template="plotly_dark",
# #                       animation_frame=df.Year,
# #                       animation_group=df.Total,
# #                    hover_name=df.Nation,category_orders=df.Year)

# create a dictionary to map areas to unique numerical values
area_to_num = {area: i for i, area in enumerate(df_1['Area'].unique())}

print(df_1['Area'].unique())

# create a new column in df_1 with the numerical values
df_1['Area_num'] = df_1['Area'].map(area_to_num)

area_num_to_name = {0: 'Antarctic', 1: 'Alaska', 2: 'Chukotka', 3: 'E. Greenland',
                    4: 'Indonesia', 5: 'Japan', 6: 'Korea', 7: 'NE Atlantic',
                    8: 'W. Greenland', 9: 'W. Iceland', 10: 'W.Indies', 11: 'Azores',
                    12: 'NW Canada', 13: 'NE Canada', 14: 'NW Pacific', 15: 'USA W coast',
                    16: 'Iceland'}

# area_color_map = {
#     "0": "blue",
#     "1": "green",
#     "2": "red",
#     "3": "yellow",
#     "4": "purple",
#     "5": "pink",
#     "6": "orange",
#     "7": "brown",
#     "8": "gray",
#     "9": "magenta",
#     "10": "cyan",
#     "11": "teal",
#     "12": "olive",
#     "13": "lavender",
#     "14": "lightblue",
#     "15": "darkgreen",
#     "16": "turquoise"
# }

colour_scale = ["blue","green","red","yellow","purple", "turquoise","orange","brown","darkblue","magenta","cyan","teal","olive","lavender","lightblue","darkgreen","pink"]
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


unique_areas = df_1['Area'].unique()
print(unique_areas)

missing_areas = set(unique_areas) - set(area_color_map.keys())
print(missing_areas)

totals = [str(num) for num in df['Total']]
totals_1 = [str(num) for num in df_1['Total']]

df['Total_str'] = totals
df_1['Total_str'] = totals_1

fig = px.choropleth(df, geojson="https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json",
                            color_continuous_scale='RdYlGn_r',                      
                            color = df.Total,animation_frame=df.Year,
                            animation_group=df.Total,
                            locations=df.CODE, featureidkey="id",
                             center={"lat": 0, "lon": 0},projection = 'natural earth')

# fig.add_scattergeo(geojson="https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json"
#                    ,locations = df['CODE'], text = df['Total_str'], featureidkey="id", mode = 'text') 

# # Update text labels for each animation frame
# for i, frame in enumerate(fig.frames):
#     # Get current data for frame
#     data = frame['data']
    
#     # Update text labels for scattergeo trace
#     for trace in data:
#         if trace.name == 'scattergeo':
#             trace.text = df[df.Year == i]['Total']
    
#     # Update frame with modified data
#     fig.frames[i]['data'] = data


scatter_fig = px.scatter_geo(df_1, lat=df_1.latitudes, lon=df_1.longitudes,size = "size",hover_name = "Area", hover_data=["Area_num", "Total"],
                            animation_group=df_1.Area,color=df_1['Area_num'],color_continuous_scale=colour_scale ,animation_frame=df_1.Year,range_color=[0,16], projection = 'natural earth',text= "Total_str")

# scatter_fig.update_layout(mapbox_style="open-street-map")

scatter_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
scatter_fig.update_traces(textfont_size=8)

scatter_fig.update_coloraxes(showscale=False)

# # # scatter_fig.update_traces(hovertemplate=None, hoverinfo='skip')

# for frame in scatter_fig.frames:
#     print(frame)
#     # frame.data[0].textposition = 'inside'
#     # frame.data[0].textfont_size = 14
#     # frame.data[0].showlegend = True

fig.add_trace(scatter_fig.data[0])

for fe, fne in zip(fig.frames, scatter_fig.frames):
        for t in fne.data:
            t.update(marker_coloraxis = "coloraxis2")
        fe.update(data=fe.data + fne.data)

fig.layout.coloraxis2 = scatter_fig.layout.coloraxis

print(fig['data'])

fig['data'][1]['marker'] = {
   'coloraxis': 'coloraxis2',
   'opacity': 0.5,
   'sizemode': 'area',
   'sizeref': 1,
   'autocolorscale': False
}

# fig.update_layout( autosize=False,width=800, height=700)
fig.update_layout(coloraxis2_colorbar_x=-0.15)



##Create Dashboard
app = Dash(__name__)
app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

     html.Div(
        dcc.Graph(
            id='example-graph',
            figure=fig
    ))
    
    html.Div(
        dcc.Graph(
            id='example-graph',
            figure=fig
    ))

])

if __name__ == '__main__':
    app.run_server(debug=True)





















# # fig.update_layout( autosize=False,width=800, height=700)
# # pointless - using mapbox !
# # fig.update_geos(fitbounds = "locations", visible = False)
# fig.layout.coloraxis2.colorbar.x = 0
# fig.show()



# fig = go.Figure()
# colors = itertools.cycle(px.colors.sequential.Viridis)



# Create the traces for each area

# Load data

# Define the layout of the map
# map_layout = go.Layout(
#     title='Whale sightings',
#     geo=dict(
#         scope='world',
#         showland=True,
#         landcolor='rgb(243, 243, 243)',
#         showcountries=True,
#         countrycolor='rgb(204, 204, 204)',
#         projection_type='equirectangular'
#     ),
#     updatemenus=[dict(
#         type='buttons',
#         showactive=False,
#         buttons=[dict(
#             label='Play',
#             method='animate',
#             args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)]
#         ), dict(
#             label='Pause',
#             method='animate',
#             args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))]
#         )]
#     )]
# )

# Define the data traces for each Area

# scatter_fig.update_layout(mapbox_bounds={"west": -180, "east": -50, "south": 20, "north": 90})
# scatter_fig.update_layout(
#     showlegend=True,
#     title=dict(x=0.5)
# ) 


# scatter_fig.update_geos

# areas = df_1['Area'].unique()
# colors = px.colors.qualitative.Set1

# for i,frame in zip(areas,scatter_fig.frames):
# # for i, year in enumerate(areas):
#     frame.data[0].marker.color = colors[i % len(colors)]

# scatter_fig.update_traces(marker=dict(symbol="diamond"))




# print(scatter_fig.frames)


# for frame in scatter_fig.frames:
#     frame.data[0].marker.symbol = "diamond"


#         #  fe.update(marker=dict(color=[area_color_map.get(legendgroup, "gray") for legendgroup in t]))
#     #     t.update(marker=dict(color=[area_color_map.get(country, "gray") for country in t.]))

   
# for fe, fne in zip(fig.frames, scatter_fig.frames):
#     print(fe.data)
    # for i, trace in enumerate(fne.data):
    #     fe.data[i].marker.color = trace.marker.color





# fig.update_layout(mapbox_style="cart-positron",
#                       mapbox_zoom=2,
#                       mapbox_center={"lat": 0, "lon": 0},
#                       margin={"r": 2, "t": 0, "l": 2, "b": 0})







# fig.show()

# # Add Choropleth trace with country outlines
# fig.add_trace(go.Choropleth(
#     geojson="https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json",
#     featureidkey="id",
#     locations=df.CODE,
#     z=[1] * len(df),
#     colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
#     marker_line_width=0.1,
#     showscale=False
# ))

# df = px.data.election()
# geojson = px.data.election_geojson()

# print(geojson)

# fig = px.choropleth_mapbox(df, geojson=geojson, color="Bergeron",
#                            locations="district", featureidkey="properties.district",
#                            center={"lat": 45.5517, "lon": -73.7073},
#                            mapbox_style="carto-positron", zoom=9)


# fig = px.choropleth_mapbox(x,
#                            geojson=x.geometry,
#                            locations=x.index,featureidkey="id",
#                            mapbox_style="carto-positron",
#                            zoom=8.5)

# # fig.update_layout(title_x=0.5, margin={"r":0, "t":30,"l":0,"b":0})





