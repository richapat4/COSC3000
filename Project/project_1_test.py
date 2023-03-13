import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv("archive/climate-risk-index-1.csv")
df_regions = pd.read_csv("archive/countries_continent_code.csv")
print(df)
print(df.columns.to_list)

continents = df_regions['Continent_Name']
codes = df_regions['Three_Letter_Country_Code']
rw_codes = df['rw_country_code']

print(continents)
print(codes)

# ['Asia' 'Europe' 'Africa' 'Oceania' 'North America' 'South America'] 

colors_t = {'Asia': 'b','Europe': 'purple', 'Africa': 'r','Oceania': 'green', 'North America' : 'magenta','South America':'pink'}



regions = []

print(rw_codes.size)
c = 0
for i in range(0,rw_codes.size):
    x = rw_codes[i]
    for j in range(0,codes.size):
        b = codes[j]
        a = continents[j]
        if((str(b) == 'nan') or (str(x)== 'nan') or (str(a)== 'nan')):
            continue
        elif(str(x) == str(b)):
            c+=1
            print('({0},{1},{2},{3})'.format(x,b,a,c))
            regions.append(a)


df['Regions'] = regions

print(df['Regions'].unique())

labels = df['rw_country_name']
x = df['cri_rank']

y = df['fatalities_rank']

fig, ax = plt.subplots()
# plt.bar(x = df['rw_country_code'], y = df['cri_rank'])


scatter = plt.scatter(x,y, c=df['Regions'].map(colors_t))
# plt.legend(handles=scatter.legend_elements()[0], 
#            title="Regions")

plt.xlabel("Climate Risk Rank")
plt.ylabel("Fatalities Rank")

annotation = ax.annotate(text = '',
    xy=(0,0),
    xytext=(15, 15), # distance from x, y
    textcoords='offset points',
    bbox={'boxstyle': 'round', 'fc': 'w'},
    arrowprops={'arrowstyle': '->'}
)
annotation.set_visible(False)


def motion_hover(event):
    annotation_visbility = annotation.get_visible()
    if event.inaxes == ax:
        is_contained, annotation_index = scatter.contains(event)
        if is_contained:
            data_point_location = scatter.get_offsets()[annotation_index['ind'][0]]
            print(annotation_index)
            for c,label, a, b in zip(rw_codes, labels, x, y):
                if((data_point_location[0] == a) and (data_point_location[1] == b)):
                     country_label = label
                     k = c
                     break

            annotation.xy = data_point_location
           
            print(data_point_location)
            text_label = '({0}, {1},{2},{3})'.format(data_point_location[0], data_point_location[1],country_label,k)
            annotation.set_text(text_label)

            # annotation.get_bbox_patch().set_facecolor(cmap(norm(colors[annotation_index['ind'][0]])))
            annotation.set_alpha(0.4)

            annotation.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if annotation_visbility:
                annotation.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', motion_hover)

plt.show()