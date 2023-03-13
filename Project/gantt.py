import plotly.express as px
import pandas as pd
import plotly.io as pio

df = pd.DataFrame([
    dict(Task="Find Data Set", Start='2023-02-21', Finish='2023-02-28', Completion_pct=0, Project = 'Visualisation'),
    dict(Task="Explore Data / Plan Data", Start='2023-03-1', Finish='2023-03-6', Completion_pct=0, Project = 'Visualisation'),
    dict(Task="Develop Report Hypothesis/ Aims", Start='2023-03-7', Finish='2023-03-12', Completion_pct=0, Project = 'Visualisation'),
    dict(Task="Data Analysis + Report Writing", Start='2023-03-12', Finish='2023-04-28', Completion_pct=0, Project = 'Visualisation'),
    dict(Task="Find Computer Graphics Topic", Start='2023-03-15', Finish='2023-03-30', Completion_pct=0, Project = 'Computer Graphics'),
    dict(Task="Building / Drafting Project Aims", Start='2023-03-30', Finish='2023-04-2', Completion_pct=0, Project = 'Computer Graphics'),
    dict(Task="Building Computer Graphics Project", Start='2023-04-5', Finish='2023-05-18', Completion_pct=0, Project = 'Computer Graphics'),
    dict(Task="Computer Graphics Report", Start='2023-04-18', Finish='2023-05-26', Completion_pct=0, Project = 'Computer Graphics'),
])

fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Project",hover_data=['Completion_pct'])

fig.update_layout(height=480, 
    paper_bgcolor='rgba(0,0,0,0)', 
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(gridcolor='lightgrey'),
    yaxis=dict(gridcolor='lightgrey'),
    margin=dict(l=400, r=50, b=10, pad=50),
    yaxis_title = None)


fig.update_yaxes(autorange="reversed");
fig.write_html("gantt.html")
