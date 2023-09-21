from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

df = pd.read_csv('dataset\Animal_Shelter_Intake_and_Outcome.csv', sep=';')

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='Przewidywanie ilości dni spędzonych w schronisku', style={'textAlign':'center'}),
    dcc.Dropdown(df['Type'].unique(), 'DOG', id='dropdown-selection'),
    dcc.Graph(id='graph-content')
])

@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value')
)
def update_graph(value):
    dff = df[df["Type"]==value]
    return px.scatter(dff, x='Color', y='Days in Shelter')

if __name__ == '__main__':
    app.run(debug=False)