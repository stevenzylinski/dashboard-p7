# -*- coding: utf-8 -*-
from pydoc import classname
from turtle import color
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import base64
import requests, json
import numpy as np
import dash_daq as daq
import shap
import plotly.figure_factory as ff

external_stylesheets = [dbc.themes.LUX]

app = Dash(__name__, external_stylesheets=external_stylesheets)
#app = dash_app.server

# Initialize the application by loading data set and getting prediction for all client from the API
df = pd.read_csv('./Data/export_datav3.csv')
object_columns = df.select_dtypes('object').columns.to_list()
numeric_columns = df.select_dtypes(exclude = 'object').columns.to_list()
df_order = df[object_columns + numeric_columns]
threshold = 0.24489795918367344

# Creating request function for api returning probability of default
def prediction_api(data_input):
    json_input = data_input.to_json(orient = 'index')
    url = 'https://msdocs-python-webapp-quickstart-szm.azurewebsites.net/api/makecalc/'
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    prediction = requests.post(url, data=json_input, headers=headers)
    return(prediction.text)

prediction = pd.read_json(prediction_api(df.iloc[0:len(df)]))
df['Target'] = np.where(prediction.iloc[:,1]>= threshold, 'Not Safe', 'Safe')

# Creating request function for api returning explainer object from shap
def explainer_shap(data_input):
    json_input = data_input.to_json(orient = 'index')
    url = 'https://msdocs-python-webapp-quickstart-szm.azurewebsites.net/api/shap_imp/'
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    response = requests.post(url, data=json_input, headers=headers)
    return(response.text)

response = pd.read_json(explainer_shap(df.iloc[0:len(df)]))
values = pd.read_json(response["Value"].to_json(),orient = 'index').to_numpy()
base_values = pd.read_json(response["Base_Value"].to_json(),orient = 'index').to_numpy()
exp_api = shap.Explanation(values[:],
                          np.squeeze(base_values),
                          data = df_order,
                          feature_names = df_order.columns)

# First part : Selecting the client and quick overview

# Creating card for main_page

card_ind = dbc.Card([
    dbc.Row([
        dbc.Col(
            dbc.CardImg(id ='image-card',className="img-fluid rounded-start"),
            className="col-md-4"),
            dbc.Col(
                dbc.CardBody([
                    html.H4(className="card-title",id = "card-title-init"),
                    html.P(className="card-text", id="card-text-init")],
                    className="col-md-8"))],
        className="g-0 d-flex align-items-center")],
    className="mb-3",
    style={"maxWidth": "540px"},id = "card-ind")

card_inc = dbc.Card([
    dbc.Row([
            dbc.Col(
                dbc.CardBody([
                    html.H4('Income',className="card-title",id = "card-title-inc"),
                    html.P(className="card-text", id="card-text-inc")],
                    className="col-md-8"))],
        className="g-0 d-flex align-items-center")],
    className="mb-3",
    style={"maxWidth": "540px"},id = "card-inc")

card_sit = dbc.Card([
    dbc.Row([
            dbc.Col(
                dbc.CardBody([
                    html.H4('Situation',className="card-title",id = "card-title-sit"),
                    html.P(className="card-text", id="card-text-sit")],
                    className="col-md-8 text-left")),
            dbc.Col(
                dbc.CardBody([
                    html.P(className="card-text",id="card-text-sit2")
                ])
            )],
        className="g-0 d-flex align-items-center")],
    className="mb-3",
    #style={"maxWidth": "1080px"},
    id = "card-sit")

gauge = daq.Gauge(
    color={"gradient":True,"ranges":{"green":[0,threshold*0.8*100],"yellow":[threshold*0.8*100,threshold*90],"red":[threshold*90,100]}},
    label='Default\'s probability',
    showCurrentValue=True,
    units="%",
    max=100,
    min=0,
    id = 'gauge',
    size = 300)

# Layout for main page
app.layout = html.Div([
    # First part for selecting client and quick overview
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Dashboard : Client Overview", className="mb-2"))
        ]),
        dbc.Row([
            dbc.Col(html.H6("Visualising different key features for a client accross all client member statistics", className="mb-4"))
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(html.H3(children = 'Client Overview',className="text-center text-light bg-dark"), body=True, color="dark"),
                    className="mb-4")
        ]),
        html.Label("Select a Client ID"),
        dcc.Dropdown(df.index,df.index[0],id="select-client"),
        html.Br(),
        dbc.Row([
            dbc.Col([card_ind,card_inc]),
            dbc.Col([gauge])],align = "center"),
        dbc.Row(
            dbc.Col([card_sit]),align = "center")
    ]),
    dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card(html.H3(children = 'Feature Importance',className="text-center text-light bg-dark"), body=True, color="dark"),
            className="mb-4")
        ]),
        dbc.Row([
            dcc.Graph(id="shap-local",style={'width': '300vh', 'height': '50vh'})
        ])
    ]),
    dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card(html.H3(children = "Analysis of client",className="text-center text-light bg-dark"), body=True, color="dark"),
            className="mb-4")
        ]),
        dbc.Row([
            dbc.Col(html.H1(children = "Analysis Univariate")),
            dbc.Col([daq.BooleanSwitch(id='our-boolean-switch', on=False,
                                        label = "Show only Shap features",
                                        labelPosition = "right")])
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Select a feature :"),
                dcc.Dropdown(id="select-feature-x-uni",options = df.columns, value = df.columns[0])
            ])
        ]),
        dbc.Row([html.Br()]),
        dbc.Row(id = 'uni-desc-cat'),
        dbc.Row([
            dcc.Graph(id = "univariate")
        ]),
        dbc.Row([
            dbc.Col(html.H1(children = "Analysis Bivariate")),
            dbc.Col([daq.BooleanSwitch(id='our-boolean-switch2', on=False,
                                        label = "Show only Shap features",
                                        labelPosition = "right")])
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Features to show on X"),
                dcc.Dropdown(id="select-feature-x")
            ]),
            dbc.Col([
                html.Label("Features to show on Y"),
                dcc.Dropdown(id="select-feature-y")
            ])
        ]),
        dbc.Row([dcc.Graph(id="graph-feature")])

    ])

])

# CallBack for part "Client Overview"
# Update information inside card for the client
@app.callback(
    Output('image-card','src'),
    Output('card-title-init','children'),
    Output('card-text-init','children'),
    Output('card-text-inc','children'),
    Output('card-text-sit','children'),
    Output('card-text-sit2','children'),
    Output('card-ind','color'),
    Output('card-inc','color'),
    Output('card-sit','color'),
    Output('gauge','value'),
    Output('shap-local','figure'),
    Input('select-client','value')
)
def create_card(client):
    # Getting prediction credit score for client and coloring card accordingly
    dff = df.iloc[client,:]
    if dff["Target"] == 'Not Safe':
        color = "danger"
    else:
        color = "success" 

    # Filling information on card
    
    if dff['CODE_GENDER']=='M':
        image = "./asset/male_portrait.png"
    else:
        image = "./asset/female_portrait.png"
    encoded_image = base64.b64encode(open(image, 'rb').read())
    image_return='data:image/png;base64,{}'.format(encoded_image.decode())

    title_ind = 'Client {}'.format(client)

    text_ind = [
        "Sexe : {}".format(dff['CODE_GENDER']),
        html.Br(),
        "Has a Car : {}".format(dff["FLAG_OWN_CAR"]),
        html.Br(),
        "Has Realty : {}".format(dff["FLAG_OWN_REALTY"]),
        html.Br(),
        "Has {} Children".format(dff["CNT_CHILDREN"])
    ]

    text_inc = [
        "Income Type : {}".format(dff['NAME_INCOME_TYPE']),
        html.Br(),
        "Occupation : {}".format(dff['OCCUPATION_TYPE']),
        html.Br(),
        "Organization : {}".format(dff['ORGANIZATION_TYPE']),
        html.Br(),
        "Income : {:,.0f}".format(dff['AMT_INCOME_TOTAL']),
    ]

    text_sit = [
        "Contract Type : {}".format(dff['NAME_CONTRACT_TYPE']),
        html.Br(),
        "Family Size : {:.0f}".format(dff['CNT_FAM_MEMBERS']),
        html.Br(),
        "Education : {}".format(dff['NAME_EDUCATION_TYPE']),
        html.Br(),
        "Family Status : {}".format(dff['NAME_FAMILY_STATUS']),
    ]

    text_sit2 = [
        "Housing Type : {}".format(dff['NAME_HOUSING_TYPE']),
        html.Br(),
        "Suite Type : {}".format(dff['NAME_TYPE_SUITE']),
        html.Br(),
        "Credit Amount : {:,.0f}".format(dff['AMT_CREDIT']),
        html.Br(),
        "Annuity Amount : {:,.0f}".format(dff['AMT_ANNUITY']),
        html.Br(),
        "Probability of Default : {:,.0%}".format(prediction.iloc[client,1])
    ]

    # Custom Waterfall charts
    top_9 = pd.DataFrame({'Features':exp_api.feature_names,
             'Contribution':exp_api.values[client],
             'Sort_index':abs(exp_api.values[client]),
                     'Data':exp_api.data.values[client]}).sort_values(
        "Sort_index",ascending = False).head(9).sort_values("Sort_index")
    new_data = []
    for i in range(0,len(top_9)):
        try:
            new_value = round(float(top_9.iloc[i,3]),3)  
        except:
            new_value = top_9.iloc[i,3]
        new_data.append(str(new_value))
    top_9["Data"]=new_data
    other_feat_text = [round(exp_api.values[client].sum()-top_9['Contribution'].sum(),2)]
    other_feat = [exp_api.values[client].sum()-top_9['Contribution'].sum()]
    top_9['New_feat_Name'] = top_9['Data'].map(str) +" = " + top_9['Features'].map(str)
    fig = go.Figure(go.Waterfall(orientation = "h",
                                 y = ["Base Value","427 other features"]+top_9["New_feat_Name"].to_list() + ["Output Model"],
                                 measure = ["relative","relative", "relative", "relative", "relative", "relative",
                                            "relative", "relative", "relative","relative","relative","relative"],
                                 x = [0] + other_feat+top_9["Contribution"].to_list() + [0],
                                 base = exp_api.base_values[client],
                                 text = ["Base Value = "+str(round(exp_api.base_values[client],3))]+other_feat_text+round(top_9["Contribution"],2).to_list(),
                                 decreasing = {"marker":{"color":"Teal"}},
                                 increasing = {"marker":{"color":"Maroon",
                                                         "line":{"color":"red", "width":2}}},
                                 totals = {"marker":{"color":"deep sky blue",
                                                     "line":{"color":"blue", "width":3}}}))

    fig.update_layout(showlegend=False,
                      waterfallgap = 0.2,
                      title="Local Explanation with SHAP")
                      
    fig.add_trace(go.Scatter(
        x=[exp_api.values[client].sum() + exp_api.base_values[client]],
        y=["Output Model"],
        marker=dict(color="blue", size=12),
        mode="markers+text",
        text = [str(round(exp_api.values[client].sum() + exp_api.base_values[client],3))],
        textposition="top center"))
    fig.add_trace(go.Scatter(
        x=[exp_api.base_values[client]],
        y=["Base Value"],
        marker=dict(color="grey", size=12),
        mode="markers+text",
        text = [str(round(exp_api.base_values[client],3))],
        textposition="bottom center"))
            
        

    return(image_return,title_ind,text_ind,text_inc,text_sit,text_sit2,color,color,color,round(prediction.iloc[client,1]*100,2),fig)

# Callback for univariate analysis
## Update the list of features available in the univariate analysis section
@app.callback(
    Output("select-feature-x-uni","options"),
    Output("select-feature-x-uni","value"),
    Input("our-boolean-switch","on"),
    Input("select-client","value")
)
def update_list_univariate(switch,client):
    if switch:
        top_9 = pd.DataFrame({'Features':exp_api.feature_names,
            'Contribution':exp_api.values[client],
            'Sort_index':abs(exp_api.values[client]),
            'Data':exp_api.data.values[client]}).sort_values(
                "Sort_index",ascending = False).head(9).sort_values("Sort_index")
        options = top_9['Features'].to_list()
    else:
        options = df.columns
    return(options, options[0])

## Update the figure according to the feature selected in the univariate section
@app.callback(
    Output("univariate","figure"),
    Output("uni-desc-cat","children"),
    Input("select-client","value"),
    Input("select-feature-x-uni","value")
)
def update_graph_univariate(client,feature):
    if df[feature].dtypes=="object":
        dff = df.groupby([feature,"Target"]).size().reset_index(name = "count")
        fig = px.histogram(dff, x=feature, y="count",
             color='Target', barmode='group',
             height=400,
             text_auto=True,
             color_discrete_sequence=['#BB394E','#39BB66'])

        child = [dbc.Col([
                dbc.Card(children = [feature + ' = {}'.format(df.loc[client,feature])],color = "primary",inverse = True)
            ]),
            dbc.Col([
                dbc.Card(children = ['Target : {}'.format(df.loc[client,'Target'])], color = "primary",inverse = True)
            ])]
    else:
        dff = df[["Target",feature]]
        fig = px.histogram(dff, x=feature, color="Target",
                   marginal="box", # or violin, rug
                   hover_data=dff.columns,
                   color_discrete_sequence=['#BB394E','#39BB66'])
        if df.loc[client,'Target'] == 'Not Safe':
            color_use = '#BB394E'
        else:
            color_use = '#39BB66'
        fig.add_vline(x=dff.loc[client,feature], line_width=3, line_dash="dash", line_color=color_use)
        child = ''
    return(fig,child)


# Update list for x-axis and y-axis features
@app.callback(
    Output("select-feature-x","options"),
    Output("select-feature-x","value"),
    Output("select-feature-y","options"),
    Output("select-feature-y","value"),
    Input("our-boolean-switch2","on"),
    Input("select-client","value")
)
def update_xy_feature(switch,client):
    if switch:
        top_9 = pd.DataFrame({'Features':exp_api.feature_names,
            'Contribution':exp_api.values[client],
            'Sort_index':abs(exp_api.values[client]),
            'Data':exp_api.data.values[client]}).sort_values(
                "Sort_index",ascending = False).head(9).sort_values("Sort_index")
        options = top_9['Features'].to_list()
    else:
        options = df.columns
    return(options, options[0],options, options[0])

# Update graphics according to features selected
@app.callback(
    Output('graph-feature','figure'),
    Input('select-feature-x','value'),
    Input('select-feature-y','value')
    )
def update_figure(feature_x,feature_y):

    if (df[feature_x].dtypes == 'object') and (df[feature_y].dtypes=='object'):
        heatmap = df.groupby([feature_x])[feature_y].value_counts().sort_index().unstack(1)
        fig = px.imshow(heatmap, text_auto=True, aspect="auto")
    elif (df[feature_x].dtypes != 'object') and (df[feature_y].dtypes!='object'):
        fig = px.scatter(x = df[feature_x],y = df[feature_y],color = df['Target'])
    else:
        fig = px.box(x = df[feature_x],y = df[feature_y],color = df['Target'])
    return fig

if __name__ == '__main__':
    app.run(debug=True)
