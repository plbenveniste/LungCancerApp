from plotly.tools import mpl_to_plotly
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import base64
from io import BytesIO
import plotly.graph_objs as go


from app import app


style = {'padding': '1.5em'}

layout = html.Div([
  html.P([html.Br()]),
  html.P([html.Br()]),
  dcc.Markdown('#### Eight questions to predict the risk of lung cancer:'),
  dcc.Markdown('Each form must be carefully filled out to obtain the prediction'),
  html.P([html.Br()]),

  dcc.Markdown('###### What is your age?'),
  dcc.Input(
    id='age',
    placeholder='Enter a value',
    type='number',
    value=''),
  
  dcc.Markdown('###### At what age did you start smoking?'),
  dcc.Input(
    id='age_started_smoking',
    placeholder='Enter a value',
    type='number',
    value=''),
  
  dcc.Markdown('###### What is your current smoking status?'),
  dcc.Dropdown(
    id='smoking_status',
    options=[
        {'label': 'Current cigarette smoker', 'value': '1'},
        {'label': 'Former cigarette smoker', 'value': '2'}
    ],
    value=''),

  dcc.Markdown("""###### How many years did you smoke?"""),
  dcc.Input(
    id='years_smoking',
    placeholder='Enter a value',
    type='number',
    value=''),
  
  dcc.Markdown("""###### How many packs of cigarettes do/did you smoke each day?"""),
  dcc.Input(
    id='packs_day',
    placeholder='Enter a value',
    type='number',
    value=''),
  
  dcc.Markdown('###### Do you have a family history of lung cancer?'),
  dcc.Dropdown(
    id='lung_cancer_family_history',
    options=[
        {'label': 'No', 'value': '0'},
        {'label': 'Yes', 'value': '1'}
    ],
    value=''),

  dcc.Markdown('###### What is your height (in inches)?'),
  dcc.Input(
    id='height',
    placeholder='Enter a value',
    type='number',
    value=''),
  
  dcc.Markdown('###### What is your weight (in lbs)?'),
  dcc.Input(
    id='weight',
    placeholder='Enter a value',
    type='number',
    value=''),
  
  html.P([html.Br()]),  
  html.P([html.Br()]),
  html.Div(html.P(['The predicted probability of lung cancer in the next 5 years is:']), style={'fontWeight': 'bold', 'color': '#48a1af', 'font-size': 'large', 'text-align': 'center'}),
  html.Div(id='LungCancerRisk-result1', style={'fontWeight': 'bold', 'color': '#48a1af', 'font-size': 'large', 'text-align': 'center'}),
  html.Div(html.P([html.Br(), html.Br(), html.Br()])),
  html.Div(id = 'LungCancerRisk-result2', style={'width' : '100%', "height": "100%", 'text-align': 'center'}),
  html.Div(html.P(['This chart shows the most important features associated with a higher (red) or lower (blue) risk of lung cancer']), style={'fontWeight': 'bold', 'color': '#48a1af', 'font-size': 'large', 'text-align': 'center'}),
  html.P([html.Br()])
])
@app.callback(
    [Output('LungCancerRisk-result1', 'children'),
    Output('LungCancerRisk-result2','children')],
    [Input('age', 'value'),
    Input('age_started_smoking', 'value'),
    Input('smoking_status', 'value'),
    Input('years_smoking', 'value'),
    Input('packs_day', 'value'),
    Input('lung_cancer_family_history', 'value'),
    Input('height', 'value'),
    Input('weight', 'value')])

def predict(age,age_started_smoking,smoking_status,years_smoking,packs_day,lung_cancer_family_history,height,weight):
  
  #Transform inputs
  age = float(age)
  age_started_smoking = float(age_started_smoking)
  smoking_status = float(smoking_status)
  years_smoking = float(years_smoking)
  packs_day = float(packs_day)
  lung_cancer_family_history = float(lung_cancer_family_history)
  height = float(height)
  weight = float(weight)

  #Compute model inputs 
  age_stopped_smoking = age_started_smoking + years_smoking
  pack_years = packs_day * years_smoking
  bmi = round(weight/(height**2) * 703, 1)

  df = pd.DataFrame(
    columns=['age','age_stopped_smoking','smoking_status','pack_years','age_started_smoking','years_smoking','lung_cancer_family_history','bmi'],
    data=[[age,age_stopped_smoking,smoking_status,pack_years,age_started_smoking,years_smoking,lung_cancer_family_history,bmi]]
  )

  model = pickle.load(open('models/model_lung_cancer.pkl', 'rb'))
  y_pred_proba = model.predict_proba(df)[:,1]
  y_pred = float(y_pred_proba) * 100
  y_pred = np.round(y_pred, 0)
  results = f'{y_pred}%'

  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(df)
  shap.force_plot(explainer.expected_value,  shap_values=shap_values, features=df.values, 
                    feature_names=df.columns, matplotlib=True, show=False, text_rotation=90, figsize=(20, 6), link="identity")
  buf = BytesIO()
  plt.savefig(buf, format="png")
  data = base64.b64encode(buf.getbuffer()).decode("ascii")
  displ = html.Div(html.P([html.Br(), html.Br(), html.Br()]))
  graph = html.Img(src='data:image/png;base64,{}'.format(data), style={'height':'300px','width' : '1000px', 'text-align': 'center'})

  return results, graph
