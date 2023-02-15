# LungCancer
Web app to predict lung cancer risk in the next 10 years. 

## Data

The Data used comes from (to be completed)

## Repository files
The repository includes the following files : 

assets
- custom css
- the logo of the web page

models
- model_lungs_final.pkl - the final XGBoost model for lung cancer risk in pickle format

tabs
- intro.py - The code for the 'About' tab
- LungCancerRisk.py - The tab to predict lung cancer risk

main app
- app.py - Initiates the Dash app
- index.py - The main Dash code with the layout and callback
- Procfile - The Procfile for Heroku
- requirements.txt - The requirements.txt file for Heroku
- runtime.txt - The python required version for Heroku

## To run locally

In order to run this code on your machine : 
- run the requirements.txt file : 'pip install -r requirements.txt
- launch the app : 'python index.py'
