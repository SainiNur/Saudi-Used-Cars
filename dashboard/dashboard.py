import streamlit as st
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import ( mean_absolute_error, 
                             r2_score, 
                             mean_absolute_percentage_error, 
                             root_mean_squared_error)


st.set_page_config(page_title="Saudi Used Cars Dashboard", layout="wide", initial_sidebar_state="expanded") # Page configuration

cars_df = pd.read_csv('dataset/cleaned_data.csv') # Load dataset
X = cars_df.drop(columns='Price') # Features
y = cars_df['Price'] # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Split dataset into training and testing sets

# Load model
try:
    with open('Saudi-Used-Cars-XGB-ML-Regression-Model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file not found.")
except Exception as e:
    print(f"Error while loading model: {e}")

# Function to predict price
def predict_price(select_year, select_option, select_make, 
                  select_type, select_origin, select_region, select_gear_type, select_engine_size, select_mileage):
  input_data = pd.DataFrame([[select_year, select_option, select_make, 
                                select_type, select_origin, select_region, 
                                select_gear_type, select_engine_size, select_mileage]],
                               columns=['Year', 'Options', 'Make', 'Type', 
                                        'Origin', 'Region', 'Gear_Type', 'Engine_Size', 'Mileage'])
  try:
      # Perform prediction
      prediction = model.predict(input_data)
      return prediction[0]
  except Exception as e:
      st.error(f"Error during prediction: {e}")
      return None
  

# Add sidebar
with st.sidebar:
  st.title('User Input Features')

  select_year = st.selectbox('Select Year', cars_df['Year'].sort_values(ascending=False).unique())
  select_option = st.selectbox('Select Options', cars_df['Options'].unique())
  select_make = st.selectbox('Select Make/Car Brand', cars_df['Make'].unique())
  select_type = st.selectbox('Select Car Type', cars_df['Type'].unique())
  select_origin = st.selectbox('Select Origin', cars_df['Origin'].unique())
  select_region = st.selectbox('Select Region', cars_df['Region'].unique())
  select_gear_type = st.selectbox('Select Gear Type', cars_df['Gear_Type'].unique())
  select_engine_size = st.slider('Select Engine Size', cars_df['Engine_Size'].min(), cars_df['Engine_Size'].max())
  select_mileage = st.slider('Select Mileage', cars_df['Mileage'].min(), cars_df['Mileage'].max())

  # button
  if st.button('Predict Price', use_container_width=True):
    # Call predict_price with the individual inputs directly
    price = predict_price(select_year, select_option, select_make, 
                          select_type, select_origin, select_region, 
                          select_gear_type, select_engine_size, select_mileage)
    if price is not None:
        st.success(f'The predicted price is: {price:.0f} SAR')

# Add tab
tab1, tab2 = st.tabs(['Model Evaluation', 'Model Dataset'])

with tab1:
  st.subheader('Model Evaluation')

  # Fit the model and predict test set
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  # Calculate metrics
  metrics = {
    'RMSE': root_mean_squared_error(y_test, y_pred),
    'MAE': mean_absolute_error(y_test, y_pred),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred),
    'R-squared': r2_score(y_test, y_pred)
  }

  # Display metrics
  st.dataframe(pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value']))

  # Create columns for feature importances and plot
  st.subheader('Feture Importances')
  
  col1, col2 = st.columns([4, 6])
  with col1:
    # Get feature importances
    feature_imp = pd.DataFrame({
        'Feature': model['preprocessing'].get_feature_names_out(),
        'Importance': model['model'].feature_importances_
    }).sort_values(by='Importance', ascending=False)

    st.dataframe(feature_imp)

  with col2:
    # Plot feature importances
    fig = px.histogram(feature_imp, x='Feature', y='Importance', title='Feature Importances')
    fig.update_traces(marker_color='#b7e4c7')
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

  # Calculate residuals and plot actual vs predicted vs residuals
  st.subheader('Actual vs Predicted & Residuals')

  residuals = y_test - y_pred
  fig = make_subplots(rows=1, cols=2, subplot_titles=['Actual vs Predicted', 'Residuals'])
  fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', marker=dict(color='#1b4332')), row=1, col=1)
  fig.add_trace(go.Scatter(x=y_test, y=residuals, mode='markers', marker=dict(color='#1b4332')), row=1, col=2)
  st.plotly_chart(fig)

with tab2:
  st.subheader('Categorical Features Distributions')

  # Create columns container
  col1, col2, col3 = st.columns(3)
  
  # Option distribution plot
  with col1:
      fig = px.histogram(cars_df, x='Options', title='Options Distribution')
      fig.update_traces(marker_color='#b7e4c7').update_layout(xaxis_title=None, yaxis_title=None)
      st.plotly_chart(fig, use_container_width=True)

  # Gear type distribution plot
  with col2:
      fig = px.histogram(cars_df, x='Gear_Type', title='Gear Type Distribution')
      fig.update_traces(marker_color='#b7e4c7').update_layout(xaxis_title=None, yaxis_title=None)
      st.plotly_chart(fig, use_container_width=True)

  # Origin distribution plot
  with col3:
      fig = px.histogram(cars_df, x='Origin', title='Origin Distribution')
      fig.update_traces(marker_color='#b7e4c7').update_layout(xaxis_title=None, yaxis_title=None)
      st.plotly_chart(fig, use_container_width=True)

  # Region and Make/Brand distribution plots
  col4, col5 = st.columns(2)
  with col4:
      fig = px.histogram(cars_df, x='Region', title='Region Distribution')
      fig.update_traces(marker_color='#40916c').update_layout(xaxis_title=None, yaxis_title=None)
      st.plotly_chart(fig, use_container_width=True)

  with col5:
      fig = px.histogram(cars_df, x='Make', title='Make/Brand Distribution')
      fig.update_traces(marker_color='#40916c').update_layout(xaxis_title=None, yaxis_title=None)
      st.plotly_chart(fig, use_container_width=True)

  st.subheader('Distribution of Numerical Features Against Labels')

  # Bivariate feature distributions
  col6, col7, col8 = st.columns(3)

  # Year vs Price plot
  with col6:
      fig = px.scatter(cars_df, x='Year', y='Price', title='Year vs Price')
      fig.update_traces(marker_color='#1b4332').update_layout(xaxis_title=None, yaxis_title=None)
      st.plotly_chart(fig, use_container_width=True)

  # Mileage vs Price plot
  with col7:
      fig = px.scatter(cars_df, x='Mileage', y='Price', title='Mileage vs Price')
      fig.update_xaxes(tickvals=list(range(0, 600000, 100000))).update_traces(marker_color='#1b4332').update_layout(xaxis_title=None, yaxis_title=None)
      st.plotly_chart(fig, use_container_width=True)

  # Engine Size vs Price plot
  with col8:
      fig = px.scatter(cars_df, x='Engine_Size', y='Price', title='Engine Size vs Price')
      fig.update_xaxes(tickvals=list(range(0, 7, 1))).update_traces(marker_color='#1b4332').update_layout(xaxis_title=None, yaxis_title=None)
      st.plotly_chart(fig, use_container_width=True)

  # Show detailed dataset
  st.subheader('Detail Dataset')
  st.dataframe(cars_df, use_container_width=True)


