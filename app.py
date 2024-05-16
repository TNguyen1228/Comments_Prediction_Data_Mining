import sys #lead to model_path
sys.path.append('Models')
import joblib # load joblib model
import streamlit as st # Streamlit app
import pandas as pd #read file
import numpy as np # calculator
from sklearn.metrics import mean_squared_error #import mse library
from random_forest import RandomForestRegressor
#  Title
st.markdown("<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'>", unsafe_allow_html=True)
st.markdown("<h1><i class='fa fa-comment'></i> Comments Prediction</h1>", unsafe_allow_html=True)
# Sidebar options
st.sidebar.title('Options')
uploaded_file = st.sidebar.file_uploader("CSV file", type=['csv'])
# model list
models = {
    "Random Forest Regressor":"model1",
    "Decision Tree Regressor":"model2"
    #add more models
}
# sidebar actions
selected_model = st.sidebar.selectbox("Choose model", list(models.keys()))
selected_model_name = models[selected_model]
# Model config
@st.cache(allow_output_mutation=True)  # Cache (faster process)
def load_model(model_name):
    return joblib.load(f'Models/{model_name.lower().replace(" ", "_")}_model.joblib')
# Process
if uploaded_file is not None:
    # read file and preprocess data
    data = pd.read_csv(uploaded_file,header=None)
    data = data.fillna(1) #replace NaN with 1
    data_cut = data.iloc[:, 50:-1] # drop the first 50 cols and the last col (target columm)
    st.write("Uploaded Data:")
    # print
    st.write(data)
    # load model
    model = load_model(selected_model)
    # predict
    X = data_cut.values #input 
    predictions = model.predict(X)
    # calc MSE
    target_column = data.iloc[:, -1] #take out target_column
    y_mean = np.mean(target_column)
    mse = mean_squared_error(target_column, predictions) #mse
    mse_baseline = np.mean((target_column - y_mean) ** 2) #mse_baseline
    # table of result
    result_df = pd.DataFrame({
         **{f'Feature_{i+1}': data_cut[col] for i, col in enumerate(data_cut.columns)},
         'Predicted Label': predictions
     })
    # print
    st.write(f"Result {type(selected_model_name).__name__}:")
    st.write(result_df)
    st.write("MSE:", mse)
    st.write(f"MSE_baseline:", mse_baseline)
    #compare
    if mse < mse_baseline:
        st.success(":heavy_check_mark: mse < mse_baseline")
    if mse > mse_baseline:
        st.error(":warning: mse > mse_baseline")