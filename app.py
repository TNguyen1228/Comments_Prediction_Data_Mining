import sys #lead to model_path
sys.path.append('Models')
import joblib # load joblib model
import streamlit as st # Streamlit app
import pandas as pd #read file
import numpy as np # calculator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error #import mse library
from random_forest import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#  Title
st.markdown("<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'>", unsafe_allow_html=True)
st.markdown("<h1><i class='fa fa-comment'></i> Comments Prediction</h1>", unsafe_allow_html=True)
# Sidebar options
st.sidebar.title('Options')
uploaded_file = st.sidebar.file_uploader("CSV file", type=['csv'])
# model list
models = {
    "Random Forest Regressor":"Random Forest Regressor",
    "Decision Tree Regressor":"Decision Tree Regressor"
    #add more models
}
# sidebar actions
selected_model = st.sidebar.selectbox("Choose model", list(models.keys()))
selected_model_name = models[selected_model]
# Model config
@st.cache(allow_output_mutation=True)  # Cache (faster process)
def load_model(model_name):
    return joblib.load(f'Models/{model_name.lower().replace(" ", "_")}_model.joblib')
# PCA
def plot_pca(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', data=pca_df)
    plt.title('PCA of Dataset')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    st.pyplot(plt)
# Draw histtogram of Target column
def plot_comment_counts(data, num_records):
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("husl", len(data.iloc[:, -1]))
    plt.bar(range(len(data)), data.iloc[:, -1], color=colors)
    plt.xlabel('Record Index')
    plt.ylabel('Comment Count')
    plt.title('Number of Comments for Each Record')
    # if num_records<=14:
    #     plt.xticks(np.arange(0, len(data), step=len(data)//10))  # Adjust x ticks
    #     # plt.yticks(np.arange(0, max(data.iloc[:, -1]), step=max(data.iloc[:, -1])//10)) 
    # else:
    #     plt.xticks(np.arange(0, len(data), step=len(data)//15))  # Adjust x ticks
    #     plt.yticks(np.arange(0, max(data.iloc[:, -1]), step=max(data.iloc[:, -1])//15))  # Adjust x ticks
    # # Add specific values on top of each bar
    # for i, value in enumerate(data.iloc[:, -1]):
    #     plt.annotate(str(round(value, 2)), xy=(i, value), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    # st.pyplot(plt)
    for i in range(len(data) - 1):
        plt.plot([i, i + 1], [data.iloc[i, -1], data.iloc[i + 1, -1]], color='gray')
    
    # Add specific values on top of each bar
    for i, value in enumerate(data.iloc[:, -1]):
        plt.annotate(str(round(value, 2)), xy=(i, value), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    
    if len(data) > 0:
        num_xticks = max(1, len(data) // 15)  # Ensure there's at least one tick
        plt.xticks(np.arange(0, len(data), step=num_xticks))  # Adjust x ticks
    
    max_y = max(data.iloc[:, -1])
    if max_y > 0:
        num_yticks = max(1, max_y // 10)  # Ensure there's at least one tick
        plt.yticks(np.arange(0, max_y, step=num_yticks))  # Adjust y ticks
    
    st.pyplot(plt)
# def plot_corr(data):
#     plt.figure(figsize=(12, 10))
#     corr_matrix = data.iloc[:, 50:-1].corr()
#     sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
#     plt.title('Correlation Matrix')
#     st.pyplot(plt)
# Process
if uploaded_file is not None:
    # read file and preprocess data
    data = pd.read_csv(uploaded_file,header=None)
    data = data.fillna(1) #replace NaN with 1
    data_cut = data.iloc[:, 50:-1] # drop the first 50 cols and the last col (target columm)
    st.write("Uploaded Data:")
    # print data and visuallize data 
    st.write(data)
    st.sidebar.subheader("Visualization Options")
    if st.sidebar.checkbox("PCA Plot"):
        plot_pca(data_cut)
    if st.sidebar.checkbox("Plot Comment Counts"):
        num_records = st.sidebar.slider("Select number of records to display", 10, 30, 10)  # Slider for selecting number of records
        plot_comment_counts(data.head(num_records), num_records)
    
    # if st.sidebar.checkbox("Correlation Matrix"):
    #     plot_corr(data)
    # if st.sidebar.checkbox("Box Plot of First 10 Features"):
    #     plot_box(data)
    # load model
    model = load_model(selected_model)
    # predict
    X = data_cut.values #input 
    predictions = model.predict(X)
    predictions_rounded=np.round(predictions).astype(int)
    # calc MSE
    target_column = data.iloc[:, -1] #take out target_column
    y_mean = np.mean(target_column)
    mse = mean_squared_error(target_column, predictions) #mse
    mse_baseline = np.mean((target_column - y_mean) ** 2) #mse_baseline
    # table of result
    result_df = pd.DataFrame({
         **{f'Feature_{i+1}': data_cut[col] for i, col in enumerate(data_cut.columns)},
         'Predicted Label': predictions_rounded
     })
    # print
    st.write("> The first 50 columns : Average,Standard deviation, min, max and median of the Attribute 51 to 60 for the source of the current blog post")
    st.write("> We drop the first 50 columns")
    st.write(f"Result {selected_model}:")
    st.write(result_df)
    st.write("MSE:", mse)
    st.write(f"MSE_baseline:", mse_baseline)
    #compare
    if mse < mse_baseline:
        st.success(":heavy_check_mark: mse < mse_baseline")
    if mse > mse_baseline:
        st.error(":warning: mse > mse_baseline")