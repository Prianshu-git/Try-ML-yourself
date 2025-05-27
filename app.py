import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs
from algorithms import AlgorithmDemonstrator
from data_loader import DataLoader
from visualizations import Visualizer

# Configure page
st.set_page_config(
    page_title="Machine Learning Algorithm Demonstrator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def initialize_components():
    return AlgorithmDemonstrator(), DataLoader(), Visualizer()

algo_demo, data_loader, visualizer = initialize_components()

# Main title
st.title("🤖 Machine Learning Algorithm Demonstrator")
st.markdown("Interactive visualizations and educational content for popular ML algorithms")

# Sidebar for algorithm selection
st.sidebar.title("Algorithm Selection")
algorithm_category = st.sidebar.selectbox(
    "Choose Algorithm Category:",
    ["Clustering", "Classification", "Regression"]
)

if algorithm_category == "Clustering":
    algorithms = ["K-Means", "DBSCAN", "Hierarchical Clustering"]
elif algorithm_category == "Classification":
    algorithms = ["Logistic Regression", "Decision Trees", "SVM", "KNN", "Random Forest"]
else:  # Regression
    algorithms = ["Linear Regression", "Polynomial Regression", "Support Vector Regression"]

selected_algorithm = st.sidebar.selectbox("Select Algorithm:", algorithms)

# Dataset selection
st.sidebar.title("Dataset Selection")
if algorithm_category == "Clustering":
    dataset_options = ["Synthetic Blobs", "Iris (2D)", "Wine (2D)"]
elif algorithm_category == "Classification":
    dataset_options = ["Iris", "Wine", "Breast Cancer", "Synthetic Classification"]
else:  # Regression
    dataset_options = ["Synthetic Linear", "Synthetic Polynomial", "Boston Housing (subset)"]

selected_dataset = st.sidebar.selectbox("Choose Dataset:", dataset_options)

# Load data
data, target, feature_names = data_loader.load_dataset(selected_dataset, algorithm_category)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"{selected_algorithm} Visualization")
    
    # Algorithm parameters section
    st.subheader("Algorithm Parameters")
    params = {}
    
    if selected_algorithm == "K-Means":
        params['n_clusters'] = st.slider("Number of Clusters", 2, 10, 3)
        params['random_state'] = st.slider("Random State", 0, 100, 42)
        
    elif selected_algorithm == "DBSCAN":
        params['eps'] = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1)
        params['min_samples'] = st.slider("Min Samples", 2, 20, 5)
        
    elif selected_algorithm == "Hierarchical Clustering":
        params['n_clusters'] = st.slider("Number of Clusters", 2, 10, 3)
        params['linkage'] = st.selectbox("Linkage", ['ward', 'complete', 'average', 'single'])
        
    elif selected_algorithm == "Logistic Regression":
        params['C'] = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
        params['random_state'] = st.slider("Random State", 0, 100, 42)
        
    elif selected_algorithm == "Decision Trees":
        params['max_depth'] = st.slider("Max Depth", 1, 20, 5)
        params['min_samples_split'] = st.slider("Min Samples Split", 2, 20, 2)
        params['random_state'] = st.slider("Random State", 0, 100, 42)
        
    elif selected_algorithm == "SVM":
        params['C'] = st.slider("Regularization (C)", 0.1, 10.0, 1.0, 0.1)
        params['kernel'] = st.selectbox("Kernel", ['rbf', 'linear', 'poly'])
        params['random_state'] = st.slider("Random State", 0, 100, 42)
        
    elif selected_algorithm == "KNN":
        params['n_neighbors'] = st.slider("Number of Neighbors", 1, 20, 5)
        params['weights'] = st.selectbox("Weights", ['uniform', 'distance'])
        
    elif selected_algorithm == "Random Forest":
        params['n_estimators'] = st.slider("Number of Trees", 10, 200, 100, 10)
        params['max_depth'] = st.slider("Max Depth", 1, 20, 5)
        params['random_state'] = st.slider("Random State", 0, 100, 42)
        
    elif selected_algorithm == "Linear Regression":
        params['fit_intercept'] = st.checkbox("Fit Intercept", True)
        
    elif selected_algorithm == "Polynomial Regression":
        params['degree'] = st.slider("Polynomial Degree", 1, 5, 2)
        params['fit_intercept'] = st.checkbox("Fit Intercept", True)
        
    elif selected_algorithm == "Support Vector Regression":
        params['C'] = st.slider("Regularization (C)", 0.1, 10.0, 1.0, 0.1)
        params['kernel'] = st.selectbox("Kernel", ['rbf', 'linear', 'poly'])
    
    # Train model and get results
    if st.button("Train & Visualize", type="primary"):
        with st.spinner("Training model..."):
            model, results = algo_demo.train_algorithm(
                selected_algorithm, data, target, params
            )
            
            # Create visualization
            fig = visualizer.create_visualization(
                selected_algorithm, algorithm_category, data, target, 
                model, results, feature_names
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics
            if results:
                st.subheader("Performance Metrics")
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    for key, value in results.items():
                        if isinstance(value, (int, float)):
                            st.metric(key.replace('_', ' ').title(), f"{value:.4f}")
                
                with metrics_col2:
                    if 'predictions' in results:
                        st.write("**Prediction Summary:**")
                        if algorithm_category == "Clustering":
                            unique_clusters = np.unique(results['predictions'])
                            st.write(f"Number of clusters found: {len(unique_clusters)}")
                            for cluster in unique_clusters:
                                count = np.sum(results['predictions'] == cluster)
                                st.write(f"Cluster {cluster}: {count} points")

with col2:
    st.subheader("Algorithm Information")
    
    # Display algorithm description and pseudo-code
    description, pseudocode = algo_demo.get_algorithm_info(selected_algorithm)
    
    st.markdown("**Description:**")
    st.markdown(description)
    
    st.markdown("**Pseudo-code:**")
    st.code(pseudocode, language="python")
    
    # Dataset information
    st.subheader("Dataset Information")
    st.write(f"**Shape:** {data.shape}")
    st.write(f"**Features:** {', '.join(feature_names[:3])}" + ("..." if len(feature_names) > 3 else ""))
    
    if algorithm_category != "Clustering":
        st.write(f"**Target classes:** {len(np.unique(target))}")
    
    # Show data sample
    df_sample = pd.DataFrame(data[:10], columns=feature_names)
    if algorithm_category != "Clustering":
        df_sample['Target'] = target[:10]
    st.dataframe(df_sample)

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit, Scikit-learn, and Plotly. "
    "Adjust parameters in the sidebar and click 'Train & Visualize' to see real-time results!"
)
