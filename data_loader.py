import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_classification, make_regression, make_blobs
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Class to handle loading and preprocessing of different datasets"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def load_dataset(self, dataset_name, algorithm_category):
        """Load and return the specified dataset"""
        if dataset_name == "Iris":
            return self._load_iris(algorithm_category)
        elif dataset_name == "Iris (2D)":
            return self._load_iris_2d()
        elif dataset_name == "Wine":
            return self._load_wine(algorithm_category)
        elif dataset_name == "Wine (2D)":
            return self._load_wine_2d()
        elif dataset_name == "Breast Cancer":
            return self._load_breast_cancer()
        elif dataset_name == "Synthetic Classification":
            return self._load_synthetic_classification()
        elif dataset_name == "Synthetic Blobs":
            return self._load_synthetic_blobs()
        elif dataset_name == "Synthetic Linear":
            return self._load_synthetic_linear()
        elif dataset_name == "Synthetic Polynomial":
            return self._load_synthetic_polynomial()
        elif dataset_name == "Boston Housing (subset)":
            return self._load_boston_subset()
        else:
            return self._load_synthetic_blobs()  # Default fallback
    
    def _load_iris(self, algorithm_category):
        """Load Iris dataset"""
        iris = load_iris()
        data = iris.data
        target = iris.target
        feature_names = iris.feature_names
        
        if algorithm_category == "Clustering":
            # For clustering, don't use target labels during training
            return data, None, feature_names
        else:
            return data, target, feature_names
    
    def _load_iris_2d(self):
        """Load Iris dataset with only 2 features for better visualization"""
        iris = load_iris()
        # Use sepal length and sepal width for 2D visualization
        data = iris.data[:, [0, 1]]
        feature_names = ['Sepal Length', 'Sepal Width']
        return data, None, feature_names
    
    def _load_wine(self, algorithm_category):
        """Load Wine dataset"""
        wine = load_wine()
        data = wine.data
        target = wine.target
        feature_names = wine.feature_names
        
        if algorithm_category == "Clustering":
            return data, None, feature_names
        else:
            return data, target, feature_names
    
    def _load_wine_2d(self):
        """Load Wine dataset with only 2 features for better visualization"""
        wine = load_wine()
        # Use first two features for 2D visualization
        data = wine.data[:, [0, 1]]
        feature_names = ['Alcohol', 'Malic Acid']
        return data, None, feature_names
    
    def _load_breast_cancer(self):
        """Load Breast Cancer dataset"""
        cancer = load_breast_cancer()
        # Use first 2 features for visualization
        data = cancer.data[:, [0, 1]]
        target = cancer.target
        feature_names = ['Mean Radius', 'Mean Texture']
        return data, target, feature_names
    
    def _load_synthetic_classification(self):
        """Generate synthetic classification data"""
        data, target = make_classification(
            n_samples=300,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            random_state=42
        )
        feature_names = ['Feature 1', 'Feature 2']
        return data, target, feature_names
    
    def _load_synthetic_blobs(self):
        """Generate synthetic blob data for clustering"""
        data, target = make_blobs(
            n_samples=300,
            centers=4,
            cluster_std=1.0,
            center_box=(-10.0, 10.0),
            random_state=42
        )
        feature_names = ['X', 'Y']
        return data, None, feature_names
    
    def _load_synthetic_linear(self):
        """Generate synthetic linear regression data"""
        np.random.seed(42)
        X = np.random.randn(100, 1) * 10
        y = 2 * X.ravel() + 1 + np.random.randn(100) * 2
        feature_names = ['X']
        return X, y, feature_names
    
    def _load_synthetic_polynomial(self):
        """Generate synthetic polynomial regression data"""
        np.random.seed(42)
        X = np.linspace(-3, 3, 100).reshape(-1, 1)
        y = 0.5 * X.ravel()**3 - 2 * X.ravel()**2 + X.ravel() + np.random.randn(100) * 2
        feature_names = ['X']
        return X, y, feature_names
    
    def _load_boston_subset(self):
        """Generate synthetic housing data (Boston dataset alternative)"""
        np.random.seed(42)
        # Create synthetic housing data
        n_samples = 200
        
        # Features: rooms, age of house
        rooms = np.random.normal(6, 1, n_samples)
        age = np.random.uniform(0, 100, n_samples)
        
        # Target: house price (based on rooms and age with some noise)
        price = 20 + 5 * rooms - 0.1 * age + np.random.normal(0, 3, n_samples)
        
        data = np.column_stack([rooms, age])
        feature_names = ['Average Rooms', 'House Age']
        
        return data, price, feature_names
    
    def preprocess_data(self, data, scale=True):
        """Preprocess the data (scaling, etc.)"""
        if scale and data.shape[1] > 1:
            # Only scale if we have multiple features and scaling is requested
            return self.scaler.fit_transform(data)
        return data
    
    def get_dataset_info(self, dataset_name):
        """Get information about a dataset"""
        info = {
            "Iris": "Classic flower dataset with 4 features and 3 classes",
            "Iris (2D)": "Iris dataset with 2 features for visualization",
            "Wine": "Wine recognition dataset with chemical analysis",
            "Wine (2D)": "Wine dataset with 2 features for visualization", 
            "Breast Cancer": "Breast cancer diagnosis dataset",
            "Synthetic Classification": "Generated 2D classification data",
            "Synthetic Blobs": "Generated blob clusters for clustering",
            "Synthetic Linear": "Generated linear relationship data",
            "Synthetic Polynomial": "Generated polynomial relationship data",
            "Boston Housing (subset)": "Housing price prediction data"
        }
        return info.get(dataset_name, "Dataset information not available")
