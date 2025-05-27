import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, silhouette_score, adjusted_rand_score,
    mean_squared_error, r2_score, classification_report
)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class AlgorithmDemonstrator:
    """Class to handle training and evaluation of different ML algorithms"""
    
    def __init__(self):
        self.algorithm_info = {
            # Clustering Algorithms
            "K-Means": {
                "description": """
                K-Means is a centroid-based clustering algorithm that partitions data into k clusters. 
                It iteratively assigns points to the nearest centroid and updates centroids based on 
                cluster means. Best for spherical, well-separated clusters of similar sizes.
                
                **Key Features:**
                - Requires pre-specified number of clusters (k)
                - Minimizes within-cluster sum of squares
                - Assumes spherical clusters
                - Sensitive to initialization and outliers
                """,
                "pseudocode": """
1. Initialize k centroids randomly
2. Repeat until convergence:
   a. Assign each point to nearest centroid
   b. Update centroids to cluster means
   c. Check for convergence
3. Return cluster assignments and centroids
                """
            },
            
            "DBSCAN": {
                "description": """
                DBSCAN (Density-Based Spatial Clustering) groups points that are closely packed 
                while marking outliers in low-density regions. It can find clusters of arbitrary 
                shape and automatically determines the number of clusters.
                
                **Key Features:**
                - No need to specify number of clusters
                - Can find arbitrary shaped clusters
                - Robust to outliers (marks them as noise)
                - Requires two parameters: eps and min_samples
                """,
                "pseudocode": """
1. For each point p:
   a. Find all neighbor points within eps distance
   b. If neighbors >= min_samples, mark as core point
2. For each core point:
   a. Create cluster with core point and neighbors
   b. Expand cluster by adding neighbors of neighbors
3. Mark non-core points as border or noise
                """
            },
            
            "Hierarchical Clustering": {
                "description": """
                Hierarchical clustering creates a tree of clusters by iteratively merging or 
                splitting clusters. Agglomerative (bottom-up) starts with individual points 
                and merges closest clusters based on linkage criteria.
                
                **Key Features:**
                - Creates a hierarchy of clusters (dendrogram)
                - No need to specify clusters beforehand
                - Different linkage methods available
                - Deterministic results
                """,
                "pseudocode": """
1. Start with each point as its own cluster
2. Repeat until one cluster remains:
   a. Find two closest clusters based on linkage
   b. Merge the closest clusters
   c. Update distance matrix
3. Cut dendrogram at desired level for k clusters
                """
            },
            
            # Classification Algorithms
            "Logistic Regression": {
                "description": """
                Logistic Regression uses the logistic function to model the probability of 
                binary or multiclass outcomes. It's a linear classifier that finds the best 
                linear boundary to separate classes using maximum likelihood estimation.
                
                **Key Features:**
                - Probabilistic output (0 to 1)
                - Linear decision boundary
                - No assumptions about feature distributions
                - Regularization options available
                """,
                "pseudocode": """
1. Initialize weights randomly
2. For each iteration:
   a. Calculate predictions using sigmoid function
   b. Compute cost using log-likelihood
   c. Update weights using gradient descent
3. Return trained weights for prediction
                """
            },
            
            "Decision Trees": {
                "description": """
                Decision Trees create a tree-like model of decisions by recursively splitting 
                data based on feature values that best separate classes. Each internal node 
                represents a test, branches represent outcomes, and leaves represent classes.
                
                **Key Features:**
                - Interpretable model structure
                - Handles both numerical and categorical features
                - No assumptions about data distribution
                - Prone to overfitting without pruning
                """,
                "pseudocode": """
1. Start with entire dataset at root
2. For each node:
   a. Find best feature and split value
   b. Split data based on criterion
   c. Create child nodes with subsets
3. Stop when stopping criterion met
4. Assign class labels to leaf nodes
                """
            },
            
            "SVM": {
                "description": """
                Support Vector Machines find the optimal hyperplane that maximally separates 
                classes by maximizing the margin between support vectors. Uses kernel trick 
                for non-linear classification in higher dimensions.
                
                **Key Features:**
                - Maximizes margin between classes
                - Uses only support vectors for decisions
                - Kernel methods for non-linear boundaries
                - Regularization parameter C controls overfitting
                """,
                "pseudocode": """
1. Transform data to higher dimension (kernel)
2. Find hyperplane that maximizes margin:
   a. Identify support vectors
   b. Solve quadratic optimization problem
   c. Determine optimal weights and bias
3. Use support vectors for prediction
                """
            },
            
            "KNN": {
                "description": """
                K-Nearest Neighbors is a lazy learning algorithm that classifies points based 
                on the majority class among k nearest neighbors. It makes no assumptions about 
                data distribution and adapts to local patterns.
                
                **Key Features:**
                - No training phase (lazy learning)
                - Non-parametric method
                - Can capture complex decision boundaries
                - Sensitive to curse of dimensionality
                """,
                "pseudocode": """
1. Store all training data
2. For prediction of new point:
   a. Calculate distance to all training points
   b. Find k nearest neighbors
   c. Take majority vote among neighbors
   d. Return predicted class
                """
            },
            
            "Random Forest": {
                "description": """
                Random Forest builds multiple decision trees using bootstrap sampling and 
                random feature selection. Final prediction is made by majority voting 
                (classification) or averaging (regression) across all trees.
                
                **Key Features:**
                - Ensemble of decision trees
                - Reduces overfitting through averaging
                - Built-in feature importance
                - Handles missing values and mixed data types
                """,
                "pseudocode": """
1. For each tree (1 to n_estimators):
   a. Bootstrap sample from training data
   b. Build decision tree with random features
   c. Train tree without pruning
2. For prediction:
   a. Get prediction from each tree
   b. Take majority vote (classification)
                """
            },
            
            # Regression Algorithms
            "Linear Regression": {
                "description": """
                Linear Regression models the relationship between features and target as a 
                linear equation. It finds the best-fitting line through data points by 
                minimizing the sum of squared residuals.
                
                **Key Features:**
                - Simple and interpretable
                - Assumes linear relationship
                - Closed-form solution available
                - Sensitive to outliers
                """,
                "pseudocode": """
1. Set up linear equation: y = wx + b
2. Minimize sum of squared errors:
   a. Calculate residuals (actual - predicted)
   b. Square residuals and sum
   c. Find weights that minimize sum
3. Return optimal weights and bias
                """
            },
            
            "Polynomial Regression": {
                "description": """
                Polynomial Regression extends linear regression by adding polynomial terms 
                of the features. It can capture non-linear relationships while still using 
                linear regression techniques on transformed features.
                
                **Key Features:**
                - Captures non-linear relationships
                - Uses polynomial feature transformation
                - Risk of overfitting with high degrees
                - Still a linear model in transformed space
                """,
                "pseudocode": """
1. Transform features to polynomial terms
2. Apply linear regression to transformed features:
   a. Create polynomial features (x, x², x³, ...)
   b. Fit linear model to expanded features
   c. Minimize squared error on polynomial terms
3. Return coefficients for polynomial equation
                """
            },
            
            "Support Vector Regression": {
                "description": """
                Support Vector Regression applies SVM principles to regression by finding 
                a function that deviates from targets by at most epsilon, while being 
                as flat as possible. Uses kernel methods for non-linear regression.
                
                **Key Features:**
                - Robust to outliers
                - Uses epsilon-insensitive loss
                - Kernel methods for non-linearity
                - Sparse solution using support vectors
                """,
                "pseudocode": """
1. Define epsilon-tube around targets
2. Find function with maximum margin:
   a. Minimize weights while staying in tube
   b. Use slack variables for violations
   c. Apply kernel transformation if needed
3. Use support vectors for prediction
                """
            }
        }
    
    def get_algorithm_info(self, algorithm_name):
        """Get description and pseudocode for an algorithm"""
        info = self.algorithm_info.get(algorithm_name, {})
        return info.get("description", ""), info.get("pseudocode", "")
    
    def train_algorithm(self, algorithm_name, data, target, params):
        """Train the specified algorithm and return model and results"""
        try:
            if algorithm_name == "K-Means":
                return self._train_kmeans(data, params)
            elif algorithm_name == "DBSCAN":
                return self._train_dbscan(data, params)
            elif algorithm_name == "Hierarchical Clustering":
                return self._train_hierarchical(data, params)
            elif algorithm_name == "Logistic Regression":
                return self._train_logistic(data, target, params)
            elif algorithm_name == "Decision Trees":
                return self._train_decision_tree(data, target, params)
            elif algorithm_name == "SVM":
                return self._train_svm(data, target, params)
            elif algorithm_name == "KNN":
                return self._train_knn(data, target, params)
            elif algorithm_name == "Random Forest":
                return self._train_random_forest(data, target, params)
            elif algorithm_name == "Linear Regression":
                return self._train_linear_regression(data, target, params)
            elif algorithm_name == "Polynomial Regression":
                return self._train_polynomial_regression(data, target, params)
            elif algorithm_name == "Support Vector Regression":
                return self._train_svr(data, target, params)
        except Exception as e:
            return None, {"error": str(e)}
    
    def _train_kmeans(self, data, params):
        """Train K-Means clustering"""
        model = KMeans(**params)
        predictions = model.fit_predict(data)
        
        results = {
            "predictions": predictions,
            "centroids": model.cluster_centers_,
            "inertia": model.inertia_
        }
        
        if len(np.unique(predictions)) > 1:
            results["silhouette_score"] = silhouette_score(data, predictions)
        
        return model, results
    
    def _train_dbscan(self, data, params):
        """Train DBSCAN clustering"""
        model = DBSCAN(**params)
        predictions = model.fit_predict(data)
        
        results = {
            "predictions": predictions,
            "n_clusters": len(set(predictions)) - (1 if -1 in predictions else 0),
            "n_noise": list(predictions).count(-1)
        }
        
        if len(np.unique(predictions)) > 1:
            results["silhouette_score"] = silhouette_score(data, predictions)
        
        return model, results
    
    def _train_hierarchical(self, data, params):
        """Train Hierarchical clustering"""
        model = AgglomerativeClustering(**params)
        predictions = model.fit_predict(data)
        
        results = {
            "predictions": predictions,
            "n_clusters": len(np.unique(predictions))
        }
        
        if len(np.unique(predictions)) > 1:
            results["silhouette_score"] = silhouette_score(data, predictions)
        
        return model, results
    
    def _train_logistic(self, data, target, params):
        """Train Logistic Regression"""
        X_train, X_test, y_train, y_test = train_test_split(
            data, target, test_size=0.2, random_state=42
        )
        
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        results = {
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "predictions": model.predict(data),
            "probabilities": model.predict_proba(data) if hasattr(model, 'predict_proba') else None,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "train_pred": train_pred, "test_pred": test_pred
        }
        
        return model, results
    
    def _train_decision_tree(self, data, target, params):
        """Train Decision Tree"""
        X_train, X_test, y_train, y_test = train_test_split(
            data, target, test_size=0.2, random_state=42
        )
        
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        results = {
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "predictions": model.predict(data),
            "feature_importance": model.feature_importances_,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "train_pred": train_pred, "test_pred": test_pred
        }
        
        return model, results
    
    def _train_svm(self, data, target, params):
        """Train SVM"""
        X_train, X_test, y_train, y_test = train_test_split(
            data, target, test_size=0.2, random_state=42
        )
        
        model = SVC(**params)
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        results = {
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "predictions": model.predict(data),
            "n_support": model.n_support_,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "train_pred": train_pred, "test_pred": test_pred
        }
        
        return model, results
    
    def _train_knn(self, data, target, params):
        """Train KNN"""
        X_train, X_test, y_train, y_test = train_test_split(
            data, target, test_size=0.2, random_state=42
        )
        
        model = KNeighborsClassifier(**params)
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        results = {
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "predictions": model.predict(data),
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "train_pred": train_pred, "test_pred": test_pred
        }
        
        return model, results
    
    def _train_random_forest(self, data, target, params):
        """Train Random Forest"""
        X_train, X_test, y_train, y_test = train_test_split(
            data, target, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        results = {
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "predictions": model.predict(data),
            "feature_importance": model.feature_importances_,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "train_pred": train_pred, "test_pred": test_pred
        }
        
        return model, results
    
    def _train_linear_regression(self, data, target, params):
        """Train Linear Regression"""
        X_train, X_test, y_train, y_test = train_test_split(
            data, target, test_size=0.2, random_state=42
        )
        
        model = LinearRegression(**params)
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        results = {
            "train_r2": r2_score(y_train, train_pred),
            "test_r2": r2_score(y_test, test_pred),
            "train_mse": mean_squared_error(y_train, train_pred),
            "test_mse": mean_squared_error(y_test, test_pred),
            "predictions": model.predict(data),
            "coefficients": model.coef_,
            "intercept": model.intercept_,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "train_pred": train_pred, "test_pred": test_pred
        }
        
        return model, results
    
    def _train_polynomial_regression(self, data, target, params):
        """Train Polynomial Regression"""
        X_train, X_test, y_train, y_test = train_test_split(
            data, target, test_size=0.2, random_state=42
        )
        
        degree = params.pop('degree', 2)
        poly_features = PolynomialFeatures(degree=degree)
        
        model = Pipeline([
            ('poly', poly_features),
            ('linear', LinearRegression(**params))
        ])
        
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        results = {
            "train_r2": r2_score(y_train, train_pred),
            "test_r2": r2_score(y_test, test_pred),
            "train_mse": mean_squared_error(y_train, train_pred),
            "test_mse": mean_squared_error(y_test, test_pred),
            "predictions": model.predict(data),
            "degree": degree,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "train_pred": train_pred, "test_pred": test_pred
        }
        
        return model, results
    
    def _train_svr(self, data, target, params):
        """Train Support Vector Regression"""
        X_train, X_test, y_train, y_test = train_test_split(
            data, target, test_size=0.2, random_state=42
        )
        
        model = SVR(**params)
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        results = {
            "train_r2": r2_score(y_train, train_pred),
            "test_r2": r2_score(y_test, test_pred),
            "train_mse": mean_squared_error(y_train, train_pred),
            "test_mse": mean_squared_error(y_test, test_pred),
            "predictions": model.predict(data),
            "n_support": len(model.support_),
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "train_pred": train_pred, "test_pred": test_pred
        }
        
        return model, results
