import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class Visualizer:
    """Class to handle all visualizations for different algorithms"""
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set1
        
    def create_visualization(self, algorithm_name, algorithm_category, data, target, model, results, feature_names):
        """Create appropriate visualization based on algorithm type"""
        try:
            if algorithm_category == "Clustering":
                return self._create_clustering_viz(algorithm_name, data, model, results, feature_names)
            elif algorithm_category == "Classification":
                return self._create_classification_viz(algorithm_name, data, target, model, results, feature_names)
            elif algorithm_category == "Regression":
                return self._create_regression_viz(algorithm_name, data, target, model, results, feature_names)
        except Exception as e:
            return self._create_error_viz(str(e))
    
    def _create_clustering_viz(self, algorithm_name, data, model, results, feature_names):
        """Create clustering visualization"""
        predictions = results.get('predictions', [])
        
        # If data has more than 2 dimensions, use PCA for visualization
        if data.shape[1] > 2:
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(data)
            x_label = f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)"
            y_label = f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)"
        else:
            data_2d = data
            x_label = feature_names[0] if len(feature_names) > 0 else "Feature 1"
            y_label = feature_names[1] if len(feature_names) > 1 else "Feature 2"
        
        fig = go.Figure()
        
        # Plot data points colored by cluster
        unique_clusters = np.unique(predictions)
        for i, cluster in enumerate(unique_clusters):
            mask = predictions == cluster
            cluster_name = f"Noise" if cluster == -1 else f"Cluster {cluster}"
            color = 'gray' if cluster == -1 else self.colors[i % len(self.colors)]
            
            fig.add_trace(go.Scatter(
                x=data_2d[mask, 0],
                y=data_2d[mask, 1],
                mode='markers',
                name=cluster_name,
                marker=dict(
                    color=color,
                    size=8,
                    opacity=0.7
                ),
                hovertemplate=f"<b>{cluster_name}</b><br>" +
                            f"{x_label}: %{{x:.2f}}<br>" +
                            f"{y_label}: %{{y:.2f}}<extra></extra>"
            ))
        
        # Add centroids for K-Means
        if algorithm_name == "K-Means" and 'centroids' in results:
            centroids = results['centroids']
            if data.shape[1] > 2:
                centroids_2d = pca.transform(centroids)
            else:
                centroids_2d = centroids
                
            fig.add_trace(go.Scatter(
                x=centroids_2d[:, 0],
                y=centroids_2d[:, 1],
                mode='markers',
                name='Centroids',
                marker=dict(
                    color='black',
                    symbol='x',
                    size=15,
                    line=dict(width=2, color='white')
                ),
                hovertemplate="<b>Centroid</b><br>" +
                            f"{x_label}: %{{x:.2f}}<br>" +
                            f"{y_label}: %{{y:.2f}}<extra></extra>"
            ))
        
        fig.update_layout(
            title=f"{algorithm_name} Clustering Results",
            xaxis_title=x_label,
            yaxis_title=y_label,
            showlegend=True,
            height=600,
            hovermode='closest'
        )
        
        return fig
    
    def _create_classification_viz(self, algorithm_name, data, target, model, results, feature_names):
        """Create classification visualization"""
        # Create subplots for training and test data
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training Data', 'Test Data'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # If data has more than 2 dimensions, use PCA
        if data.shape[1] > 2:
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(data)
            x_label = f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)"
            y_label = f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)"
            X_train_2d = pca.transform(results['X_train'])
            X_test_2d = pca.transform(results['X_test'])
        else:
            data_2d = data
            x_label = feature_names[0] if len(feature_names) > 0 else "Feature 1"
            y_label = feature_names[1] if len(feature_names) > 1 else "Feature 2"
            X_train_2d = results['X_train']
            X_test_2d = results['X_test']
        
        # Plot training data
        unique_classes = np.unique(target)
        for i, class_label in enumerate(unique_classes):
            train_mask = results['y_train'] == class_label
            test_mask = results['y_test'] == class_label
            color = self.colors[i % len(self.colors)]
            
            # Training data
            fig.add_trace(go.Scatter(
                x=X_train_2d[train_mask, 0],
                y=X_train_2d[train_mask, 1],
                mode='markers',
                name=f'Class {class_label}',
                marker=dict(color=color, size=8, opacity=0.7),
                legendgroup=f'class_{class_label}',
                hovertemplate=f"<b>Class {class_label}</b><br>" +
                            f"{x_label}: %{{x:.2f}}<br>" +
                            f"{y_label}: %{{y:.2f}}<extra></extra>"
            ), row=1, col=1)
            
            # Test data
            fig.add_trace(go.Scatter(
                x=X_test_2d[test_mask, 0],
                y=X_test_2d[test_mask, 1],
                mode='markers',
                name=f'Class {class_label}',
                marker=dict(color=color, size=8, opacity=0.7),
                legendgroup=f'class_{class_label}',
                showlegend=False,
                hovertemplate=f"<b>Class {class_label}</b><br>" +
                            f"{x_label}: %{{x:.2f}}<br>" +
                            f"{y_label}: %{{y:.2f}}<extra></extra>"
            ), row=1, col=2)
        
        # Add decision boundary for 2D data
        if data.shape[1] == 2 or (data.shape[1] > 2 and hasattr(model, 'predict')):
            try:
                self._add_decision_boundary(fig, model, data_2d, unique_classes, pca if data.shape[1] > 2 else None)
            except:
                pass  # Skip decision boundary if it fails
        
        fig.update_layout(
            title=f"{algorithm_name} Classification Results",
            height=600,
            hovermode='closest'
        )
        
        fig.update_xaxes(title_text=x_label)
        fig.update_yaxes(title_text=y_label)
        
        return fig
    
    def _create_regression_viz(self, algorithm_name, data, target, model, results, feature_names):
        """Create regression visualization"""
        if data.shape[1] == 1:
            # 1D regression - line plot
            return self._create_1d_regression_viz(algorithm_name, data, target, model, results, feature_names)
        else:
            # Multi-dimensional regression - scatter plot with predicted vs actual
            return self._create_nd_regression_viz(algorithm_name, data, target, model, results, feature_names)
    
    def _create_1d_regression_viz(self, algorithm_name, data, target, model, results, feature_names):
        """Create 1D regression visualization"""
        fig = go.Figure()
        
        # Sort data for smooth line plotting
        sort_idx = np.argsort(data.ravel())
        X_sorted = data[sort_idx]
        y_sorted = target[sort_idx]
        
        # Plot actual data points
        fig.add_trace(go.Scatter(
            x=data.ravel(),
            y=target,
            mode='markers',
            name='Actual Data',
            marker=dict(color='blue', size=8, opacity=0.6),
            hovertemplate="<b>Actual Data</b><br>" +
                        f"{feature_names[0]}: %{{x:.2f}}<br>" +
                        "Target: %{y:.2f}<extra></extra>"
        ))
        
        # Plot regression line
        if hasattr(model, 'predict'):
            y_pred_sorted = model.predict(X_sorted)
            fig.add_trace(go.Scatter(
                x=X_sorted.ravel(),
                y=y_pred_sorted,
                mode='lines',
                name='Regression Line',
                line=dict(color='red', width=3),
                hovertemplate="<b>Predicted</b><br>" +
                            f"{feature_names[0]}: %{{x:.2f}}<br>" +
                            "Predicted: %{y:.2f}<extra></extra>"
            ))
        
        fig.update_layout(
            title=f"{algorithm_name} Regression Results",
            xaxis_title=feature_names[0],
            yaxis_title="Target Value",
            height=600,
            hovermode='closest'
        )
        
        return fig
    
    def _create_nd_regression_viz(self, algorithm_name, data, target, model, results, feature_names):
        """Create multi-dimensional regression visualization"""
        predictions = results.get('predictions', [])
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Predicted vs Actual', 'Residuals Plot'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Predicted vs Actual plot
        fig.add_trace(go.Scatter(
            x=target,
            y=predictions,
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', size=8, opacity=0.6),
            hovertemplate="<b>Prediction</b><br>" +
                        "Actual: %{x:.2f}<br>" +
                        "Predicted: %{y:.2f}<extra></extra>"
        ), row=1, col=1)
        
        # Perfect prediction line
        min_val = min(target.min(), predictions.min())
        max_val = max(target.max(), predictions.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ), row=1, col=1)
        
        # Residuals plot
        residuals = target - predictions
        fig.add_trace(go.Scatter(
            x=predictions,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='green', size=8, opacity=0.6),
            hovertemplate="<b>Residual</b><br>" +
                        "Predicted: %{x:.2f}<br>" +
                        "Residual: %{y:.2f}<extra></extra>"
        ), row=1, col=2)
        
        # Zero line for residuals
        fig.add_trace(go.Scatter(
            x=[predictions.min(), predictions.max()],
            y=[0, 0],
            mode='lines',
            name='Zero Line',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ), row=1, col=2)
        
        fig.update_layout(
            title=f"{algorithm_name} Regression Results",
            height=600,
            hovermode='closest'
        )
        
        fig.update_xaxes(title_text="Actual Values", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Values", row=1, col=1)
        fig.update_xaxes(title_text="Predicted Values", row=1, col=2)
        fig.update_yaxes(title_text="Residuals", row=1, col=2)
        
        return fig
    
    def _add_decision_boundary(self, fig, model, data_2d, unique_classes, pca=None):
        """Add decision boundary to classification plot"""
        if not hasattr(model, 'predict'):
            return
        
        # Create a mesh
        h = 0.02
        x_min, x_max = data_2d[:, 0].min() - 1, data_2d[:, 0].max() + 1
        y_min, y_max = data_2d[:, 1].min() - 1, data_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Transform back to original space if using PCA
        if pca is not None:
            # For PCA, we need to inverse transform
            try:
                mesh_points_orig = pca.inverse_transform(mesh_points)
                Z = model.predict(mesh_points_orig)
            except:
                return  # Skip if inverse transform fails
        else:
            Z = model.predict(mesh_points)
        
        Z = Z.reshape(xx.shape)
        
        # Add contour plot for decision boundary
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            z=Z,
            showscale=False,
            opacity=0.3,
            hoverinfo='skip',
            colorscale='Viridis',
            showlegend=False
        ), row=1, col=1)
    
    def _create_error_viz(self, error_message):
        """Create error visualization"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating visualization:<br>{error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Visualization Error",
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
