# Machine Learning Algorithm Demonstrator

An interactive Streamlit application that demonstrates popular machine learning algorithms with real-time visualizations and educational content.

## Features

### Clustering Algorithms
- **K-Means**: Centroid-based clustering with customizable number of clusters
- **DBSCAN**: Density-based clustering that finds arbitrary shaped clusters
- **Hierarchical Clustering**: Tree-based clustering with different linkage methods

### Classification Algorithms
- **Logistic Regression**: Linear probabilistic classifier
- **Decision Trees**: Tree-based interpretable classifier
- **Support Vector Machine (SVM)**: Maximum margin classifier with kernel methods
- **K-Nearest Neighbors (KNN)**: Instance-based lazy learning classifier
- **Random Forest**: Ensemble of decision trees

### Regression Algorithms
- **Linear Regression**: Simple linear relationship modeling
- **Polynomial Regression**: Non-linear relationships with polynomial features
- **Support Vector Regression (SVR)**: SVM applied to regression problems

## Datasets Included
- **Iris Dataset**: Classic flower classification dataset
- **Wine Dataset**: Wine recognition with chemical analysis
- **Breast Cancer Dataset**: Medical diagnosis dataset
- **Synthetic Datasets**: Generated data for various algorithm types

## Quick Start

### Local Development

1. **Install Python 3.8+** (if not already installed)

2. **Clone or download** this project to your computer

3. **Install dependencies**:
   ```bash
   pip install streamlit numpy pandas scikit-learn plotly
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

### Alternative Setup
You can also run the setup script:
```bash
python setup_local.py
```

## How to Use

1. **Select Algorithm Category** in the sidebar (Clustering, Classification, or Regression)
2. **Choose Algorithm** from the dropdown menu
3. **Pick a Dataset** suitable for your selected algorithm
4. **Adjust Parameters** using the interactive controls
5. **Click "Train & Visualize"** to see results
6. **Explore** the algorithm description and pseudocode on the right panel

## Educational Content

Each algorithm includes:
- **Detailed Description**: How the algorithm works and when to use it
- **Key Features**: Important characteristics and limitations
- **Pseudocode**: Step-by-step algorithm implementation
- **Interactive Visualizations**: Real-time plots showing algorithm behavior
- **Performance Metrics**: Accuracy, clustering scores, and other relevant metrics

## Hosting Options

### Free Hosting Platforms
- **Streamlit Cloud**: Connect your GitHub repo for automatic deployment
- **Heroku**: Free tier available for small applications
- **Railway**: Simple deployment with GitHub integration
- **Replit**: Online development and hosting environment

### Local Network Access
To access from other devices on your network:
```bash
streamlit run app.py --server.address 0.0.0.0
```

## Requirements

All dependencies are standard Python packages that work across platforms:
- `streamlit` - Web application framework
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning algorithms
- `plotly` - Interactive visualizations

## Project Structure

```
├── app.py              # Main Streamlit application
├── algorithms.py       # Algorithm implementations and descriptions
├── data_loader.py      # Dataset loading and preprocessing
├── visualizations.py   # Plotting and visualization logic
├── setup_local.py      # Local development setup script
└── .streamlit/
    └── config.toml     # Streamlit configuration
```

## Contributing

Feel free to extend this project by:
- Adding new algorithms
- Including more datasets
- Improving visualizations
- Adding more educational content

## License

This project is open source and available for educational use.