# ML Data Analytics App

An interactive Streamlit-based machine learning application that allows users 
to upload their own dataset and perform end-to-end analysis — from data 
cleaning to predictive modeling — without writing any code.

## Features
- **File Upload & Auto Data Cleaning**: Upload CSV/Excel files, automatically 
  detect and separate numeric, categorical, and date columns
- **Supervised Learning**: Train classification or regression models 
  (Random Forest) with automatic target detection, feature importance, 
  and performance metrics (accuracy, RMSE, R²)
- **Unsupervised Learning**: K-Means clustering with PCA visualization, 
  silhouette score, and Elbow Method trend analysis for optimal cluster selection
- **Next Purchase Prediction**: Predict when a customer is likely to make 
  their next purchase based on historical order patterns
- **Interactive Visualizations**: Real-time charts using Streamlit's native 
  charting and Matplotlib

## Tech Stack
- Python, Streamlit, Pandas, NumPy
- Scikit-learn (RandomForest, KMeans, PCA, StandardScaler)
- Matplotlib

## How to Run
```bash
pip install -r requirements.txt
streamlit run prediction.py
```

## Use Case
Built for exploring retail/sales datasets (e.g. SuperMart-style data) to 
demonstrate practical data analytics and ML skills for Data Analyst / 
Business Research Analyst roles.
