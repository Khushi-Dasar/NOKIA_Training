# Setup Instructions

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

1. Clone or download this repository to your local machine

2. Navigate to the project directory:
```bash
cd TELECOM_CHURN_CAPSTONE
```

3. Install required packages:
```bash
pip install -r req.txt
```

## Project Structure

- `Submission.ipynb` - Main Jupyter notebook with the complete analysis pipeline
- `app.py` - Flask application for serving predictions
- `telecom_master.csv` - Master dataset after preprocessing
- `churn_model.pkl` - Trained machine learning model
- `feature_names.pkl` - Feature names for model input
- `DATA/` - Directory containing raw data files
  - `customers.csv` - Customer information
  - `usage_data.csv` - Usage statistics
  - `complaints.csv` - Customer complaints
  - `billing.csv` - Billing information

## Running the Analysis

### Option 1: Jupyter Notebook

Open and run the notebook:
```bash
jupyter notebook Submission.ipynb
```

Execute cells sequentially from top to bottom.

### Option 2: Flask Application

Start the web application:
```bash
python app.py
```

The application will start on http://localhost:5000

## Model Training

The notebook includes the following steps:

1. Data loading and exploration
2. Data merging and preprocessing
3. Feature engineering
4. Model training (Logistic Regression, Decision Tree, Random Forest)
5. Hyperparameter tuning
6. Model evaluation
7. Prediction on new data

## Making Predictions

To make predictions on new data, ensure your CSV file has the same structure as the training data and run the prediction cells in the notebook.

## Output Files

- `churn_predictions.csv` - Predictions with probabilities
- `evaluation.json` - Model evaluation metrics
- `telecom_master.csv` - Processed master dataset
