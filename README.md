# Time for Claim Resolution Predictor

⏱️ A machine learning application that predicts insurance claim resolution times

## Overview

This application uses machine learning to predict how long it will take to resolve insurance claims based on various factors. It provides data exploration, prediction capabilities, and model interpretation using SHAP values to explain the predictions.

## Features

- **Data Exploration**: Visualize and analyze insurance claim data
- **Prediction**: Estimate claim resolution time based on claim details
- **Model Interpretation**: Understand what factors influence prediction results

## Live Demo

The application is deployed and available at: [Your Streamlit App URL]

## Getting Started

### Prerequisites

- Python 3.8+
- Required libraries listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```
   git clone [your-repository-url]
   cd [repository-name]
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application locally:
   ```
   streamlit run app.py
   ```

## How to Use

### Data Upload
- Upload your CSV file through the sidebar
- Default dataset will be used if no file is uploaded

### Data Exploration
- View key metrics and statistics about the dataset
- Analyze claim resolution time distributions
- Discover correlations between different factors

### Prediction
1. Enter claim details in the form
2. Click the "Predict" button
3. View the estimated resolution time

### Model Interpretation
- Understand which features impact predictions the most
- View SHAP visualizations to see how each factor affects individual predictions
- Use What-If analysis to explore how changing inputs affects outcomes

## Model Information

- Model Type: XGBoost Regressor
- Features Used: [List key features]
- Training Dataset: [Brief description of data]
- Performance Metrics: [RMSE, MAE, R² values]

## Technical Details

The application is built with:
- Streamlit for the web interface
- Pandas and NumPy for data processing
- Plotly for interactive visualizations
- XGBoost for the prediction model
- SHAP for model interpretation

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── requirements.txt        # Dependencies
├── style.css               # Custom CSS styling
├── saved_xgb_model.pkl     # Trained XGBoost model
├── encoder_scaler.pkl      # Preprocessing components
├── data/
│   └── Insurance_data.csv  # Default dataset
└── src/
    ├── __init__.py         # Package initialization
    ├── explore_page.py     # Data exploration components
    ├── predict_page.py     # Prediction functionality
    ├── xai_page.py         # Model interpretation (XAI)
    ├── train_model.py      # Model training script
    └── utils.py            # Shared utilities
```

## Development

### Training New Models
To train a new model:
```
python src/train_model.py
```

### Customization
- Modify `style.css` to change the app's appearance
- Update feature ranges in `src/utils.py`
- Add new visualizations in `src/explore_page.py`

## License

[Your chosen license]

## Acknowledgments

- Data source: [A.K.I]
- Strathmore University Masters Data Science class of 2023
