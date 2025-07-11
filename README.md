# 🚗 Car Price Prediction System

A comprehensive machine learning application that predicts car prices using data from Cars24 and Spinny. The system combines multiple datasets, trains various ML models, and provides an interactive Streamlit interface for price predictions.

## 📋 Features

- **Data Integration**: Combines data from Cars24 and Spinny datasets
- **Advanced ML Models**: Multiple regression models including Random Forest, Gradient Boosting, Linear Regression, Ridge, Lasso, and SVR
- **Interactive Interface**: Beautiful Streamlit web application with multiple pages
- **Data Analysis**: Comprehensive data visualization and analysis
- **Price Prediction**: Real-time car price predictions with confidence scores
- **Model Comparison**: Performance comparison across different algorithms

## 🛠️ Installation

1. **Clone or download the project files**
   ```bash
   # Make sure you have the following files in your directory:
   # - pricePredictor.py
   # - combined_Cars24_data.csv
   # - combined_spinny_data.csv
   # - requirements.txt
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run pricePredictor.py
   ```

## 🚀 Usage

### 1. Home Page
- Overview of the system
- Basic statistics about the dataset
- Quick navigation to other features

### 2. Data Analysis Page
- **Dataset Overview**: Shape, columns, and missing values
- **Price Distribution**: Histogram of car prices
- **Brand Analysis**: Average prices by car brand
- **Year Trends**: Price trends over years
- **Correlation Analysis**: Feature correlation heatmap

### 3. Model Training Page
- **Train Models**: Click to train all ML models
- **Performance Comparison**: R² scores, RMSE, and MAE for each model
- **Visual Results**: Bar chart showing model performance
- **Best Model Selection**: Automatic selection of the best performing model

### 4. Price Prediction Page
- **Input Form**: Enter car details (brand, model, year, transmission, fuel type, etc.)
- **Real-time Prediction**: Get instant price predictions
- **Price Analysis**: Compare with similar cars in the dataset
- **Confidence Metrics**: Model confidence and price ranges
- **Feature Importance**: Key factors affecting the price

## 📊 Dataset Information

The system uses two main datasets:
- **Cars24 Data**: Contains car listings from Cars24 platform
- **Spinny Data**: Contains car listings from Spinny platform

### Features Used for Prediction:
- **Make**: Car brand (e.g., Maruti, Hyundai, Honda)
- **Model**: Car model name
- **Year**: Manufacturing year
- **Transmission Type**: Manual or Automatic
- **Fuel Type**: Petrol, Diesel, CNG, Hybrid, Electric
- **Ownership**: Number of previous owners
- **EMI**: Monthly EMI amount
- **Odometer**: Total kilometers driven
- **City**: Location of the car

## 🤖 Machine Learning Models

The system trains and compares the following models:

1. **Random Forest Regressor**: Ensemble method using multiple decision trees
2. **Gradient Boosting Regressor**: Sequential ensemble learning
3. **Linear Regression**: Basic linear model
4. **Ridge Regression**: Linear regression with L2 regularization
5. **Lasso Regression**: Linear regression with L1 regularization
6. **Support Vector Regression (SVR)**: Non-linear regression using SVM

## 📈 Model Performance

The system automatically selects the best performing model based on R² score. Typical performance metrics:
- **R² Score**: Measures how well the model explains the variance in prices
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)

## 🎯 How to Get Accurate Predictions

1. **Train Models First**: Always train the models before making predictions
2. **Use Realistic Values**: Enter realistic values for all fields
3. **Check Similar Cars**: Compare your prediction with similar cars in the dataset
4. **Consider Price Range**: The system provides a price range (±10% of predicted price)

## 🔧 Technical Details

### Data Preprocessing:
- Handles missing values using median imputation
- Removes outliers using IQR method
- Standardizes categorical variables (transmission, fuel type)
- Encodes categorical features using Label Encoding

### Model Training:
- 80-20 train-test split
- Feature scaling for SVR model
- Cross-validation for model evaluation
- Automatic hyperparameter selection

### Prediction Pipeline:
- Input validation and preprocessing
- Feature encoding for categorical variables
- Model prediction with confidence scoring
- Post-processing to ensure realistic prices

## 🐛 Troubleshooting

### Common Issues:

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Loading Errors**: Ensure CSV files are in the same directory as the script

3. **Model Training Issues**: Check if the dataset is properly formatted

4. **Prediction Errors**: Make sure models are trained before making predictions

### Performance Tips:

1. **Large Datasets**: The system can handle large datasets but may take longer to train
2. **Memory Usage**: Close other applications if you encounter memory issues
3. **Browser Compatibility**: Use modern browsers for the best experience

## 📝 Example Usage

```python
# The system is designed to be used through the Streamlit interface
# However, you can also use the CarPricePredictor class directly:

from pricePredictor import CarPricePredictor

# Initialize predictor
predictor = CarPricePredictor()

# Load and combine data
data = predictor.load_and_combine_data()

# Prepare features
X, y = predictor.prepare_features(data)

# Train models
models = predictor.train_models(X, y)

# Make prediction
input_data = {
    'make': 'Maruti',
    'model': 'Swift',
    'year': 2020,
    'transmissiontype': 'Manual',
    'fueltype': 'Petrol',
    'ownership': 1,
    'emi': 15000,
    'odometer': 50000,
    'city': 'Mumbai'
}

predicted_price = predictor.predict_price(input_data)
print(f"Predicted Price: ₹{predicted_price:,.0f}")
```

## 🤝 Contributing

Feel free to contribute to this project by:
- Adding new ML models
- Improving the UI/UX
- Adding more data sources
- Optimizing performance
- Adding new features

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Data sources: Cars24 and Spinny
- Machine learning libraries: scikit-learn
- Web framework: Streamlit
- Visualization: Plotly

---

**Happy Car Price Predicting! 🚗💰** 