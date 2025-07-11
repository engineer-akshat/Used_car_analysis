import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class CarPricePredictor:
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_columns = []
        
    def load_and_combine_data(self):
        """Load and combine both datasets"""
        try:
            # Load both datasets
            cars24_data = pd.read_csv('combined_Cars24_data.csv')
            spinny_data = pd.read_csv('combined_spinny_data.csv')
            
            # Standardize column names
            cars24_data.columns = cars24_data.columns.str.lower()
            spinny_data.columns = spinny_data.columns.str.lower()
            
            # Add source identifier
            cars24_data['source'] = 'Cars24'
            spinny_data['source'] = 'Spinny'
            
            # Combine datasets
            combined_data = pd.concat([cars24_data, spinny_data], ignore_index=True)
            
            # Clean the data
            combined_data = self.clean_data(combined_data)
            
            return combined_data
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def clean_data(self, data):
        """Clean and preprocess the data"""
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Handle missing values
        data['emi'] = data['emi'].fillna(data['emi'].median())
        data['odometer'] = data['odometer'].fillna(data['odometer'].median())
        
        # Clean make and model names
        data['make'] = data['make'].str.strip().str.title()
        data['model'] = data['model'].str.strip().str.title()
        
        # Standardize transmission types
        data['transmissiontype'] = data['transmissiontype'].str.lower()
        data['transmissiontype'] = data['transmissiontype'].map({
            'manual': 'Manual',
            'automatic': 'Automatic',
            'amt': 'Automatic',
            'cvt': 'Automatic',
            'dct': 'Automatic',
            'imt': 'Manual'
        }).fillna('Manual')
        
        # Standardize fuel types
        data['fueltype'] = data['fueltype'].str.lower()
        data['fueltype'] = data['fueltype'].map({
            'petrol': 'Petrol',
            'diesel': 'Diesel',
            'cng': 'CNG',
            'petrol+cng': 'CNG',
            'hybrid': 'Hybrid',
            'electric': 'Electric'
        }).fillna('Petrol')
        
        # Remove outliers from price and odometer
        Q1_price = data['score'].quantile(0.25)
        Q3_price = data['score'].quantile(0.75)
        IQR_price = Q3_price - Q1_price
        data = data[(data['score'] >= Q1_price - 1.5 * IQR_price) & 
                   (data['score'] <= Q3_price + 1.5 * IQR_price)]
        
        Q1_odo = data['odometer'].quantile(0.25)
        Q3_odo = data['odometer'].quantile(0.75)
        IQR_odo = Q3_odo - Q1_odo
        data = data[(data['odometer'] >= Q1_odo - 1.5 * IQR_odo) & 
                   (data['odometer'] <= Q3_odo + 1.5 * IQR_odo)]
        
        return data
    
    def prepare_features(self, data):
        """Prepare features for modeling"""
        # Select relevant features
        feature_cols = ['make', 'model', 'year', 'transmissiontype', 'fueltype', 
                       'ownership', 'emi', 'odometer', 'city']
        
        # Create feature matrix
        X = data[feature_cols].copy()
        y = data['score']
        
        # Encode categorical variables
        categorical_cols = ['make', 'model', 'transmissiontype', 'fueltype', 'city']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                # Handle new categories in test data
                X[col] = X[col].astype(str)
                X[col] = X[col].map(lambda x: x if x in self.label_encoders[col].classes_ else 'Unknown')
                X[col] = self.label_encoders[col].transform(X[col])
        
        self.feature_columns = X.columns.tolist()
        
        return X, y
    
    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'SVR': SVR(kernel='rbf', C=100, gamma='scale')
        }
        
        best_score = -np.inf
        best_model_name = None
        
        # Train and evaluate models
        for name, model in models.items():
            try:
                if name in ['SVR']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                self.models[name] = {
                    'model': model,
                    'r2': r2,
                    'mse': mse,
                    'mae': mae,
                    'rmse': np.sqrt(mse)
                }
                
                if r2 > best_score:
                    best_score = r2
                    best_model_name = name
                    
            except Exception as e:
                st.warning(f"Error training {name}: {str(e)}")
        
        # Select best model
        if best_model_name:
            self.best_model = self.models[best_model_name]['model']
            st.success(f"Best model: {best_model_name} (R² = {best_score:.4f})")
        
        return self.models
    
    def predict_price(self, input_data):
        """Predict car price using the best model"""
        if self.best_model is None:
            return None
        
        # Prepare input data
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        for col in ['make', 'model', 'transmissiontype', 'fueltype', 'city']:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(str)
                if col in self.label_encoders:
                    # Handle unknown categories
                    if input_df[col].iloc[0] not in self.label_encoders[col].classes_:
                        input_df[col] = 'Unknown'
                    input_df[col] = self.label_encoders[col].transform(input_df[col])
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training data
        input_df = input_df[self.feature_columns]
        
        # Scale features if using SVR
        if isinstance(self.best_model, SVR):
            input_df = self.scaler.transform(input_df)
        
        # Make prediction
        prediction = self.best_model.predict(input_df)[0]
        
        return max(0, prediction)  # Ensure non-negative price

def main():
    st.set_page_config(
        page_title="Car Price Predictor",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 Car Price Prediction System")
    st.markdown("---")
    
    # Initialize predictor with session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = CarPricePredictor()
    
    predictor = st.session_state.predictor
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "💰 Price Prediction"]
    )
    
    # Show model training status in sidebar
    st.sidebar.markdown("---")
    if st.session_state.get('models_trained', False):
        st.sidebar.success("✅ Models Trained")
        if 'models' in st.session_state and predictor.best_model:
            best_model_name = None
            best_score = -1
            for name, metrics in st.session_state.models.items():
                if metrics['r2'] > best_score:
                    best_score = metrics['r2']
                    best_model_name = name
            if best_model_name:
                st.sidebar.info(f"Best Model: {best_model_name}")
                st.sidebar.metric("R² Score", f"{best_score:.4f}")
    else:
        st.sidebar.warning("⚠️ Models Not Trained")
        st.sidebar.info("Train models to use price prediction")
    
    if page == "🏠 Home":
        st.header("Welcome to Car Price Predictor!")
        st.markdown("""
        This application helps you predict car prices using machine learning models trained on data from Cars24 and Spinny.
        
        **Features:**
        - 📊 Comprehensive data analysis
        - 🤖 Multiple ML models (Random Forest, Gradient Boosting, etc.)
        - 💰 Interactive price prediction
        - 📈 Model performance comparison
        
        **How to use:**
        1. Go to "Data Analysis" to explore the dataset
        2. Train models in "Model Training"
        3. Predict prices in "Price Prediction"
        """)
        
        # Load and show basic stats
        with st.spinner("Loading data..."):
            data = predictor.load_and_combine_data()
            if data is not None:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Cars", len(data))
                with col2:
                    st.metric("Average Price", f"₹{data['score'].mean():,.0f}")
                with col3:
                    st.metric("Brands", data['make'].nunique())
                with col4:
                    st.metric("Cities", data['city'].nunique())
    
    elif page == "📊 Data Analysis":
        st.header("Data Analysis")
        
        with st.spinner("Loading data..."):
            data = predictor.load_and_combine_data()
            
        if data is not None:
            # Data overview
            st.subheader("Dataset Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Dataset Shape:**", data.shape)
                st.write("**Columns:**", list(data.columns))
            
            with col2:
                st.write("**Missing Values:**")
                missing_data = data.isnull().sum()
                st.write(missing_data[missing_data > 0])
            
            # Price distribution
            st.subheader("Price Distribution")
            fig = px.histogram(data, x='score', nbins=50, title="Car Price Distribution")
            fig.update_layout(xaxis_title="Price (₹)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
            
            # Price by brand
            st.subheader("Average Price by Brand (Top 15)")
            brand_prices = data.groupby('make')['score'].mean().sort_values(ascending=False).head(15)
            fig = px.bar(x=brand_prices.values, y=brand_prices.index, orientation='h',
                        title="Average Price by Brand")
            fig.update_layout(xaxis_title="Average Price (₹)", yaxis_title="Brand")
            st.plotly_chart(fig, use_container_width=True)
            
            # Price by year
            st.subheader("Price Trend by Year")
            year_prices = data.groupby('year')['score'].mean().sort_index()
            fig = px.line(x=year_prices.index, y=year_prices.values, title="Average Price by Year")
            fig.update_layout(xaxis_title="Year", yaxis_title="Average Price (₹)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("Feature Correlations")
            numeric_data = data.select_dtypes(include=[np.number])
            correlation_matrix = numeric_data.corr()
            fig = px.imshow(correlation_matrix, title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🤖 Model Training":
        st.header("Model Training")
        
        # Check if models are already trained
        if st.session_state.get('models_trained', False):
            st.success("✅ Models are already trained!")
            st.info("You can now use the Price Prediction page or retrain the models below.")
            
            # Display existing model results
            if 'models' in st.session_state:
                st.subheader("Current Model Performance")
                models = st.session_state.models
                
                # Create comparison table
                results_data = []
                for name, metrics in models.items():
                    results_data.append({
                        'Model': name,
                        'R² Score': f"{metrics['r2']:.4f}",
                        'RMSE': f"{metrics['rmse']:,.0f}",
                        'MAE': f"{metrics['mae']:,.0f}"
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
        
        if st.button("Train Models" if not st.session_state.get('models_trained', False) else "Retrain Models"):
            with st.spinner("Loading and preparing data..."):
                data = predictor.load_and_combine_data()
                
            if data is not None:
                with st.spinner("Preparing features..."):
                    X, y = predictor.prepare_features(data)
                
                with st.spinner("Training models..."):
                    models = predictor.train_models(X, y)
                
                # Store models in session state
                st.session_state.models = models
                st.session_state.models_trained = True
                
                # Display model results
                st.subheader("Model Performance Comparison")
                
                # Create comparison table
                results_data = []
                for name, metrics in models.items():
                    results_data.append({
                        'Model': name,
                        'R² Score': f"{metrics['r2']:.4f}",
                        'RMSE': f"{metrics['rmse']:,.0f}",
                        'MAE': f"{metrics['mae']:,.0f}"
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Visualize model performance
                fig = go.Figure()
                model_names = list(models.keys())
                r2_scores = [models[name]['r2'] for name in model_names]
                
                fig.add_trace(go.Bar(
                    x=model_names,
                    y=r2_scores,
                    text=[f"{score:.4f}" for score in r2_scores],
                    textposition='auto',
                    name='R² Score'
                ))
                
                fig.update_layout(
                    title="Model Performance (R² Scores)",
                    xaxis_title="Model",
                    yaxis_title="R² Score",
                    yaxis_range=[0, 1]
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("Models trained successfully! You can now use the Price Prediction page.")
    
    elif page == "💰 Price Prediction":
        st.header("Car Price Prediction")
        
        # Check if models are trained
        if not st.session_state.get('models_trained', False) or predictor.best_model is None:
            st.warning("⚠️ Please train the models first in the 'Model Training' page.")
            st.info("💡 Go to the 'Model Training' page and click 'Train Models' button.")
            return
        
        # Input form
        st.subheader("Enter Car Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Load data to get unique values
            data = predictor.load_and_combine_data()
            if data is None:
                return
            
            makes = sorted(data['make'].unique())
            models = sorted(data['model'].unique())
            cities = sorted(data['city'].unique())
            
            make = st.selectbox("Car Brand", makes)
            
            # Filter models based on selected make
            make_models = sorted(data[data['make'] == make]['model'].unique())
            model = st.selectbox("Car Model", make_models)
            
            year = st.slider("Year", min_value=2010, max_value=2024, value=2020)
            
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Hybrid", "Electric"])
        
        with col2:
            ownership = st.slider("Number of Previous Owners", min_value=1, max_value=5, value=1)
            
            emi = st.number_input("EMI Amount (₹)", min_value=0, max_value=100000, value=15000)
            
            odometer = st.number_input("Odometer Reading (km)", min_value=0, max_value=500000, value=50000)
            
            city = st.selectbox("City", cities)
        
        # Prediction button
        if st.button("Predict Price", type="primary"):
            with st.spinner("Calculating price..."):
                input_data = {
                    'make': make,
                    'model': model,
                    'year': year,
                    'transmissiontype': transmission,
                    'fueltype': fuel_type,
                    'ownership': ownership,
                    'emi': emi,
                    'odometer': odometer,
                    'city': city
                }
                
                predicted_price = predictor.predict_price(input_data)
                
                if predicted_price is not None:
                    st.success("Price Prediction Complete!")
                    
                    # Display result
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Predicted Price", f"₹{predicted_price:,.0f}")
                    
                    with col2:
                        # Calculate price range
                        price_range = predicted_price * 0.1  # ±10%
                        st.metric("Price Range", f"₹{predicted_price - price_range:,.0f} - ₹{predicted_price + price_range:,.0f}")
                    
                    with col3:
                        # Show confidence based on model performance
                        if 'Random Forest' in predictor.models:
                            confidence = predictor.models['Random Forest']['r2'] * 100
                            st.metric("Model Confidence", f"{confidence:.1f}%")
                    
                    # Additional insights
                    st.subheader("Price Analysis")
                    
                    # Compare with similar cars
                    similar_cars = data[
                        (data['make'] == make) & 
                        (data['model'] == model) &
                        (data['year'] >= year - 2) &
                        (data['year'] <= year + 2)
                    ]
                    
                    if len(similar_cars) > 0:
                        avg_similar_price = similar_cars['score'].mean()
                        price_diff = predicted_price - avg_similar_price
                        price_diff_pct = (price_diff / avg_similar_price) * 100
                        
                        st.write(f"**Average price of similar cars:** ₹{avg_similar_price:,.0f}")
                        st.write(f"**Price difference:** ₹{price_diff:,.0f} ({price_diff_pct:+.1f}%)")
                        
                        if abs(price_diff_pct) < 10:
                            st.success("✅ Price prediction is within normal range")
                        elif price_diff_pct > 10:
                            st.warning("⚠️ Predicted price is higher than average")
                        else:
                            st.info("ℹ️ Predicted price is lower than average")
                    
                    # Price factors
                    st.subheader("Key Price Factors")
                    
                    # Create a simple feature importance visualization
                    if hasattr(predictor.best_model, 'feature_importances_'):
                        feature_importance = pd.DataFrame({
                            'Feature': predictor.feature_columns,
                            'Importance': predictor.best_model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(feature_importance.head(10), x='Importance', y='Feature', orientation='h',
                                   title="Top 10 Most Important Features")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Error making prediction. Please try again.")

if __name__ == "__main__":
    main()
