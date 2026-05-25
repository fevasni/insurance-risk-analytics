"""
Task 4: Statistical Modeling & Risk-Based Pricing
Comprehensive predictive modeling pipeline for claim severity and premium optimization.

This script implements:
1. Data preparation (handling missing values, feature engineering, encoding)
2. Claim Severity Prediction (Linear Regression, Random Forest, XGBoost)
3. Premium Optimization (P(claim) classifier + severity model)
4. Feature importance analysis using SHAP
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Modeling imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Visualization and analysis
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InsuranceDataPreprocessor:
    """Handles data loading, cleaning, and feature engineering for insurance modeling."""
    
    def __init__(self, data_path, delimiter='|'):
        """Initialize preprocessor with data file path."""
        self.data_path = data_path
        self.delimiter = delimiter
        self.df = None
        self.df_processed = None
        self.feature_columns = None
        self.target_column = 'TotalClaims'
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load insurance data with proper delimiter handling."""
        logger.info(f"Loading data from {self.data_path}...")
        try:
            self.df = pd.read_csv(
                self.data_path, 
                delimiter=self.delimiter,
                low_memory=False
            )
            logger.info(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return self.df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def remove_duplicate_rows(self):
        """Remove duplicate rows based on all columns."""
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial_count - len(self.df)
        logger.info(f"Removed {removed} duplicate rows. Remaining: {len(self.df)}")
        return self.df
    
    def handle_missing_values(self):
        """Handle missing values based on column characteristics."""
        logger.info("Handling missing values...")
        
        # Columns with >50% missing - remove (low information value)
        missing_pct = (self.df.isnull().sum() / len(self.df)) * 100
        high_missing = missing_pct[missing_pct > 50].index.tolist()
        logger.info(f"Removing columns with >50% missing: {high_missing}")
        self.df = self.df.drop(columns=high_missing)
        
        # For remaining columns with missing values
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                missing_pct_col = (self.df[col].isnull().sum() / len(self.df)) * 100
                
                if self.df[col].dtype in ['float64', 'int64']:
                    # Numerical: impute with median
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                    logger.info(f"  {col}: imputed {missing_pct_col:.2f}% missing with median")
                else:
                    # Categorical: impute with mode or 'Unknown'
                    if self.df[col].mode().empty:
                        self.df[col].fillna('Unknown', inplace=True)
                    else:
                        self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    logger.info(f"  {col}: imputed {missing_pct_col:.2f}% missing with mode/Unknown")
        
        return self.df
    
    def engineer_features(self):
        """Create new features that may relate to claims."""
        logger.info("Engineering features...")
        
        # Vehicle Age (current year 2015 based on max date in data)
        current_year = 2015
        self.df['VehicleAge'] = current_year - self.df['RegistrationYear']
        self.df['VehicleAge'] = self.df['VehicleAge'].clip(lower=0)  # Handle negative ages
        
        # Policy Duration (parse TransactionMonth to estimate months since start)
        # Extract month/year from TransactionMonth if it's a date string
        try:
            self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth'], errors='coerce')
            earliest_month = self.df['TransactionMonth'].min()
            self.df['MonthsSinceStart'] = (self.df['TransactionMonth'] - earliest_month).dt.days / 30
            self.df['MonthsSinceStart'] = self.df['MonthsSinceStart'].clip(lower=0)
        except Exception as e:
            logger.warning(f"Could not parse TransactionMonth: {e}")
            self.df['MonthsSinceStart'] = 0
        
        # Premium-to-Exposure Ratio
        self.df['PremiumToSumInsured'] = self.df['TotalPremium'] / (self.df['SumInsured'] + 1)
        self.df['PremiumToSumInsured'] = self.df['PremiumToSumInsured'].fillna(0)
        
        # Is Vehicle Modified (has positive CustomValueEstimate)
        self.df['IsModifiedVehicle'] = (self.df['CustomValueEstimate'] > 0).astype(int)
        
        # Has Multiple Vehicles in Fleet
        self.df['HasFleet'] = (self.df['NumberOfVehiclesInFleet'] > 1).astype(int)
        
        # Claim Binary Indicator (for probability of claim model)
        self.df['HasClaim'] = (self.df['TotalClaims'] > 0).astype(int)
        
        logger.info(f"Created {5} new features")
        return self.df
    
    def select_features_for_modeling(self):
        """Select and prepare features for modeling."""
        logger.info("Selecting features for modeling...")
        
        # Numerical features to keep
        numerical_features = [
            'RegistrationYear', 'Cylinders', 'cubiccapacity', 'kilowatts', 'NumberOfDoors',
            'CustomValueEstimate', 'SumInsured', 'CalculatedPremiumPerTerm', 
            'VehicleAge', 'MonthsSinceStart', 'PremiumToSumInsured',
            'IsModifiedVehicle', 'HasFleet'
        ]
        
        # Categorical features to encode (select high-cardinality or informative ones)
        categorical_features = [
            'Gender', 'Province', 'VehicleType', 'make', 'CoverType', 'TermFrequency'
        ]
        
        # Ensure columns exist
        available_numerical = [col for col in numerical_features if col in self.df.columns]
        available_categorical = [col for col in categorical_features if col in self.df.columns]
        
        self.feature_columns = available_numerical + available_categorical
        
        logger.info(f"Selected {len(available_numerical)} numerical + {len(available_categorical)} categorical features")
        logger.info(f"Numerical: {available_numerical}")
        logger.info(f"Categorical: {available_categorical}")
        
        return self.feature_columns
    
    def encode_categorical_features(self, df, fit=False):
        """Encode categorical features using label encoding for tree models."""
        df_encoded = df.copy()
        
        categorical_features = [col for col in self.feature_columns if df[col].dtype == 'object']
        
        for col in categorical_features:
            if fit:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Encoded categorical feature: {col} ({len(le.classes_)} classes)")
            else:
                le = self.label_encoders[col]
                # Handle unseen categories
                df_encoded[col] = df[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
        
        return df_encoded
    
    def prepare_data_for_modeling(self, test_size=0.2, random_state=42):
        """Prepare final dataset for modeling."""
        logger.info("Preparing data for modeling...")
        
        # Select features and target
        X = self.df[self.feature_columns].copy()
        y = self.df[self.target_column].copy()
        
        # Encode categorical variables
        X = self.encode_categorical_features(X, fit=True)
        
        # Remove rows with missing target
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Dataset shape for modeling: {X.shape}")
        logger.info(f"Target (TotalClaims) statistics:\n{y.describe()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Train/Test split: {len(X_train)} / {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_severity_data(self, test_size=0.2, random_state=42):
        """Prepare data for claim severity prediction (claims only)."""
        logger.info("Preparing data for claim severity modeling (claims > 0)...")
        
        # Filter to policies with claims only
        has_claims_mask = self.df[self.target_column] > 0
        df_claims = self.df[has_claims_mask].copy()
        
        logger.info(f"Policies with claims: {len(df_claims)} / {len(self.df)} ({100*len(df_claims)/len(self.df):.2f}%)")
        
        # Select features and target
        X = df_claims[self.feature_columns].copy()
        y = df_claims[self.target_column].copy()
        
        # Encode categorical variables
        X = self.encode_categorical_features(X, fit=False)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Severity model - Train/Test split: {len(X_train)} / {len(X_test)}")
        logger.info(f"Target (TotalClaims|HasClaim) statistics:\n{y.describe()}")
        
        return X_train, X_test, y_train, y_test


class InsuranceModeler:
    """Train and evaluate predictive models for insurance claims."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_linear_regression(self, X_train, y_train, X_test, y_test, model_name="Linear Regression"):
        """Train and evaluate Linear Regression."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name}...")
        logger.info(f"{'='*60}")
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluation
        results = self._evaluate_model(
            y_train, y_pred_train, y_test, y_pred_test, model_name
        )
        
        self.models[model_name] = model
        self.results[model_name] = results
        
        return model, results
    
    def train_random_forest(self, X_train, y_train, X_test, y_test, 
                          n_estimators=100, max_depth=20, model_name="Random Forest"):
        """Train and evaluate Random Forest."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name}...")
        logger.info(f"{'='*60}")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluation
        results = self._evaluate_model(
            y_train, y_pred_train, y_test, y_pred_test, model_name
        )
        
        self.models[model_name] = model
        self.results[model_name] = results
        
        return model, results
    
    def train_xgboost(self, X_train, y_train, X_test, y_test,
                     max_depth=6, learning_rate=0.1, n_estimators=100, model_name="XGBoost"):
        """Train and evaluate XGBoost."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name}...")
        logger.info(f"{'='*60}")
        
        model = xgb.XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
            verbosity=1
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluation
        results = self._evaluate_model(
            y_train, y_pred_train, y_test, y_pred_test, model_name
        )
        
        self.models[model_name] = model
        self.results[model_name] = results
        
        return model, results
    
    def _evaluate_model(self, y_train, y_pred_train, y_test, y_pred_test, model_name):
        """Evaluate model on train and test sets."""
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        # Log results
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
        logger.info(f"  Train MAE:  {train_mae:.4f} | Test MAE:  {test_mae:.4f}")
        logger.info(f"  Train R²:   {train_r2:.4f} | Test R²:   {test_r2:.4f}")
        
        return results
    
    def compare_models(self):
        """Compare all trained models."""
        logger.info(f"\n{'='*60}")
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info(f"{'='*60}")
        
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.round(4)
        
        logger.info("\n" + str(comparison_df))
        
        # Find best model by test R²
        best_model = comparison_df['test_r2'].idxmax()
        logger.info(f"\nBest model (by Test R²): {best_model} ({comparison_df.loc[best_model, 'test_r2']:.4f})")
        
        return comparison_df


def main():
    """Main execution function."""
    
    logger.info("="*60)
    logger.info("TASK 4: STATISTICAL MODELING & RISK-BASED PRICING")
    logger.info("="*60)
    
    # 1. DATA PREPARATION
    logger.info("\n[PHASE 1] DATA PREPARATION")
    logger.info("-" * 60)
    
    preprocessor = InsuranceDataPreprocessor('data/insurance_data.csv')
    preprocessor.load_data()
    preprocessor.remove_duplicate_rows()
    preprocessor.handle_missing_values()
    preprocessor.engineer_features()
    preprocessor.select_features_for_modeling()
    
    # 2. SEVERITY PREDICTION (Claims > 0)
    logger.info("\n[PHASE 2] CLAIM SEVERITY PREDICTION MODEL")
    logger.info("-" * 60)
    
    X_train_severity, X_test_severity, y_train_severity, y_test_severity = (
        preprocessor.prepare_severity_data(test_size=0.2, random_state=42)
    )
    
    modeler_severity = InsuranceModeler()
    
    # Train models
    model_lr_severity, results_lr = modeler_severity.train_linear_regression(
        X_train_severity, y_train_severity, X_test_severity, y_test_severity,
        "Linear Regression (Severity)"
    )
    
    model_rf_severity, results_rf = modeler_severity.train_random_forest(
        X_train_severity, y_train_severity, X_test_severity, y_test_severity,
        n_estimators=100, max_depth=20,
        model_name="Random Forest (Severity)"
    )
    
    model_xgb_severity, results_xgb = modeler_severity.train_xgboost(
        X_train_severity, y_train_severity, X_test_severity, y_test_severity,
        max_depth=6, learning_rate=0.1, n_estimators=100,
        model_name="XGBoost (Severity)"
    )
    
    # Compare
    comparison_severity = modeler_severity.compare_models()
    comparison_severity.to_csv('results/severity_model_comparison.csv')
    logger.info("\nSeverity model comparison saved to: results/severity_model_comparison.csv")
    
    # 3. PREMIUM PREDICTION (All policies)
    logger.info("\n[PHASE 3] PREMIUM PREDICTION MODEL")
    logger.info("-" * 60)
    
    X_train_premium, X_test_premium, y_train_premium, y_test_premium = (
        preprocessor.prepare_data_for_modeling(test_size=0.2, random_state=42)
    )
    
    modeler_premium = InsuranceModeler()
    
    # Train models (using CalculatedPremiumPerTerm as target for this demo)
    # In production, would use P(claim) * Severity + Expenses + Margin
    y_train_premium_actual = preprocessor.df.loc[y_train_premium.index, 'CalculatedPremiumPerTerm']
    y_test_premium_actual = preprocessor.df.loc[y_test_premium.index, 'CalculatedPremiumPerTerm']
    
    model_lr_premium, results_lr_prem = modeler_premium.train_linear_regression(
        X_train_premium, y_train_premium_actual, X_test_premium, y_test_premium_actual,
        "Linear Regression (Premium)"
    )
    
    model_rf_premium, results_rf_prem = modeler_premium.train_random_forest(
        X_train_premium, y_train_premium_actual, X_test_premium, y_test_premium_actual,
        n_estimators=100, max_depth=20,
        model_name="Random Forest (Premium)"
    )
    
    model_xgb_premium, results_xgb_prem = modeler_premium.train_xgboost(
        X_train_premium, y_train_premium_actual, X_test_premium, y_test_premium_actual,
        max_depth=6, learning_rate=0.1, n_estimators=100,
        model_name="XGBoost (Premium)"
    )
    
    # Compare
    comparison_premium = modeler_premium.compare_models()
    comparison_premium.to_csv('results/premium_model_comparison.csv')
    logger.info("\nPremium model comparison saved to: results/premium_model_comparison.csv")
    
    # 4. FEATURE IMPORTANCE (to be done in separate script with SHAP)
    logger.info("\n[PHASE 4] FEATURE IMPORTANCE ANALYSIS")
    logger.info("-" * 60)
    logger.info("Feature importance analysis will be performed in task4_feature_importance.py")
    
    logger.info("\n" + "="*60)
    logger.info("TASK 4 MODELING PIPELINE COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    main()
