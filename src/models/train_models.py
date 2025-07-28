import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import joblib
import logging
from typing import Tuple, Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, data_path: str):
        """Initialize the ModelTrainer with data path."""
        self.data_path = data_path
        self.models = {}
        self.vectorizers = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare the data."""
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def prepare_features(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.Series]:
        """Prepare features for model training."""
        # Use the cleaned job description
        text_vectorizer = TfidfVectorizer(max_features=1000)
        text_features = text_vectorizer.fit_transform(df['job_description_text_clean'].fillna(''))
        self.vectorizers['text'] = text_vectorizer
        
        # Categorical features
        location_dummies = pd.get_dummies(df['Location'], prefix='location')
        experience_dummies = pd.get_dummies(df['Experience Level'], prefix='exp')
        
        # Combine all features
        X = pd.concat([
            pd.DataFrame(text_features.toarray()),
            location_dummies.reset_index(drop=True),
            experience_dummies.reset_index(drop=True)
        ], axis=1)
        X.columns = X.columns.astype(str)  # Ensure all column names are strings
        
        # Target variable (average of min and max salary)
        y = (df['Salary Min'] + df['Salary Max']) / 2
        
        return X, y

    def train_salary_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train salary prediction models."""
        logger.info("Training salary prediction models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Define models to train
        models = {
            'random_forest': RandomForestRegressor(random_state=42),
            'xgboost': XGBRegressor(random_state=42),
            'lasso': Lasso(random_state=42)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"{name} - MSE: {mse:.2f}, R2: {r2:.2f}")
            
            # Save model
            self.models[name] = model
            
        logger.info("Salary prediction models training completed!")

    def train_experience_classifier(self, df: pd.DataFrame) -> None:
        """Train experience level classifier."""
        logger.info("Training experience level classifier...")
        
        # Prepare features
        text_vectorizer = TfidfVectorizer(max_features=1000)
        X = text_vectorizer.fit_transform(df['job_description_text_clean'].fillna(''))
        y = df['Experience Level']
        
        self.vectorizers['experience'] = text_vectorizer
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train classifier
        classifier = RandomForestClassifier(random_state=42)
        classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = classifier.predict(X_test)
        report = classification_report(y_test, y_pred)
        logger.info(f"Classification Report:\n{report}")
        
        # Save model
        self.models['experience_classifier'] = classifier
        
        logger.info("Experience level classifier training completed!")

    def save_models(self, output_dir: str) -> None:
        """Save trained models and vectorizers."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            model_path = os.path.join(output_dir, f"{name}.joblib")
            joblib.dump(model, model_path)
            logger.info(f"Saved model: {model_path}")
        
        # Save vectorizers
        for name, vectorizer in self.vectorizers.items():
            vectorizer_path = os.path.join(output_dir, f"{name}_vectorizer.joblib")
            joblib.dump(vectorizer, vectorizer_path)
            logger.info(f"Saved vectorizer: {vectorizer_path}")

def main():
    """Main function to train models."""
    try:
        trainer = ModelTrainer('data/processed_data.csv')
        df = trainer.load_data()
        X, y = trainer.prepare_features(df)
        # Only train salary model if y has valid values
        if not y.isnull().all():
            trainer.train_salary_model(X, y)
        else:
            logger.warning("Salary data is missing. Skipping salary prediction model training.")
        trainer.train_experience_classifier(df)
        trainer.save_models('models/saved_models')
        logger.info("Model training pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Error in model training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 