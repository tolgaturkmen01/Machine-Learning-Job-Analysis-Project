import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import Dict, Any
import os

# Set page config
st.set_page_config(
    page_title="ML Job Analysis Dashboard",
    page_icon="⭐",
    layout="wide"
)

class JobAnalysisDashboard:
    def __init__(self):
        """Initialize the dashboard with necessary components."""
        self.models = {}
        self.vectorizers = {}
        self.data = None
        self.load_models()
        self.load_data()

    def load_models(self):
        """Load trained models and vectorizers."""
        models_dir = 'models/saved_models'
        try:
            # Load models
            for model_name in ['random_forest', 'xgboost', 'lasso']:
                model_path = os.path.join(models_dir, f"{model_name}.joblib")
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)

            # Load vectorizers
            for vectorizer_name in ['text', 'experience']:
                vectorizer_path = os.path.join(models_dir, f"{vectorizer_name}_vectorizer.joblib")
                if os.path.exists(vectorizer_path):
                    self.vectorizers[vectorizer_name] = joblib.load(vectorizer_path)

        except Exception as e:
            st.error(f"Error loading models: {str(e)}")

    def load_data(self):
        """Load processed data."""
        try:
            self.data = pd.read_csv('data/processed_data.csv')
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

    def predict_salary(self, job_description: str, location: str, experience_level: str) -> float:
        """Predict salary for given job details."""
        try:
            # Transform text features
            text_features = self.vectorizers['text'].transform([job_description])
            
            # Create location and experience features
            location_dummies = pd.get_dummies(pd.Series([location]), prefix='location')
            experience_dummies = pd.get_dummies(pd.Series([experience_level]), prefix='exp')
            
            # Combine features
            features = pd.concat([
                pd.DataFrame(text_features.toarray()),
                location_dummies,
                experience_dummies
            ], axis=1)
            
            # Make prediction using the best model (random forest)
            prediction = self.models['random_forest'].predict(features)[0]
            return prediction
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None

    def predict_experience_level(self, job_description: str) -> str:
        """Predict experience level for given job description."""
        try:
            # Transform text features
            features = self.vectorizers['experience'].transform([job_description])
            
            # Make prediction
            prediction = self.models['experience_classifier'].predict(features)[0]
            return prediction
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None

    def create_skill_wordcloud(self):
        """Create word cloud of skills."""
        if self.data is None:
            return None
            
        # Combine all skills
        all_skills = ' '.join([' '.join(skills) for skills in self.data['Skills']])
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white'
        ).generate(all_skills)
        
        # Display
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig

    def create_salary_by_location(self):
        """Create salary distribution by location."""
        if self.data is None:
            return None
            
        fig = px.box(
            self.data,
            x='Location',
            y='Salary Min',
            title='Salary Distribution by Location'
        )
        return fig

    def create_experience_distribution(self):
        """Create experience level distribution."""
        if self.data is None:
            return None
            
        fig = px.pie(
            self.data,
            names='Experience Level',
            title='Experience Level Distribution'
        )
        return fig

def main():
    """Main function to run the Streamlit app."""
    st.title("🤖 Machine Learning Job Analysis Dashboard")
    
    # Initialize dashboard
    dashboard = JobAnalysisDashboard()
    
    # Sidebar for predictions
    st.sidebar.header("Make Predictions")
    
    # Input fields
    job_description = st.sidebar.text_area(
        "Job Description",
        height=200,
        help="Enter the job description to analyze"
    )
    
    location = st.sidebar.selectbox(
        "Location",
        options=['Remote', 'San Francisco, CA', 'New York, NY', 'Seattle, WA']
    )
    
    experience_level = st.sidebar.selectbox(
        "Experience Level",
        options=['Junior', 'Mid', 'Senior']
    )
    
    # Make predictions
    if st.sidebar.button("Predict"):
        if job_description:
            # Predict salary
            salary = dashboard.predict_salary(
                job_description,
                location,
                experience_level
            )
            
            if salary:
                st.sidebar.success(f"Predicted Salary: ${salary:,.2f}")
            
            # Predict experience level
            predicted_level = dashboard.predict_experience_level(job_description)
            if predicted_level:
                st.sidebar.info(f"Predicted Experience Level: {predicted_level}")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Skills Analysis")
        wordcloud_fig = dashboard.create_skill_wordcloud()
        if wordcloud_fig:
            st.pyplot(wordcloud_fig)
    
    with col2:
        st.header("Experience Level Distribution")
        exp_fig = dashboard.create_experience_distribution()
        if exp_fig:
            st.plotly_chart(exp_fig)
    
    st.header("Salary Analysis")
    salary_fig = dashboard.create_salary_by_location()
    if salary_fig:
        st.plotly_chart(salary_fig)

if __name__ == "__main__":
    main() 