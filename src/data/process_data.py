import pandas as pd
import numpy as np
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from typing import List, Dict, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        """Initialize the DataProcessor with necessary NLP tools."""
        try:
            self.nlp = spacy.load('en_core_web_sm')
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            logger.error(f"Error initializing DataProcessor: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean and preprocess text data."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text

    def extract_skills(self, text: str) -> List[str]:
        doc = self.nlp(text)
        ml_skills = {
            'python', 'tensorflow', 'pytorch', 'scikit-learn', 'keras',
            'numpy', 'pandas', 'sql', 'spark', 'hadoop', 'aws', 'azure',
            'gcp', 'docker', 'kubernetes', 'mlops', 'nlp', 'computer vision',
            'deep learning', 'machine learning', 'ai', 'data science'
        }
        found_skills = []
        for token in doc:
            if token.text.lower() in ml_skills:
                found_skills.append(token.text.lower())
        return list(set(found_skills))

    def standardize_location(self, locality: str, region: str) -> str:
        if not isinstance(locality, str):
            locality = "Unknown"
        if not isinstance(region, str):
            region = "Unknown"
        if 'remote' in str(locality).lower() or 'remote' in str(region).lower():
            return 'Remote'
        return f"{locality}, {region}"

    def extract_experience_level(self, text: str) -> str:
        text = text.lower()
        junior_keywords = ['junior', 'entry', 'entry-level', '0-2', '0-3']
        senior_keywords = ['senior', 'lead', 'principal', '5+', '7+', '10+']
        if any(keyword in text for keyword in senior_keywords):
            return 'Senior'
        elif any(keyword in text for keyword in junior_keywords):
            return 'Junior'
        else:
            return 'Mid'

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting data processing...")
        processed_df = df.copy()
        # Clean and process job description
        processed_df['job_description_text_clean'] = processed_df['job_description_text'].apply(self.clean_text)
        # Extract skills
        processed_df['Skills'] = processed_df['job_description_text_clean'].apply(self.extract_skills)
        # Standardize locations
        processed_df['Location'] = processed_df.apply(lambda row: self.standardize_location(row['company_address_locality'], row['company_address_region']), axis=1)
        # Extract experience level (if not already present)
        if 'seniority_level' not in processed_df or processed_df['seniority_level'].isnull().all():
            processed_df['Experience Level'] = processed_df['job_description_text_clean'].apply(self.extract_experience_level)
        else:
            processed_df['Experience Level'] = processed_df['seniority_level'].fillna('').replace('', 'Mid')
        # There is no salary column, so leave Salary Min/Max as NaN for now
        processed_df['Salary Min'] = np.nan
        processed_df['Salary Max'] = np.nan
        logger.info("Data processing completed successfully!")
        return processed_df

def main():
    """Main function to process data."""
    try:
        processor = DataProcessor()
        df = pd.read_csv('data/1000_ml_jobs_us.csv')
        processed_df = processor.process_dataframe(df)
        processed_df.to_csv('data/processed_data.csv', index=False)
        logger.info("Data processing pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Error in main processing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 