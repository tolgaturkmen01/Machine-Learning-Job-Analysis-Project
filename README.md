# TolgaTurkmen 2025
# Machine Learning Job Analysis and Salary Prediction System
![](https://cdn-icons-png.flaticon.com/512/3716/3716795.png)


This project analyzes machine learning job postings in the US, predicts salaries, and classifies job levels using NLP and machine learning techniques.

## Features

- Job posting analysis and data preprocessing
- Exploratory Data Analysis (EDA)
- Experience level classification (Junior/Mid/Senior)
- Salary prediction for job postings
- Interactive dashboard with visualizations
- Web application for predictions

## Project Structure

```
├── data/                   # Data directory
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Source code
│   ├── data/             # Data processing modules
│   ├── models/           # ML models
│   ├── nlp/              # NLP processing
│   ├── visualization/    # Visualization modules
│   └── web/              # Web application
├── tests/                # Unit tests
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Data Processing:
bash
python src/data/process_data.py
```

2. Train Models:
```bash
python src/models/train_models.py
```

3. Run Web Application:
```bash
streamlit run src/web/app.py
```

## Project Components

### 1. Data Processing
- Text cleaning and preprocessing
- Feature extraction
- Location standardization
- Missing value handling

### 2. EDA
- Skill demand analysis
- Location-based job distribution
- Company analysis
- Remote work trends

### 3. Experience Level Classification
- Text-based classification
- Multiple model comparison
- Performance evaluation

### 4. Salary Prediction
- Feature engineering
- Multiple regression models
- Model evaluation and selection

### 5. Visualization
- Interactive dashboards
- Skill word clouds
- Salary heatmaps
- Trend analysis


# TolgaTurkmen 2025

