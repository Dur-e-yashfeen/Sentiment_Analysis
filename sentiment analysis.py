# lets create sentiment analysis model
#importng necessary libraries 
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.impute import SimpleImputer  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
# Download stopwords from nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

#remove warning
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
# Replace './dataset/imdb_top_1000.csv' with the actual path to your CSV file
data = pd.read_csv('./dataset/imdb_top_1000.csv')

# Check the first 5 rows of the dataset
print(data.head())

data.isnull().sum().sort_values(ascending=False)

data['Certificate'] = data['Certificate'].ffill()
data['Gross'] = data['Gross'].ffill()
data['Meta_score'] = data['Meta_score'].ffill()

# Preprocessing function
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text  

data['cleaned_reviews'] = data['Overview'].apply(preprocess_text)

 #Check if 'Overview' exists in the DataFrame
if 'Overview' in data.columns:
    # Create the Sentiment column based on the Overview
    data['Sentiment'] = data['Overview'].apply(lambda x: 1 if 'good' in x.lower() or 'great' in x.lower() or 'excellent' in x.lower() else 0)

    # Feature engineering
    X = data['cleaned_reviews']
    y = data['Sentiment']

    # Convert text data into numerical format using CountVectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Print evaluation metrics
print(f'Accuracy: {accuracy:.2f} 🎯')
print(f'Precision: {precision:.2f} 💯')
print(f'Recall: {recall:.2f} 🔍')

