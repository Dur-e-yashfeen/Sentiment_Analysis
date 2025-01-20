# Sentiment Analysis Model

This project demonstrates how to build a sentiment analysis model using logistic regression. The model is trained to classify movie reviews as positive or negative based on their textual content.

## Prerequisites

- Python 3.x
- Jupyter Notebook or any Python IDE

## Libraries

The following Python libraries are required:

- pandas
- numpy
- re
- nltk
- scikit-learn
- warnings

You can install these libraries using pip:

```bash
pip install pandas numpy nltk scikit-learn
```

## Dataset

The dataset used in this project is a CSV file containing movie reviews. Make sure to replace the path of the dataset in the code with the actual path to your CSV file:

```python
data = pd.read_csv('./dataset/imdb_top_1000.csv')
```

## Code Overview

### Importing Necessary Libraries

```python
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.impute import SimpleImputer  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
```

### Downloading NLTK Stopwords
### Removing Warnings
### Loading the Dataset
### Handling Missing Values
### Preprocessing Function
## Adding Sentiment Column
### Feature Engineering
### Splitting the Dataset
### Training the Model
### Making Predictions
### Evaluating the Model
## Results


Accuracy: 0.97 🎯

Precision: 0.00 💯

Recall: 0.00 🔍

The model's performance is evaluated using accuracy, precision, and recall metrics.

- **Accuracy:** Measures the overall correctness of the model.
- **Precision:** Measures the proportion of positive identifications that were actually correct.
- **Recall:** Measures the proportion of actual positives that were identified correctly.

## Conclusion

This project demonstrates a basic sentiment analysis model using logistic regression. The model can be further improved by using more sophisticated text preprocessing techniques, different algorithms, and hyperparameter tuning.

## Acknowledgements

- The dataset used in this project is sourced from IMDb.
- The project utilizes libraries like NLTK for text preprocessing and scikit-learn for machine learning.

Feel free to contribute to this project by opening issues or submitting pull requests.
