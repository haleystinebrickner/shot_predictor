from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform

# Define the file paths (add paths as needed)
"""
files = ["NBA_2004_SHOTS.csv", "NBA_2005_SHOTS.csv", "NBA_2006_SHOTS.csv",
         "NBA_2007_SHOTS.csv", "NBA_2008_SHOTS.csv", "NBA_2009_SHOTS.csv",
         "NBA_2010_SHOTS.csv", "NBA_2011_SHOTS.csv", "NBA_2012_SHOTS.csv",
         "NBA_2013_SHOTS.csv", "NBA_2014_SHOTS.csv", "NBA_2015_SHOTS.csv",
         "NBA_2016_SHOTS.csv", "NBA_2017_SHOTS.csv", "NBA_2018_SHOTS.csv",
         "NBA_2019_SHOTS.csv", "NBA_2020_SHOTS.csv", "NBA_2021_SHOTS.csv",
         "NBA_2022_SHOTS.csv", "NBA_2023_SHOTS.csv", "NBA_2024_SHOTS.csv"]
"""
files= ["NBA_2020_SHOTS.csv", "NBA_2021_SHOTS.csv",
         "NBA_2022_SHOTS.csv", "NBA_2023_SHOTS.csv", "NBA_2024_SHOTS.csv"]


# Load and concatenate the files
dataframes = [pd.read_csv(file) for file in files]
dataset = pd.concat(dataframes, ignore_index=True)

# Select relevant columns and rename
dataset = dataset[["POSITION", "BASIC_ZONE", "ZONE_NAME", "ZONE_RANGE", "LOC_X", "LOC_Y", "SHOT_DISTANCE", "QUARTER", "MINS_LEFT", "SECS_LEFT", "SHOT_TYPE", "ACTION_TYPE", "SHOT_MADE"]]
dataset.columns = ['pos', 'bzone', 'zone', 'zoner', 'x', 'y', 'dist', 'quarter', 'mins', 'secs', 'shot', 'type', 'made']

# Create dependent variable class
factor = pd.factorize(dataset['made'])
dataset = dataset.copy()
dataset['made_factor'] = factor[0].astype(int)
definitions = factor[1]

# One-Hot Encode categorical variables
dataset_encoded = pd.get_dummies(dataset, columns=['pos', 'bzone', 'zone', 'zoner', 'shot', 'type'])


# Split the data into independent and dependent variables
X = dataset_encoded.drop(columns=['made', 'made_factor']).values
y = dataset_encoded['made_factor'].values

# Split the data into independent and dependent variables
X = dataset_encoded.drop(columns=['made', 'made_factor']).values
y = dataset_encoded['made_factor'].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the parameter distribution for random search
param_dist = {
    'n_estimators': randint(50, 200),          # Random integers between 50 and 200
    'max_depth': [None, 10, 20, 30],          # Fixed values
    'min_samples_split': randint(2, 11),      # Random integers between 2 and 10
    'min_samples_leaf': randint(1, 5),        # Random integers between 1 and 4
    'criterion': ['gini', 'entropy'],         # Fixed values
    'max_features': ['sqrt', 'log2', None],   # Fixed values
}

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Perform Randomized Search
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,            # Number of parameter settings to sample
    scoring='accuracy',   # Evaluation metric
    cv=3,                 # 3-fold cross-validation
    random_state=42,      # Ensure reproducibility
    n_jobs=-1,            # Use all available cores
    verbose=2             # Show progress during search
)
random_search.fit(X_train, y_train)

# Get the best model
best_rf = random_search.best_estimator_
print(f"Best Parameters: {random_search.best_params_}")

# Evaluate the model on the test set
y_pred = best_rf.predict(X_test)

# Metrics evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['miss', 'made'])
disp.plot()
plt.show()
plt.savefig('RFC_Confusion_Matrix10.png')

"""Accuracy: 0.65
Precision: 0.64
Recall: 0.88
F1 Score: 0.74

Accuracy: 0.63
Precision: 0.69
Recall: 0.40
F1 Score: 0.50


Accuracy: 0.63
Precision: 0.69
Recall: 0.39
F1 Score: 0.50
"""