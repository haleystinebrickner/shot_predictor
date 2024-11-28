from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import randint, uniform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Load and concatenate the files
"""
files = ["NBA_2004_SHOTS.csv", "NBA_2005_SHOTS.csv", "NBA_2006_SHOTS.csv",
         "NBA_2007_SHOTS.csv", "NBA_2008_SHOTS.csv", "NBA_2009_SHOTS.csv",
         "NBA_2010_SHOTS.csv", "NBA_2011_SHOTS.csv", "NBA_2012_SHOTS.csv",
         "NBA_2013_SHOTS.csv", "NBA_2014_SHOTS.csv", "NBA_2015_SHOTS.csv",
         "NBA_2016_SHOTS.csv", "NBA_2017_SHOTS.csv", "NBA_2018_SHOTS.csv",
         "NBA_2019_SHOTS.csv", "NBA_2020_SHOTS.csv", "NBA_2021_SHOTS.csv",
         "NBA_2022_SHOTS.csv", "NBA_2023_SHOTS.csv", "NBA_2024_SHOTS.csv"]
"""
files=["NBA_2020_SHOTS.csv", "NBA_2021_SHOTS.csv",
         "NBA_2022_SHOTS.csv", "NBA_2023_SHOTS.csv", "NBA_2024_SHOTS.csv"]

dataframes = [pd.read_csv(file) for file in files]
dataset = pd.concat(dataframes, ignore_index=True)

dataset = dataset[["POSITION", "BASIC_ZONE", "ZONE_NAME", "ZONE_RANGE", "LOC_X", "LOC_Y", "SHOT_DISTANCE", "QUARTER", "MINS_LEFT", "SECS_LEFT", "SHOT_TYPE", "ACTION_TYPE", "SHOT_MADE"]]
dataset.columns = ['pos', 'bzone', 'zone', 'zoner', 'x', 'y', 'dist', 'quarter', 'mins', 'secs', 'shot', 'type', 'made']

# Split data into features and target
X = dataset.drop("made", axis=1)
y = dataset["made"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Define categorical and numerical features
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), categorical_features),
        ("num", StandardScaler(), numerical_features),
    ]
)

# Gradient Boosting pipeline
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=42)),
    ]
)

# Define hyperparameter distributions
param_dist = {
    "classifier__n_estimators": randint(50, 200),
    "classifier__learning_rate": uniform(0.01, 0.3),
    "classifier__max_depth": randint(3, 10),
    "classifier__min_samples_split": randint(2, 10),
    "classifier__min_samples_leaf": randint(1, 5),
    "classifier__subsample": uniform(0.7, 0.3),  # Subsampling fraction
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=50,            # Number of random combinations to try
    scoring="accuracy",   # Scoring metric
    cv=3,                 # Cross-validation folds
    random_state=42,      # Reproducibility
    n_jobs=-1,            # Use all CPU cores
    verbose=2             # Show progress
)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Best pipeline
best_pipeline = random_search.best_estimator_
print(f"Best Parameters: {random_search.best_params_}")

# Predict on the test set using the best pipeline
y_pred = best_pipeline.predict(X_test)

# Generate classification report
report = classification_report(y_test, y_pred)

print(f"\nClassification Report:")
print(report)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Miss", "Made"])
disp.plot()
plt.title("Confusion Matrix for Gradient Boosting")
plt.show()
plt.savefig("Confusion_MatrixGB10.png")

"""
Classification Report:
              precision    recall  f1-score   support

       False       0.61      0.85      0.71    109641
        True       0.70      0.39      0.50     96708

    accuracy                           0.63    206349
   macro avg       0.65      0.62      0.61    206349
weighted avg       0.65      0.63      0.61    206349


Classification Report:
              precision    recall  f1-score   support

       False       0.61      0.85      0.71     54950
        True       0.70      0.39      0.50     48225

    accuracy                           0.64    103175
   macro avg       0.65      0.62      0.61    103175
weighted avg       0.65      0.64      0.61    103175
"""