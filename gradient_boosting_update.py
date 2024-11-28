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

files=["NBA_2020_SHOTS.csv", "NBA_2021_SHOTS.csv",
         "NBA_2022_SHOTS.csv", "NBA_2023_SHOTS.csv", "NBA_2024_SHOTS.csv"]

dataframes = [pd.read_csv(file) for file in files]
dataset = pd.concat(dataframes, ignore_index=True)

dataset = dataset[["POSITION", "BASIC_ZONE", "ZONE_NAME", "ZONE_RANGE", "LOC_X", "LOC_Y", "SHOT_DISTANCE", "QUARTER", "MINS_LEFT", "SECS_LEFT", "SHOT_TYPE", "ACTION_TYPE", "SHOT_MADE"]]
dataset.columns = ['pos', 'bzone', 'zone', 'zoner', 'x', 'y', 'dist', 'quarter', 'mins', 'secs', 'shot', 'type', 'made']

X = dataset.drop("made", axis=1)
y = dataset["made"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), categorical_features),
        ("num", StandardScaler(), numerical_features),
    ]
)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=42)),
    ]
)

param_dist = {
    "classifier__n_estimators": randint(50, 200),
    "classifier__learning_rate": uniform(0.01, 0.3),
    "classifier__max_depth": randint(3, 10),
    "classifier__min_samples_split": randint(2, 10),
    "classifier__min_samples_leaf": randint(1, 5),
    "classifier__subsample": uniform(0.7, 0.3),  
}

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=50,            
    scoring="accuracy",   
    cv=3,                 
    random_state=42,      
    n_jobs=-1,           
    verbose=2             
)


random_search.fit(X_train, y_train)


best_pipeline = random_search.best_estimator_
print(f"Best Parameters: {random_search.best_params_}")

y_pred = best_pipeline.predict(X_test)


report = classification_report(y_test, y_pred)

print(f"\nClassification Report:")
print(report)

conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Miss", "Made"])
disp.plot()
plt.title("Confusion Matrix for Gradient Boosting")
plt.show()
plt.savefig("Confusion_MatrixGB.png")
