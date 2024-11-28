import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

files = ["NBA_2004_SHOTS.csv", "NBA_2005_SHOTS.csv", "NBA_2006_SHOTS.csv",
         "NBA_2007_SHOTS.csv", "NBA_2008_SHOTS.csv", "NBA_2009_SHOTS.csv",
         "NBA_2010_SHOTS.csv", "NBA_2011_SHOTS.csv", "NBA_2012_SHOTS.csv",
         "NBA_2013_SHOTS.csv", "NBA_2014_SHOTS.csv", "NBA_2015_SHOTS.csv",
         "NBA_2016_SHOTS.csv", "NBA_2017_SHOTS.csv", "NBA_2018_SHOTS.csv",
         "NBA_2019_SHOTS.csv", "NBA_2020_SHOTS.csv", "NBA_2021_SHOTS.csv",
         "NBA_2022_SHOTS.csv", "NBA_2023_SHOTS.csv", "NBA_2024_SHOTS.csv"]

# Load and concatenate the files
dataframes = [pd.read_csv(file) for file in files]
dataset = pd.concat(dataframes, ignore_index=True)

dataset = dataset[["POSITION", "BASIC_ZONE", "ZONE_NAME", "ZONE_RANGE", "LOC_X", "LOC_Y", "SHOT_DISTANCE", "QUARTER", "MINS_LEFT", "SECS_LEFT", "SHOT_TYPE", "ACTION_TYPE", "SHOT_MADE"]]
dataset.columns = ['pos', 'bzone', 'zone', 'zoner', 'x', 'y', 'dist', 'quarter', 'mins', 'secs', 'shot', 'type', 'made']

# Split data into features and target
X = dataset.drop("made", axis=1)
y = dataset["made"]

test_sizes= [0.1,0.2,0.3]
validations= [5,10,20]

print("Gradient Boosting")
for test_size in test_sizes:
    print(f'Test Size: {test_size}')

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size= test_size, random_state=42
    )

    # Define categorical and numerical features
    categorical_features = X.select_dtypes(
       include=["object"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
       include=["float64", "int64"]
    ).columns.tolist()

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

    # Perform 5-fold cross-validation
    for validation in validations:
        print(f'Cross Validation: {validation}')
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=validation)

        # Fit the model on the training data
        pipeline.fit(X_train, y_train)

        # Predict on the test set
        y_pred = pipeline.predict(X_test)

        # Generate classification report
        report = classification_report(y_test, y_pred)

        print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")
        print("\nClassification Report:")
        print(report)