import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Define the file paths (add paths as needed)
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

test_sizes= [0.1,0.2,0.3]
estimators= [10,25,50,100]

print("Random Forest")
for test_size in test_sizes:

    # Create the Training and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f'Test Size: {test_size}')

    for estimator in estimators:
        # Fit Random Forest Classification to the Training set
        classifier = RandomForestClassifier(n_estimators=estimator, criterion='entropy', random_state=42)
        classifier.fit(X_train, y_train)

        # Predict the Test set results
        y_pred = classifier.predict(X_test)

        # Reverse factorize (converting y_pred from 0s, 1s, etc., to original values)
        reversefactor = dict(zip(range(len(definitions)), definitions))
        y_test = np.vectorize(reversefactor.get)(y_test)
        y_pred = np.vectorize(reversefactor.get)(y_pred)

        # Evaluate model accuracy using numeric labels (0 and 1)
        accuracy = accuracy_score(y_test, y_pred)
        # Precision, Recall, and F1 Score using binary labels
        # Make sure y_test and y_pred are using binary values (0 or 1)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')

        print(f'Estimator: {estimator}')
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1 Score: {f1:.2f}')

        # Confusion Matrix
        print(pd.crosstab(y_test, y_pred, rownames=['Actual Shot Result'], colnames=['Predicted Shot Result']))

        # Confusion Matrix with binary labels (0 and 1)
        conf_matrix = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['made', 'miss'])
        disp.plot()

