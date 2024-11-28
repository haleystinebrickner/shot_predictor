import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputClassifier

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

# Factorize 'type' to create 'action_factor'
action_factor = pd.factorize(dataset['type'])
dataset['action_factor'] = action_factor[0].astype(int)  # Add 'action_factor' to dataset
action_definitions = action_factor[1]

# One-Hot Encode categorical variables
dataset_encoded = pd.get_dummies(dataset, columns=['pos', 'bzone', 'zone', 'zoner', 'shot', 'type'])


#dataset_encoded['action_factor'] = dataset['action_factor']

# Encode 'action_type' as integers for the second target
action_factor = pd.factorize(dataset['type'])
dataset['action_factor'] = action_factor[0].astype(int)
action_definitions = action_factor[1]

# Split the data into independent and dependent variables
X = dataset_encoded.drop(columns=['made', 'made_factor','action_factor' ]).values

y = dataset_encoded[['made_factor', 'action_factor']].values


test_sizes= [0.1,0.2,0.3]
estimators= [10,25,50,100]
#test_sizes=[0.1]
#estimators=[10]

print("Random Forest with Multi-Output")
for test_size in test_sizes:

    # Create the Training and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f'Test Size: {test_size}')

    for estimator in estimators:
        # Use MultiOutputClassifier with RandomForestClassifier
        base_classifier = RandomForestClassifier(n_estimators=estimator, criterion='entropy', random_state=42)
        classifier = MultiOutputClassifier(base_classifier)
        classifier.fit(X_train, y_train)

        # Predict the Test set results
        y_pred = classifier.predict(X_test)

        # Reverse factorize (convert integer predictions to original labels)
        made_reversefactor = dict(zip(range(len(definitions)), definitions))
        action_reversefactor = dict(zip(range(len(action_definitions)), action_definitions))

        y_test_made = np.vectorize(made_reversefactor.get)(y_test[:, 0])
        y_pred_made = np.vectorize(made_reversefactor.get)(y_pred[:, 0])

        y_test_action = np.vectorize(action_reversefactor.get)(y_test[:, 1])
        y_pred_action = np.vectorize(action_reversefactor.get)(y_pred[:, 1])


        # Evaluate 'made_factor'
        accuracy_made = accuracy_score(y_test_made, y_pred_made)
        print(f'Estimator: {estimator} - Accuracy (made_factor): {accuracy_made:.2f}')

        # Evaluate 'action_type_factor'
        accuracy_action = accuracy_score(y_test_action, y_pred_action)
        print(f'Estimator: {estimator} - Accuracy (action_type_factor): {accuracy_action:.2f}')

        # Confusion Matrix for 'made_factor'
        conf_matrix_made = confusion_matrix(y_test_made, y_pred_made)
        print("Confusion Matrix for 'made_factor':\n", conf_matrix_made)

        # Set the labels (you can customize the labels based on the problem, but typically 'miss' and 'made')
        disp_made = ConfusionMatrixDisplay(
            confusion_matrix=conf_matrix_made,
            display_labels=['miss', 'made']
        )

        # Plot the confusion matrix for 'made_factor'
        disp_made.plot()


        # Explicitly show the plot to ensure it's rendered
        #plt.show()
        plt.savefig(f'CMMadeFactor-{test_size}-{estimator}.png')







