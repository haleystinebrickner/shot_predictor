from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform

files= ["NBA_2020_SHOTS.csv", "NBA_2021_SHOTS.csv",
         "NBA_2022_SHOTS.csv", "NBA_2023_SHOTS.csv", "NBA_2024_SHOTS.csv"]


dataframes = [pd.read_csv(file) for file in files]
dataset = pd.concat(dataframes, ignore_index=True)

dataset = dataset[["POSITION", "BASIC_ZONE", "ZONE_NAME", "ZONE_RANGE", "LOC_X", "LOC_Y", "SHOT_DISTANCE", "QUARTER", "MINS_LEFT", "SECS_LEFT", "SHOT_TYPE", "ACTION_TYPE", "SHOT_MADE"]]
dataset.columns = ['pos', 'bzone', 'zone', 'zoner', 'x', 'y', 'dist', 'quarter', 'mins', 'secs', 'shot', 'type', 'made']


factor = pd.factorize(dataset['made'])
dataset = dataset.copy()
dataset['made_factor'] = factor[0].astype(int)
definitions = factor[1]


dataset_encoded = pd.get_dummies(dataset, columns=['pos', 'bzone', 'zone', 'zoner', 'shot', 'type'])

X = dataset_encoded.drop(columns=['made', 'made_factor']).values
y = dataset_encoded['made_factor'].values

X = dataset_encoded.drop(columns=['made', 'made_factor']).values
y = dataset_encoded['made_factor'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_dist = {
    'n_estimators': randint(50, 200),        
    'max_depth': [None, 10, 20, 30],          
    'min_samples_split': randint(2, 11),     
    'min_samples_leaf': randint(1, 5),        
    'criterion': ['gini', 'entropy'],         
    'max_features': ['sqrt', 'log2', None],   
}


rf = RandomForestClassifier(random_state=42)


random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,            
    scoring='accuracy',   
    cv=3,                 
    random_state=42,      
    n_jobs=-1,           
    verbose=2            
)
random_search.fit(X_train, y_train)

best_rf = random_search.best_estimator_
print(f"Best Parameters: {random_search.best_params_}")

y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')


conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['miss', 'made'])
disp.plot()
plt.show()
plt.savefig('RFC_Confusion_Matrix10.png')
