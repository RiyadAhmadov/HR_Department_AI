#Let's create model
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , f1_score
from sklearn.ensemble import GradientBoostingClassifier
import warnings as wg
wg.filterwarnings('ignore')

path = r'C:\Users\HP\OneDrive\İş masası\df1.csv'

df = pd.read_csv(path)
del df['Unnamed: 0']
# Split data into features and target
X = df.drop(columns = ['resume_quality'], axis=1)
y = df['resume_quality']

# Split data into training and testing sets
X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_val, y_val, test_size=0.1, random_state=42)

gb_model = GradientBoostingClassifier(
    n_estimators=280, 
    learning_rate=0.2,  
    max_depth=5,  
    min_samples_split=6,
    min_samples_leaf=4,
    subsample=1.0, 
    max_features ='log2',
    random_state=42
)
gb_model.fit(X_train, y_train)

y_pred = gb_model.predict(X_test)
y_predtr = gb_model.predict(X_train)

fx = f1_score(y_test, y_pred)
print(f'F1 Score: {fx:.4f}')

new = np.array([[1,0,1,1,1,1,1,1,1,1]])
result = gb_model.predict(new)
print(result.tolist()[0])
print(X.columns)
#Load the saved model from the Pickle file
# with open('gb_model.pkl', 'wb') as model_file:
#     pickle.dump(gb_model, model_file)