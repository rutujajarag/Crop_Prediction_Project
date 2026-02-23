#1 Import Necessary Libraries
import pickle

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#2 Create the Dataset
df=pd.read_csv('crop_recommendation.csv')
#3 Extract X and y
X=df[['N','P','K','temperature','humidity','ph','rainfall']]
y=df['label']

#4 Create Moddel
model=DecisionTreeClassifier()

#5 Train model
model.fit(X,y)

# Save model
with open('crop_model.pkl', 'wb') as f:
 pickle.dump(model, f)