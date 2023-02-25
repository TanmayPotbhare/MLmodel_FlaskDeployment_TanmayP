import pandas as pd
import numpy as np
import pickle
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score, confusion_matrix,f1_score

sdf = pd.read_excel('T:\Work\Internship_Data_Glacier\Week 4\healthcare-dataset-stroke-data.xlsx')
sdf = sdf[['id','gender','age','avg_glucose_level','bmi','stroke']]
sdf.drop('id',axis=1)
sdf['gender'].value_counts()
sdf = sdf[(sdf["gender"] == "Female") | (sdf["gender"] =="Male")]
sdf["bmi"]=sdf['bmi'].fillna(sdf.median().iloc[0])
sdf['gender'] = pd.get_dummies(sdf["gender"],dtype=np.int64,prefix="Gender",drop_first=True)

# ML MODEL
X=sdf.drop(['stroke'],axis=1)
y=sdf['stroke']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=123)
LogRandom = LogisticRegression()
LogRandom.fit(X_train,y_train)

pickle.dump(LogRandom, open('StrokeP.pkl', 'wb'))
sdf = pickle.load(open('StrokeP.pkl','rb'))




