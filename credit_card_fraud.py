import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,recall_score,confusion_matrix,roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

df=pd.read_csv("creditcard.csv")
print(df.head())

print(df.isnull().sum())
df.drop_duplicates(inplace=True)
df.fillna(0,inplace=True)
print(df.describe().T)
print(df["Class"].value_counts())

x=df.drop("Class",axis=1)
y=df["Class"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43,stratify=y)

log_model=LogisticRegression(max_iter=10000,class_weight="balanced")
log_model.fit(x_train,y_train)
proba=log_model.predict_proba(x_test)[:,1]
y_pred=(proba > 0.4).astype(int)
confusion=confusion_matrix(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
roc=roc_auc_score(y_test,proba)

print("confusion matrix:",confusion)
print("precision score:",precision)
print("recall score:",recall)
print("roc auc is :",roc)
print("probability is:",proba)

random_model=RandomForestClassifier(random_state=43,class_weight="balanced")
random_model.fit(x_train,y_train)
y_prediction=random_model.predict(x_test)
print("random model scores are :")
print("confusion matrix:",confusion_matrix(y_test,y_prediction))
print("precision score:",precision_score(y_test,y_prediction))
print("recall score:",recall_score(y_test,y_prediction))
print("roc auc is :",roc_auc_score(y_test,proba))

param_dist={
    "n_estimators":[200,300,400],
    "max_depth":[10,20,30,None],
    "min_samples_split":[2,5,10],
    "min_samples_leaf":[1,2,4]
}

rf=RandomForestClassifier(random_state=43,class_weight="balanced")
random_search=RandomizedSearchCV(estimator=rf,n_iter=20,n_jobs=-1,cv=5,scoring="recall",param_distributions=param_dist,random_state=43)
random_search.fit(x_train,y_train)
best_model=random_search.best_estimator_

best_model.fit(x_train,y_train)
probability=best_model.predict_proba(x_test)[:,1]
prediction=(probability > 0.4).astype(int)
print("after tuning the new scores of the machine are:")

print("confusion matrix:",confusion_matrix(y_test,prediction))
print("precision score:",precision_score(y_test,prediction))
print("recall score:",recall_score(y_test,prediction))
print("roc auc is :",roc_auc_score(y_test,probability))