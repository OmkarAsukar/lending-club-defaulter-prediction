# --------------
import pandas as pd
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(filepath_or_buffer =path,compression='zip',low_memory =False)

X = df.drop('loan_status',axis=1)
y = df['loan_status']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=4)

print(X_train.head())
# Code ends here


# --------------
# Code starts  here
col = df.isna().sum()

percent = ((col/df.isnull().count())*100).sort_values(ascending=False)

col_drop = list(percent[percent>25].index)

for col in df.columns:
    if df[col].nunique()==1:
        print(col)
        col_drop.append(col)
# Code ends here

print(col_drop)

X_train = X_train.drop(col_drop,axis=1)
X_test = X_test.drop(col_drop,axis=1)



# --------------
import numpy as np


# Code starts here
y_train = np.where((y_train == 'Fully Paid') |(y_train == 'Current'), 0, 1)

y_test = np.where((y_test == 'Fully Paid') |(y_test == 'Current'), 0, 1)


# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder


# categorical and numerical variables
cat = X_train.select_dtypes(include = 'O').columns.tolist()
num = X_train.select_dtypes(exclude = 'O').columns.tolist()

# Code starts here
for col in num:
    X_train[col].fillna(X_train[col].mean(),inplace=True)
    X_test[col].fillna(X_test[col].mean(),inplace=True)

for col in cat:
    X_train[col].fillna(X_train[col].mode()[0],inplace=True)
    X_test[col].fillna(X_test[col].mode()[0],inplace=True)

le = LabelEncoder()
for col in cat:
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.fit_transform(X_test[col])
# Code ends here


# --------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,confusion_matrix,classification_report
from sklearn import metrics
import matplotlib.pyplot as plt

# Code starts here
rf = RandomForestClassifier(random_state=42,max_depth=2,min_samples_leaf=5000)

rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

f1 = f1_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
roc_auc = roc_auc_score(y_test,y_pred)

print(confusion_matrix)
print(classification_report)

y_pred_proba = rf.predict_proba(X_test)[:,1]

fpr,tpr,thresholds = metrics.roc_curve(y_test,y_pred_proba)

auc = roc_auc_score(y_test,y_pred_proba)

plt.plot(fpr,tpr,label='Random Forest Model,auc='+str(auc))
plt.show()

# Code ends here


# --------------
from xgboost import XGBClassifier

# Code starts here
xgb = XGBClassifier(learning_rate=0.0001)
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)


f1 = f1_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
roc_auc = roc_auc_score(y_test,y_pred)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

y_pred_proba = xgb.predict_proba(X_test)[:,1]

fpr,tpr,_ = metrics.roc_curve(y_test,y_pred_proba)

auc = roc_auc_score(y_test,y_pred_proba)

plt.plot(fpr,tpr,label='XGBoost Model,auc='+str(auc))
plt.show()

# Code ends here


