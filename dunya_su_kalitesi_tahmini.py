# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 14:14:17 2025

@author: ozgeb
"""

import numpy as np
import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import precision_score, confusion_matrix

from sklearn import tree

df=pd.read_csv("water_potability.csv")

describe=df.describe()

df.info()
pio.renderers.default='browser'

# dependent variable analysis
d=pd.DataFrame(df["Potability"].value_counts())
fig=px.pie(d,values="Potability", names=["Not Potable", "Potable"], hole = 0.35, opacity=0.8,
           labels={"label":"Potability", "Potability":"Number of Sample"})
fig.update_layout(title=dict(text="Pie Chart of Potability Feature"))
fig.update_traces(textposition="outside",textinfo="percent+label")
fig.show()
fig.write_html("potability_pie_chart.html")

# korelasyon analizi
sns.clustermap(df.corr(),cmap="vlag", dendrogram_ratio=(0.1,0.2), annot=True, linewidths=0.8, figsize=(10,10))
plt.show()

#distribution of features

non_potable=df.query("Potability==0")
potable=df.query("Potability==1")

plt.figure(figsize=(15,15))
for ax, col in enumerate(df.columns[:9]):
    plt.subplot(3,3, ax + 1)
    plt.title(col)
    sns.kdeplot(x = non_potable[col], label="Non Potable")
    sns.kdeplot(x = potable[col], label="Potable")
    plt.legend()

    
plt.tight_layout()

#missing value
msno.bar(df)

df.isnull().sum().sort_values(ascending=False).plot(kind='bar')
plt.title("Eksik Veri Sayısı")
plt.show()



# %% Preprocessing: missing valur problem, train test split, normalization
# filling the missing values with mean value

#df["ph"].fillna(value=df["ph"].mean(), inplace=True)


df["Sulfate"].fillna(value=df["Sulfate"].mean(), inplace=True)
df["Trihalomethanes"].fillna(value=df["Trihalomethanes"].mean(), inplace=True)

#eksik degerlerisu sekilde doldur potablity 0 olanlarin eksik ph degerlerini potability 0 olanlarin ortalamsiyla;
#1 olanlari 1 olanlarin ortalamasiyla doldur.



potablePhMeanValue=df.loc[(df["Potability"]==1)&(df["ph"].notnull()), "ph"].mean()

nonPotablePhMeanValue=df.loc[(df["Potability"]==0)&(df["ph"].notnull()), "ph"].mean()


df.loc[(df["Potability"]==0)&(df["ph"].isnull()), "ph"]=nonPotablePhMeanValue
df.loc[(df["Potability"]==1)&(df["ph"].isnull()), "ph"]=potablePhMeanValue



df.isnull().sum()


#train test split

X = df.drop("Potability", axis=1).values #independent values

y = df["Potability"].values #target value potable or non-potable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


x_train_max = np.max(X_train)
x_train_min = np.min(X_train)
X_train = (X_train - x_train_min)/(x_train_max - x_train_min)
X_test = (X_test - x_train_min) / (x_train_max - x_train_min)

# %% Modelling: decision tree and random forest

models = [("DTC" , DecisionTreeClassifier(max_depth=3)),
          ("RF", RandomForestClassifier())]

finalResult = [] # score list
cmList = [] # confusion matrix list

for name, model in models:
    model.fit(X_train, y_train) #training
    model_result = model.predict(X_test) #prediction
    
    score = precision_score(y_test, model_result)
    finalResult.append((name,score))
    
    cm = confusion_matrix(y_test, model_result)
    cmList.append((name,cm))
    
print(finalResult)

for name, i in cmList:
    plt.figure()
    sns.heatmap(i, annot = True, linewidths = 0.8, fmt = ".0f")
    plt.title(name)
    plt.show()

# %% Evaluation: decision tree visualizatiion

decisionTree_classifier = models[0][1]

plt.figure(figsize = (25,20))
tree.plot_tree(decisionTree_classifier, feature_names = df.columns.tolist()[:-1],
               class_names = ["0","1"],
               filled = True,
               precision = 5)

plt.show()

# %% Hyperparameter tuning: random forest

model_params = {
    "Random Forest":
        {
            "model":RandomForestClassifier(),
            "params":
                {
                    "n_estimators": [10,50,100],
                    "max_features": ["auto", "sqrt", "log2"],
                    "max_depth": list(range(1,21,3))
                    }
            }
    }

crossValidation = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 2)
scores = []

for model_name, params in model_params.items():
    
    rs = RandomizedSearchCV(params["model"], params["params"], cv = crossValidation, n_iter = 10)
    rs.fit(X,y)
    scores.append([model_name, dict(rs.best_params_), rs.best_score_])

print(scores)




































