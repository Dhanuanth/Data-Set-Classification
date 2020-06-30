import sys
import scipy
import numpy as np
import matplotlib.pyplot as plt
import  pandas as pd
from pandas.plotting import scatter_matrix
import sklearn
from sklearn import linear_model
from sklearn.datasets import load_iris
import seaborn as sns
from sklearn import model_selection
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
dataset=load_iris()

x=pd.DataFrame(dataset.data,columns=['sepal length','sepal width','petal length','petal width'])
y=pd.DataFrame(dataset.target,columns=['species'])
X=pd.concat([x,y],axis=1)

x.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
x.hist()
scatter_matrix(x)
#sns.pairplot(X,hue='species',size=8)
plt.show()


x=X.iloc[1:,:4]
y=X.iloc[1:,4]
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2,random_state=1)
models=[]
#######       Logistic Regression     ##############
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
#######         Linear Discriminant Analysis       #######
models.append(('LDA',LinearDiscriminantAnalysis()))
######        K-Nearest Neighbours ###########
models.append(('KNN',KNeighborsClassifier()))
######   Classification And Regression trees  ####

##### GaussianNaiveBayes #######
models.append(('NB',GaussianNB()))
######## Support Vector machines ######
models.append(('SVM',SVC(gamma='auto')))

results=[]
names=[]
for name , model in models:
    kfold=StratifiedKFold(n_splits=20,random_state=1,shuffle=True)
    cv_results=cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print("%s %f (%f)" %(name,cv_results.mean(),cv_results.std()))

plt.boxplot(results,labels=names)
plt.title('Algorithm Comparison')
plt.show()

svm=SVC(gamma='auto')
svm.fit(x_train,y_train)
y_pred=svm.predict(x_test)
print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))
print(classification_report(y_pred,y_test))