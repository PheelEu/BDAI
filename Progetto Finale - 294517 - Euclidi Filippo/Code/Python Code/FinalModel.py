
# @Author Euclidi Filippo - matr. 294517

# This file contains the final model tuned as best as possible
# The model chosen after comparing all the results is the Logistic Regression (no suspense)

#Importing all the needed libraries:

from xml.etree.ElementPath import find
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Features Selection methods imports
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
  
# Data preprocessing methods imports
from sklearn import preprocessing 

# Train test split imports
from sklearn.model_selection import train_test_split

# Score evaluating methods imports
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Classifications Models imports
from sklearn.linear_model import LogisticRegression

# Model Selection and Cross Validation imports
from sklearn.model_selection import KFold

# We are going to use this function to calculate the accuracy of the predicted values

def score(y_test, y_pred, specification):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=specification)
    c_mat = confusion_matrix(y_test, y_pred)
    return acc, f1, c_mat

# Lets re-create our function which we will use to determine the best number of features
def findBestFeatures(model, X_train, y_train):
    accuracy_list_train = []
    k=np.arange(1,21,1)
    for each in k:
        x_new = SelectKBest(score_func=chi2, k=each).fit_transform(X_train, y_train)
        model.fit(x_new,y_train)
        accuracy_list_train.append(model.score(x_new,y_train))   
    plt.plot(k,accuracy_list_train,color="green",label="train")
    plt.xlabel("k values")
    plt.ylabel("train accuracy")
    plt.legend()
    plt.show()

# Lets also use the kfold cross-validation as this dataset is not very big 
 
def kFoldCrossValidation(model, X, y):
    k = 10
    kf = KFold(n_splits=k, random_state=None)
    acc_score = []
    f1_score = []
    prec_matrix = []
    i = 0
        
    for train_index , test_index in kf.split(X):
        i+=1
        X_train , X_test = X[train_index,:],X[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]
        
        model.fit(X_train,y_train)
        pred_values = model.predict(X_test)
        acc, f1, prec_m = score(y_test, pred_values, 'weighted')
        acc_score.append(acc)
        f1_score.append(f1)
        prec_matrix.append(prec_m)
        print("Round number "+ str(i) + ", Accuracy score: "+str(acc) + ", F1 Score:" + str(f1) 
          + ",\nPrecision Matrix:\n" + str(prec_m) + "\n")
            
    avg_acc_score = sum(acc_score)/k
    avg_f1_score = sum(f1_score)/k
    avg_prec_mat = sum(prec_matrix)/k
    print("Average results after 10 folds cross validation: ")
    print("Accuracy score: "+str(avg_acc_score) + " \nF1 Score:" + str(avg_f1_score) 
          + " \nAverage Precision Matrix:\n" + str(avg_prec_mat)+"\n")

# Setting train and test set by importing the datasets
df_train = pd.read_csv("./Code/Python Code/archive/train.csv")
df_test = pd.read_csv("./Code/Python Code/archive/test.csv")

# First of all lets create our dataset X (multidimensional array) by removing the price_range
# column which will be our target value to check how good is a classifier

# We should also make X a numpy array and we do that by using the .to_numpy() function

X = df_train.drop(columns = ['price_range'])
X = df_train.iloc[:, 0:-1]

X_t = df_test.drop(columns = ['id'])

# Visualizing the dataset
print("Training set:\n" + str(X.head()))
print("Test set:\n" + str(X_t.head()))

# Now we will create our y array

y = df_train['price_range'].values

# Lets do the necessary preprocessing by applying the Standard Scaler 

std_scaler = preprocessing.StandardScaler().fit(X)

X = std_scaler.transform(X) #Preprocessed for better accuracy

X_t = std_scaler.transform(X_t)

print(X) #Lets visualize it

# Lets split our database in to training and test 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, stratify=y)

# Now lets train the Logistic Regression Classifier

# Now lets train the Logistic Regression Classifier

print("Logistic Regression Classifier\n")

# We will do ten tries and report the results for each step
logreg = LogisticRegression()
kFoldCrossValidation(logreg, X, y)
logreg.fit(X_train, y_train)    
pred = logreg.predict(X_test)
acc, f1, c_mat = score(y_test, pred, 'weighted')
print("Final results using the full training and test set:\n" + "Accuracy score: "+str(acc)
      +", F1 Score:" + str(f1) + ",\nConfusion Matrix:\n" + str(c_mat))

final_prediction = logreg.predict(X_t)

print("")

df_test['price_range'] = final_prediction

print(df_test.head(10))