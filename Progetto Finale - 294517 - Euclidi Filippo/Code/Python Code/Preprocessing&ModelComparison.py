# @Author Euclidi Filippo - matr. 294517

# Importing all needed libraries.

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Features Selection methods imports
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
  
# Data preprocessing methods imports
from sklearn import preprocessing 

# Train test split imports
from sklearn.model_selection import train_test_split

# Score evaluating methods imports
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Classifications Models imports
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from xgboost import XGBClassifier  #To use this, use compatible kernel 
                                    # (conda tensorflow on mac m1 is not compatible)

# Model Selection and Cross Validation imports
from sklearn.model_selection import KFold

# Neural Network imports
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers


# I'm now going to create a function to calculate the accuracy of the predicted values

def score(y_test, y_pred, specification):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=specification)
    c_mat = confusion_matrix(y_test, y_pred)
    return acc, f1, c_mat

def kFoldCrossValidation(model, X, y):
    k = 10
    kf = KFold(n_splits=k, random_state=None)
    acc_score = []
    f1_score = []
    prec_matrix = []
        
    for train_index , test_index in kf.split(X):
        X_train , X_test = X[train_index,:],X[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]
        
        model.fit(X_train,y_train)
        pred_values = model.predict(X_test)
        acc, f1, prec_m = score(y_test, pred_values, 'weighted')
        acc_score.append(acc)
        f1_score.append(f1)
        prec_matrix.append(prec_m)
            
    avg_acc_score = sum(acc_score)/k
    avg_f1_score = sum(f1_score)/k
    avg_prec_mat = sum(prec_matrix)/k
    print("Accuracy score: "+str(avg_acc_score) + " \nF1 Score:" + str(avg_f1_score) 
          + " \nAverage Precision Matrix:\n" + str(avg_prec_mat))
    
    
# Setting train and test set by importing the datasets
df_train = pd.read_csv("./Code/Python Code/archive/train.csv")
df_test = pd.read_csv("./Code/Python Code/archive/test.csv")


# First of all lets create our dataset X (multidimensional array) by removing the price_range
# column which will be our target value to check how good is a classifier

# We should also make X a numpy array and we do that by using the .to_numpy() function

X = df_train.drop(columns = ['price_range'])
X = df_train.iloc[:, 0:-1]

# Now we will create our y array containing all the values which we expect from our classifier

y = df_train['price_range'].values

# We are now going to do some feature selection using the chi-square function 

bestfeatures = SelectKBest(score_func=chi2, k=2)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score'] #naming the dataframe columns
print(featureScores)

# Lets create a function to determine the best number of features for all the models
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
    
    
# We need to do some preprocessing 
# Lets apply a Standard Scaler which standardizes the features by removing the mean
# and scaling to unit variance 
# .fit(X) computes the mean and std (standard deviation) of the X array
# .transform(X) uses the computed mean and std and performs on X the standardization

X_NoPrep = X.to_numpy()

X = preprocessing.StandardScaler().fit(X).transform(X) #Preprocessed for better accuracy
X_norm = preprocessing.MinMaxScaler().fit(X).transform(X) #Normalized for feature selection

# Doing some feature selection is going to give us better results and reduce the time needed 
# for computation
X_new = SelectKBest(chi2, k=5).fit_transform(X_norm, y)

print(X) #Lets visualize it
print(X_norm)

# Lets split our database in to training and test and do some model comparison on 
# these datasets
 
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.10, stratify=y)
X_trainNoP, X_validationNoP, y_trainNoP, y_validationNop = train_test_split(X_norm, y, test_size = 0.10, stratify=y)


# As first Model we will use a Logistic Regression Classifier

#avg = np.mean(X_train, axis=0)
#std = np.std(X_train,axis=0)
#X_train = (X_train-avg/std)

# I tried adding doing some feature scaling, but the models Accuracy didn't seem to be
# improving in any way while doing so, some also showed worse results

print("Logistic Regression Classifier")
logreg = LogisticRegression()
kFoldCrossValidation(logreg, X, y)

#findBestFeatures(logreg, X_trainNoP, y_trainNoP)
print("")

# As second model we will use a Decision Tree Classifier

print("Decision Tree Classifier")
clf=DecisionTreeClassifier(max_depth=6)
kFoldCrossValidation(clf, X, y)
plot_tree(clf, filled=True)
plt.show()
print("")

# Since the decision tree classifier wasn't very successful lets try with 
# a Random Forest Classifier

print("Random Forest Classifier")
rf = RandomForestClassifier(n_estimators=200)
kFoldCrossValidation(rf, X, y)
print("")

# We will now also try with a Support Vector Machine, since the random forest model wasn't 
# a great success by any means

print("Support Vector Machine Classifier")
svm_clf = SVC()
kFoldCrossValidation(svm_clf, X_NoPrep, y)
#findBestFeatures(svm_clf, X_trainNoP, y_trainNoP)
print("")

# Lets also try with some other classifiers like the Bagging Classifier, the AdaBoost 
# (adaptive boosting), the Gradient Boosting and the XGBoost (extreme gradient boosting)
# 200 estimators will be used for all of them, to keep it even and not make computational 
# time too long

# First lets use the bagging classifier

print("Bagging Classifier")
bc = BaggingClassifier(n_estimators = 200)
kFoldCrossValidation(bc, X, y)
#findBestFeatures(bc, X_trainNoP, y_trainNoP)
print("")

#The bagging classifier seems to give an ok result although the computational time it's very long

# Then we will use the Adaptive Boosting

print("AdaBoost")

ab = AdaBoostClassifier(n_estimators = 200)
X_AdaBoost = SelectKBest(chi2, k=3).fit_transform(X_norm,y)

#X_AdaBoost = X_AdaBoost[:, [0,1,2,5,6]]
#print(X_AdaBoost)

kFoldCrossValidation(ab, X_AdaBoost, y)
#findBestFeatures(ab, X_norm, y)
print("")

# Then the Gradient Boosting

print("Gradient Boosting")
gb = GradientBoostingClassifier(n_estimators = 200)
kFoldCrossValidation(gb, X, y)
#findBestFeatures(gb, X_trainNoP, y_trainNoP)
print("")


# And finally the Extreme Gradient Boosting

#Uncomment only if using xgboost compatible kernel
#print("Extreme Gradient Boosting:")
# Using label encoder false, to remove a warning, and eval_metric='logloss' for improved 
# results eval_metric='mlogloss' also shows good results
#xgb = XGBClassifier(n_estimators = 200,  use_label_encoder=False, eval_metric='logloss')
#kfoldCrossValidation(xgb, X, y)
#print("")

# All of these models could be implemented in to a for cycle, but i wanted to show all the 
# actual full code behind it even if very repetitive for explainability purposes.

# As we can see the best models are the Linear Regression, the Logistic Regression,
# the Gradient Boosting and the Extreme Gradient Boosting.
# Some decent results were also seen in the Random Forest Classifier, the Bagging 
# Classifier and the Support Vector Machine.
# The support vector machine shows great results when there is no preprocessing applied 
# to the X_train dataset 95% accuracy

# Lets see now how a Neural Network can deal with this dataset and its classification

# We need to make y categorical as we are going to use a softmax as final layer 
# for multi-class classification
y = to_categorical(y)

# We are going to use a different training set and test set for the Neural Network since we 
# made y categorical
X_train, X_validation, y_train, y_validation = train_test_split(X_new, y, test_size = 0.10)

# The model that we are going to use is sequential
model = models.Sequential()
#The first layer that you define is the input layer. This layer needs to know the 
# input dimensions of your data.
# Dense = fully connected layer (each neuron is fully connected to all neurons 
# in the previous layer)
model.add(layers.Dense(18, activation='relu',input_dim=5))
# Add one hidden layer (after the first layer, you don't need to specify 
# the size of the input anymore)
model.add(layers.Dense(18, activation='relu'))
# If you don't specify anything, no activation is applied (ie."linear" activation: a(x) = x)
model.add(layers.Dense(4,activation='softmax'))

opt = optimizers.Adam()

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Fit the model to the training data and record events into a History object.
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, verbose=1)

# Model evaluation
test_loss,test_pr = model.evaluate(X_validation,y_validation)
y_pred = model.predict(X_validation)
print(test_pr)

pred = list()

test = list()

for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))

for i in range(len(y_validation)):
    test.append(np.argmax(y_validation[i]))

# Plot loss (y axis) and epochs (x axis) for training set and validation set
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(history.epoch,
np.array(history.history['loss']),label='Train loss')
plt.plot(history.epoch,
np.array(history.history['val_loss']),label = 'Val loss')
plt.legend()
plt.show()

acc, f1, prec_m = score(pred, test, 'weighted')
print("Accuracy score: "+str(acc) + " \nF1 Score:" + str(f1) 
          + " \nPrecision Matrix:\n" + str(prec_m))


