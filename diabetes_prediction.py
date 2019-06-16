
#importing libraries
import pandas as pnd

#importing the dataset
dataset = pnd.read_csv('diabetes.csv') #please provide the proper directory while testing

#independent variable matrix
X = dataset.iloc[:,0:8].values 

#dependent variable
y = dataset.iloc[:,8].values

#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 ,random_state = 0)

#scaling the dataset to take dataset to a common scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler();
X_train = sc_X.fit_transform(X_train);
X_test = sc_X.fit_transform(X_test);

#performing logistic regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train,y_train)

#prediction
y_predict = log_reg.predict(X_test)

#confusion matrix for the accuracy of system
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predict)
accuracy = (cm[0,0]+cm[1,1])/len(X_test) * 100 
print("accuracy " + str(accuracy) + "%" )
