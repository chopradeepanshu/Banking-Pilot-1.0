
# coding: utf-8

# # Use case - Customer Churn Prediction and Analysis

# In[114]:

#Loding and importing all the required packages
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns
import sklearn.svm
import sklearn.tree
import sklearn.ensemble
import sklearn.neighbors
import sklearn.linear_model
import sklearn.metrics
import sklearn.preprocessing
from scipy import stats
from sklearn import preprocessing as prep
from sklearn.preprocessing import Imputer
import pylab as pl
plt.style.use('ggplot')
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn import tree
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA # Principal Component Analysis module
from sklearn.cluster import KMeans # KMeans clustering 
from scipy.spatial.distance import cdist
from sklearn import cluster
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
from mlxtend.plotting import plot_confusion_matrix


# ### Loading dataset

# In[115]:

file = 'BankCustomerChurnDataFeb19_Heavy.xlsx'


# In[116]:

df = pd.read_excel(file, sheetname="BankCustomerData") 
print (df.shape)


# ## Data Pre-Processing

# In[117]:

df.head()


# ## Loading Sentimental Analysis Score of the Customer

# In[118]:

#file = 'BOACustomerSentimentScore.csv'
file = 'CustomerSentimentScoreData.xlsx'


# In[119]:

#tweets_df = pd.read_csv(file) 
tweets_df = pd.read_excel(file, sheetname="SentimentScore") 
tweets_df.rename(columns={'CUSTOMER.CODE': 'CUSTOMER.CODE', 'SCORE': 'Sentimental.Score'}, inplace=True)
#tweets_df
#.drop(["Unnamed: 0"], axis = 1, inplace=True)


# In[120]:

tweets_df['Sentimental.Score'] = (tweets_df['Sentimental.Score']).astype(int) 
#tweets_df


# ## Merge the Original Dataframe with the Sentimental Score of Customer

# In[121]:

#df['Sentimental.Score'] = df['Sentimental.Score'].round()
tweets_df['Sentimental.Score'].fillna(0, inplace=True)
#tweets_df


# In[122]:

df = pd.merge(df, tweets_df, on='CUSTOMER.CODE', how='outer')
df.shape


# In[123]:

df['Sentimental.Score'].fillna(0, inplace=True)
df


# # Exploratory Data Analysis

# In[124]:

y_True = df["Exited"][df["Exited"] == 1]
print ("Churn Percentage = "+str( (y_True.shape[0] / df["Exited"].shape[0]) * 100 ), '%') 


# In[125]:

labels = 'Retained', 'Exited'
sizes = [df.Exited[df['Exited']==0].count(), df.Exited[df['Exited']==1].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90)
ax1.axis('equal')
plt.title("Proportion of customer churned and retained", size = 20)
plt.savefig("1.png")
plt.show()
plt.clf()


# In[126]:

#sentimental_df = tweets_df


# In[127]:

def func(x):
    if x < -5:
        return "Worst"
    elif x < 0 and x > -5: 
        return "Bad"
    elif x == 0:
        return "Neutral"
    elif x > 0 and x < 3:
        return "Good"
    elif x > 3 and x < 5:
        return "Very Good"
    else:
        return 'Excellent'

df['Sentimental.Category'] = df['Sentimental.Score'].apply(func) 
df


# In[128]:

def change_column_order(df, col_name, index):
    cols = df.columns.tolist()
    cols.remove(col_name)
    cols.insert(index, col_name)
    return df[cols]


# In[129]:

df = change_column_order(df, 'Exited', len(df.columns)-1)
df


# In[130]:

# We first review the 'Status' relation with categorical variables
fig1, axarr1 = plt.subplots(1, 1, figsize=(10, 6))
sns.countplot(x='Exited', hue = 'Exited',data = df, ax=axarr1)
fig, axarr = plt.subplots(3, 3, figsize=(20, 12))
sns.countplot(x='COUNTRY', hue = 'Exited',data = df, ax=axarr[0][0])
sns.countplot(x='GENDER', hue = 'Exited',data = df, ax=axarr[0][1])
sns.countplot(x='HasCreditCard', hue = 'Exited',data = df, ax=axarr[0][2])
sns.countplot(x='IsActiveMember', hue = 'Exited',data = df, ax=axarr[1][0])
sns.countplot(x='CUSTOMER.RATING', hue = 'Exited',data = df, ax=axarr[1][1])
sns.countplot(x='CALC.RISK.CLASS', hue = 'Exited',data = df, ax=axarr[1][2])
sns.countplot(x='Sentimental.Category', hue = 'Exited',data = df, ax=axarr[2][0])
sns.countplot(x='Marital Status', hue = 'Exited',data = df, ax=axarr[2][1])
sns.countplot(x='Education', hue = 'Exited',data = df, ax=axarr[2][2])
plt.savefig("2.png")
plt.show()
plt.clf()


# In[131]:

# We first review the 'Status' relation with categorical variables
fig1, axarr1 = plt.subplots(1, 1, figsize=(20, 6))
sns.countplot(x='TypeofLoan', hue = 'Exited',data = df, ax=axarr1)
fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.countplot(x='AccountType', hue = 'Exited',data = df, ax=axarr[0][0])
sns.countplot(x='HomeOwnership', hue = 'Exited',data = df, ax=axarr[0][1])
sns.countplot(x='ReasonWhyCustomerLeft', hue = 'Exited',data = df, ax=axarr[1][0])
sns.countplot(x='Number of Banking Issues raised', hue = 'Exited',data = df, ax=axarr[1][1])
plt.savefig("3.png")
plt.show()
plt.clf()


# In[64]:

#_, ax = plt.subplots(1, 3, figsize=(18, 6))
#plt.subplots_adjust(wspace=0.3)
#sns.swarmplot(x = "NumOfProducts", y = "AGE", hue="Exited", data = df, ax= ax[0])
#sns.swarmplot(x = "HasCrCard", y = "AGE", data = df, hue="Exited", ax = ax[1])
#sns.swarmplot(x = "IsActiveMember", y = "AGE", hue="Exited", data = df, ax = ax[2])


# In[132]:

df[['AGE']]=df[['AGE']].astype(int)
AgeDf=df.sort_values(by='AGE')
AgeDf["AGE"].value_counts().plot.bar(figsize=(20,6))
plt.title("Customer Count Per Age")
plt.savefig("4.png")
plt.show()
plt.clf()


# In[133]:

facet = sns.FacetGrid(df, hue="Exited",aspect=3)
facet.map(sns.kdeplot,"AGE",shade= True)
facet.set(xlim=(0, df["AGE"].max()))
facet.add_legend()
plt.title('Customer Exited Per Age Distribution')
plt.savefig("5.png")
plt.show()
plt.clf()


# In[134]:

'''_, ax =  plt.subplots(1, 2, figsize=(15, 7))
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
sns.scatterplot(x = "AGE", y = "Max End Balance", hue = "Exited", cmap = cmap, sizes = (10, 200), data = df, ax=ax[0])
sns.scatterplot(x = "AGE", y = "CreditScore", hue = "Exited", cmap = cmap, sizes = (10, 200), data = df, ax=ax[1])'''


# In[135]:

#plt.figure(figsize=(8, 8))
#sns.swarmplot(x = "HasCreditCard", y = "AGE", data = df, hue="Exited")


# In[136]:

'''facet = sns.FacetGrid(df, hue="Exited",aspect=3)
facet.map(sns.kdeplot,"Max End Balance",shade= True)
facet.set(xlim=(0, df["Max End Balance"].max()))
facet.add_legend()
plt.show()'''


# In[137]:

'''facet = sns.FacetGrid(df, hue="Exited",aspect=3)
facet.map(sns.kdeplot,"CreditScore",shade= True)
facet.set(xlim=(0, df["CreditScore"].max()))
facet.add_legend()
plt.show()'''


# In[140]:

Boxdf=df
#Boxdf.drop(["CUSTOMER.CODE"], axis = 1, inplace=True) 


# In[141]:


plt.figure(figsize=(25,8),dpi=80, facecolor='w', edgecolor='k')
bplot = Boxdf.boxplot(patch_artist=True)
plt.xticks(rotation=90)
plt.title('Boxplot Analysis for Outliers')
plt.savefig("6.png")
plt.show()
plt.clf()


# In[142]:

Positive_Sentiments = df['Sentimental.Score'][df['Sentimental.Score'] > 0].count()
Negative_Sentiments = df['Sentimental.Score'][df['Sentimental.Score'] < 0].count()
Neutral_Sentiments = df['Sentimental.Score'][df['Sentimental.Score'] == 0].count()
print('Positive_Sentiments : ', Positive_Sentiments)
print('Negative_Sentiments : ', Negative_Sentiments)
print('Neutral_Sentiments : ', Neutral_Sentiments)


# In[144]:

labels = 'Worst', 'Bad', 'Neutral', 'Good', 'Very Good', 'Excellent'
sizes = [df['Sentimental.Score'][df['Sentimental.Score'] <- 5].count(), 
         df['Sentimental.Score'][(df['Sentimental.Score'] > - 5) & (df['Sentimental.Score'] < 0)].count(),
         df['Sentimental.Score'][df['Sentimental.Score'] == 0].count(),
         df['Sentimental.Score'][(df['Sentimental.Score'] > 0) & (df['Sentimental.Score'] < 3)].count(),
         df['Sentimental.Score'][(df['Sentimental.Score'] > 3) & (df['Sentimental.Score'] < 5)].count(),
         df['Sentimental.Score'][df['Sentimental.Score'] > 5].count()]

explode = (0.05,0.05,0.05,0.05,0.05,0.05)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=False, startangle=90)
ax1.axis('equal')
plt.title("Customer Sentiments Count", size = 20)
plt.savefig("7.png")
plt.show()
plt.clf()


# In[145]:

df.drop(["ReasonWhyCustomerLeft","Sentimental.Category"], axis = 1, inplace=True) 


# ### Handling the Categorical attributes by Encoding to Numerical values for analysis

# In[146]:

# Discreet value integer encoder
label_encoder = preprocessing.LabelEncoder()


# In[147]:

object_cols = df.select_dtypes(include=['object']).copy()
ObjCols = []

for col in object_cols:  # Iterate over chosen columns
    df[col] = label_encoder.fit_transform(df[col])
    le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    ObjCols.append((col,le_name_mapping))

discreet_encoding_values = pd.DataFrame(ObjCols)
print(discreet_encoding_values)


# In[148]:

y = df['Exited'].as_matrix().astype(np.int)
y.size


# In[149]:

# Remove the Dependent Variable from the dataframe and put it into Independent Variable X
df.drop(["Exited"], axis = 1, inplace=True) 
X = df.as_matrix().astype(np.float)


# ### Standardaizing the data

# In[150]:

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)


# In[151]:

### Splitting the dataset into the Training set and Test set


# In[152]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1234, stratify = y)


# In[153]:

### Get the Accuracy and Precision


# In[154]:

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def getAccuracy(y_test, y_pred):
    return accuracy_score(y_test, y_pred)*100

def getPrecision(y_test, y_pred):
    return classification_report(y_test, y_pred)


# #************************************************************************
# # Part 2 - Now let's make the ANN!
# #************************************************************************

# In[155]:

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential # To initialize our Neural Network.
from keras.layers import Dense      # That requires to build our ANN.
from keras.layers import Dropout


# ### Initialising the ANN. (We wanted to make our Sequence of Layers Model)

# In[168]:

classifier = Sequential()


# In[169]:

# Adding the input layer and the first hidden layer
# Why units = 6. Explanation: It is first Hidden Layer. We have 11 Independent Variables. Formula use 11+1/2 ==> 12/2 ==>6    
# Activation = relu. Explanation: We take Activation function in hidden layer as Rectifier Function and Output as Sigmoid.
# input_dim = 11. Explanation: As we have 11 independent variables.


# In[170]:

classifier.add(Dense(units = 53, kernel_initializer = 'uniform', activation = 'relu', input_dim = 104))


# In[171]:

#Dropout to be used when you find, the model is too much overfitted. 0.1 means it is going to eliminate 10% of neurons from the current layer. 
# If the overfitting problem does not solve, keep increasing dropout to 0.1 till 0.5 but not go beyond 0.5 else underftting problem will occur.


# In[172]:

#classifier.add(Dropout(p = 0.1))


# In[173]:

# Adding the second hidden layer
#classifier.add(Dense(units = 53, kernel_initializer = 'uniform', activation = 'relu'))
# Everything remains the same only input_dim will not be used here, because that is only required for 1st Hidden Layer.
# classifier.add(Dropout(p = 0.1))


# In[174]:

# Adding the output layer
# Since this is an Output Layer, we wanated to have the Activation Function as Sigmoid.
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# In[175]:

# Hardcoded for Development, this has to be taken from Tuning the ANN

optimizer = 'rmsprop'
epochs = 20
batch_size = 1000


# In[176]:

# Compiling the ANN
# optimizer = The Algorithm you wanna to use to find the optimal set of weights in NN. The algorithm use is Stochastic Gradient Descent. And the best Algo for this is Adam.
# loss      = Loss function within the Adam (Stochastic GD Algo) algo. That is the loss function we wanted to optimize for the optimal weights. binary_crossentropy because we have 2 categories only.
# metrics   = is required to be in list hence we added the values in parenthesis.
classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[177]:

callbacks = [EarlyStopping(monitor='val_loss', patience=2),ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)] 


# In[178]:

# Fitting the ANN to the Training set
# batch_size = Can be understood from the flowchart of training ANN using Stochastic GD. Setp 6. We need to update the weights after each observation (Reinforcement Learning) 
# Or we can update the weights after only batch of observations. Hence we took batch size = 10. which means after 10 observation, we will update the weights.
# epochs = Number of Iterations.
# 75 Batches of 10 Iterations each
classifier.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, callbacks=callbacks)


# In[51]:

#************************************************************************
# Part 3 - Making predictions and evaluating the model
#************************************************************************


# In[179]:

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#new_prediction = classifier.predict(scaler.transform([X_test[9]]))
#new_prediction = (new_prediction > 0.5)

y_test_record = []
y_pred_record = []

for i in range(len(y_test)):
    print('count ', i, 'Original Value :: ',y_test[i], 'Predicted Value :: ',y_pred[i])
    y_test_record.append(y_test)
    y_pred_record.append(y_pred)
    

#print('Original Value :: ',y_test[9])
#print('Predicted Value :: ',y_pred[9])



# In[180]:

#y_pred = classifier.predict(X_test[9])

new_prediction = classifier.predict(np.array([X_test[1908]]))

print(new_prediction)
new_prediction = (new_prediction > 0.5)
print(new_prediction)


# In[181]:

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True,  fmt='')
plt.title('ANN Confusion Matrix')
plt.savefig("8.png")
plt.show()
plt.clf()
ann_test_accuracy_score = getAccuracy(y_test, y_pred)
print('ANN Test Accuracy: ', ann_test_accuracy_score, '%')
print(getPrecision(y_test, y_pred))


# In[182]:

joblib.dump(classifier, "churn-model.pkl")
joblib.dump(discreet_encoding_values, "discreet_encoding_values.pkl")


# In[183]:

# Evaluating the ANN
'''from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 53, kernel_initializer = 'uniform', activation = 'relu', input_dim = 103))
    #classifier.add(Dense(units = 53, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 1000, epochs = 10)
# cv = We have to do the cross validation 10 times.
# n_jobs will use all the cpu's to build the CV in order to complete the processing fast. But this is not working in my system.
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()

print('accuracies :: ', accuracies, ' mean :: ', mean, ' variance :: ', variance)'''


# In[184]:

# Tuning the ANN
'''def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 53, kernel_initializer = 'uniform', activation = 'relu', input_dim = 104))
    #classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [1000, 2000],
              'epochs': [10, 20],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
no_of_splits = grid_search.n_splits_

print('grid_search ', grid_search)
print('best_parameters ', best_parameters)
print('best_accuracy ', best_accuracy)
print('cv results ', grid_search.cv_results_)
print('scorer ', grid_search.scorer_)
print('number of splits  ', grid_search.n_splits_)'''


# In[ ]:




# In[ ]:



