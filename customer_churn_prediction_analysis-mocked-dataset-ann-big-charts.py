
# coding: utf-8

# # Use case - Customer Churn Prediction and Analysis

# In[41]:

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

# In[42]:

file = 'BankCustomerChurnDataFeb19_Heavy.xlsx'


# In[43]:

df = pd.read_excel(file, sheetname="BankCustomerData") 
print (df.shape)


# ## Data Pre-Processing

# In[44]:

df.head()


# ## Loading Sentimental Analysis Score of the Customer

# In[45]:

#file = 'BOACustomerSentimentScore.csv'
file = 'CustomerSentimentScoreData.xlsx'


# In[46]:

#tweets_df = pd.read_csv(file) 
tweets_df = pd.read_excel(file, sheetname="SentimentScore") 
tweets_df.rename(columns={'CUSTOMER.CODE': 'CUSTOMER.CODE', 'SCORE': 'Sentimental.Score'}, inplace=True)
#tweets_df
#.drop(["Unnamed: 0"], axis = 1, inplace=True)


# In[47]:

tweets_df['Sentimental.Score'] = (tweets_df['Sentimental.Score']).astype(int) 
#tweets_df


# ## Merge the Original Dataframe with the Sentimental Score of Customer

# In[48]:

#df['Sentimental.Score'] = df['Sentimental.Score'].round()
tweets_df['Sentimental.Score'].fillna(0, inplace=True)
#tweets_df


# In[49]:

df = pd.merge(df, tweets_df, on='CUSTOMER.CODE', how='outer')
df.shape


# In[50]:

df['Sentimental.Score'].fillna(0, inplace=True)
df


# # Exploratory Data Analysis

# In[51]:

# Check for the Nulls in complete data.
plt.figure(figsize=(15,5))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.savefig("MissingData.png")
plt.show()
plt.clf()


# In[52]:

#Checking missing data - Show percentage of Missing Data in the complete Dataset
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
f, ax = plt.subplots(figsize=(20, 8))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['Percent'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
plt.savefig("PercentageMissingDataByFeature.png")
plt.show()
plt.clf()
#missing_data.head()


# In[53]:

y_True = df["Exited"][df["Exited"] == 1]
print ("Churn Percentage = "+str( (y_True.shape[0] / df["Exited"].shape[0]) * 100 ), '%') 


# In[54]:

labels = 'Retained', 'Exited'
sizes = [df.Exited[df['Exited']==0].count(), df.Exited[df['Exited']==1].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90)
ax1.axis('equal')
plt.title("Proportion of customer churned and retained", size = 20)
plt.savefig("Overall.png")
plt.show()
plt.clf()


# In[55]:

#sentimental_df = tweets_df


# In[56]:

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


# In[57]:

def change_column_order(df, col_name, index):
    cols = df.columns.tolist()
    cols.remove(col_name)
    cols.insert(index, col_name)
    return df[cols]


# In[58]:

df = change_column_order(df, 'Exited', len(df.columns)-1)
df


# In[59]:

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
plt.savefig("ChurnBasedOnCategories_1.png")
plt.show()
plt.clf()


# In[60]:

# We first review the 'Status' relation with categorical variables
fig1, axarr1 = plt.subplots(1, 1, figsize=(20, 6))
sns.countplot(x='TypeofLoan', hue = 'Exited',data = df, ax=axarr1)
fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.countplot(x='AccountType', hue = 'Exited',data = df, ax=axarr[0][0])
sns.countplot(x='HomeOwnership', hue = 'Exited',data = df, ax=axarr[0][1])
sns.countplot(x='ReasonWhyCustomerLeft', hue = 'Exited',data = df, ax=axarr[1][0])
sns.countplot(x='Number of Banking Issues raised', hue = 'Exited',data = df, ax=axarr[1][1])
plt.savefig("ChurnBasedOnCategories_2.png")
plt.show()


# In[61]:

#_, ax = plt.subplots(1, 3, figsize=(18, 6))
#plt.subplots_adjust(wspace=0.3)
#sns.swarmplot(x = "NumOfProducts", y = "AGE", hue="Exited", data = df, ax= ax[0])
#sns.swarmplot(x = "HasCrCard", y = "AGE", data = df, hue="Exited", ax = ax[1])
#sns.swarmplot(x = "IsActiveMember", y = "AGE", hue="Exited", data = df, ax = ax[2])


# In[62]:

churn     = df[df["Exited"] == 1]
not_churn = df[df["Exited"] == 0]


# In[63]:

import plotly.offline as py#visualization
#py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization


# In[ ]:


#function  for pie plot for customer attrition types
def plot_pie(column) :
    trace1 = go.Pie(values  = churn[column].value_counts().values.tolist(),
                    labels  = churn[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    domain  = dict(x = [0,.48]),
                    name    = "Churn Customers",
                    marker  = dict(line = dict(width = 2,
                                               color = "rgb(243,243,243)")),
                    hole    = .6)
    trace2 = go.Pie(values  = not_churn[column].value_counts().values.tolist(),
                    labels  = not_churn[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    marker  = dict(line = dict(width = 2,
                                               color = "rgb(243,243,243)")
                                  ),
                    domain  = dict(x = [.52,1]),
                    hole    = .6,
                    name    = "Non churn customers")


    layout = go.Layout(dict(title = column + " distribution in customer attrition ",
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            annotations = [dict(text = "churn customers",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .15, y = .5),
                                           dict(text = "Non churn customers",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .88,y = .5)]))
    data = [trace1,trace2]
    fig  = go.Figure(data = data,layout = layout)
    #py.iplot(fig)
    py.plot(fig, filename=column, image='png')

    
object_cols = df.select_dtypes(include=['object']).copy()   
#for all categorical columns plot pie
for i in object_cols :
    plot_pie(i)


# In[ ]:

#function  for histogram for customer attrition types
def histogram(column) :
    trace1 = go.Histogram(x  = churn[column],
                          histnorm= "percent",
                          name = "Churn Customers",
                          marker = dict(line = dict(width = .5,
                                                    color = "black"
                                                    )
                                        ),
                         opacity = .9 
                         ) 
    
    trace2 = go.Histogram(x  = not_churn[column],
                          histnorm = "percent",
                          name = "Non churn customers",
                          marker = dict(line = dict(width = .5,
                                              color = "black"
                                             )
                                 ),
                          opacity = .9
                         )
    
    data = [trace1,trace2]
    layout = go.Layout(dict(title =column + " distribution in customer attrition ",
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = column,
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = "percent",
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                           )
                      )
    fig  = go.Figure(data=data,layout=layout)
    
    #py.iplot(fig)
    py.plot(fig, filename=column, image='png')


g=df.select_dtypes(include=['int64','int32','float64']).copy()
g.drop(['CUSTOMER.CODE','@ID','ACCOUNT.NUMBER','CUSTOMER.NO','LAST.CONTACT.DATE','Exited','DATE.LAST.CR.CUST','DATE.LAST.CR.AUTO','DATE.LAST.CR.BANK','DATE.LAST.DR.CUST',
        'DATE.LAST.DR.AUTO','AVAILABLE.DATE','DATE.LAST.DR.BANK','RESIDENCE.SINCE'], axis = 1, inplace=True)
#num_cols   = [x for x in g.columns if x not in object_cols + target_col + Id_col]
#for all categorical columns plot histogram    
for i in g :
    histogram(i)


# In[64]:

df[['AGE']]=df[['AGE']].astype(int)
AgeDf=df.sort_values(by='AGE')
AgeDf["AGE"].value_counts().plot.bar(figsize=(20,6))
plt.savefig("AgeDistributionByChurn_BarChart.png")
plt.show()
plt.clf()


# In[65]:

# Check Age Distribution w.r.t Exited (Dependent Variable)
facet = sns.FacetGrid(df, hue="Exited",aspect=3)
facet.map(sns.kdeplot,"AGE",shade= True)
facet.set(xlim=(0, df["AGE"].max()))
facet.add_legend()
plt.title('Customer Exited Per Age Distribution')
plt.savefig("AgeDistributionByChurn_AreaDistribution.png")
plt.show()
plt.clf()


# In[66]:

facet = sns.FacetGrid(df, hue="Exited",aspect=3)
facet.map(sns.kdeplot,"Max End Balance",shade= True)
facet.set(xlim=(0, df["Max End Balance"].max()))
facet.add_legend()
plt.show()
plt.clf()


# In[67]:

facet = sns.FacetGrid(df, hue="Exited",aspect=3)
facet.map(sns.kdeplot,"CreditScore",shade= True)
facet.set(xlim=(0, df["CreditScore"].max()))
facet.add_legend()
plt.show()


# In[68]:

Boxdf=df.copy()
Boxdf.drop(["CUSTOMER.CODE"], axis = 1, inplace=True) 
plt.figure(figsize=(20,8))
bplot = Boxdf.boxplot(patch_artist=True)
plt.xticks(rotation=90)       
plt.savefig('OutliersInData.png')
plt.show()
plt.clf()


# In[69]:

Positive_Sentiments = df['Sentimental.Score'][df['Sentimental.Score'] > 0].count()
Negative_Sentiments = df['Sentimental.Score'][df['Sentimental.Score'] < 0].count()
Neutral_Sentiments = df['Sentimental.Score'][df['Sentimental.Score'] == 0].count()
print('Positive_Sentiments : ', Positive_Sentiments)
print('Negative_Sentiments : ', Negative_Sentiments)
print('Neutral_Sentiments : ', Neutral_Sentiments)


# In[70]:

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
plt.savefig("4.png")
plt.show()
plt.clf()


# In[71]:

df.drop(["ReasonWhyCustomerLeft","Sentimental.Category"], axis = 1, inplace=True) 


# In[72]:

Correlationdf=df[df.columns[1:]].corr()['Exited'][:-1]
#Correlationdf=Correlationdf.frame(row=rownames(Correlationdf)[row(Correlationdf)], col=colnames(Correlationdf)[col(Correlationdf)], corr=c(Correlationdf))
Correlationdf=pd.DataFrame({'Features':Correlationdf.index, 'Correlation':Correlationdf.values})
Correlationdf=Correlationdf.sort_values(by='Correlation')
#print(Correlationdf)

Correlationdf = Correlationdf[Correlationdf.Features != 'Exited']
correlationvalues = Correlationdf.Correlation
my_colors = 'rgbkymc'
ax = Correlationdf.plot(kind='bar', title ="Correlation Comparision", figsize=(25, 15), legend=True, fontsize=12)
ax.set_xlabel("Features", fontsize=25)
ax.set_ylabel("Correlation", fontsize=25)
ax.set_xticklabels(Correlationdf.Features,fontsize=15)
plt.savefig("CorrelationOfFeaturesWithOutput.png")
fig =plt.show()


# In[ ]:



