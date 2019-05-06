
# coding: utf-8

# In[40]:

import pandas as pd
import numpy as np
import re
import collections
import matplotlib.pyplot as plt
import  plotly
# Packages for data preparation
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras import models
from keras import layers
from keras import regularizers
from sklearn.externals import joblib


# In[41]:

file = 'BOA_Customer_Tweets_And_Its_Score.csv'


# In[42]:

tweets_df = pd.read_csv(file, index_col=False)
tweets_df



# In[43]:

Customer_Tweet = tweets_df.loc[tweets_df['Customer.Code'] == 15009449]
Customer_Tweet = Customer_Tweet[['Customer.Code','Sentimental.Score', 'text']]

print(Customer_Tweet)


# In[44]:

joblib.dump(tweets_df, "customer-tweets.pkl")


# In[ ]:




# In[ ]:




# In[ ]:



