#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import joblib


# In[2]:


scaler = joblib.load('reintubate_scaler.sav')
logmodel = joblib.load("reintubate_model_log.sav")


# In[3]:


mask=['spontRR', 'stdABP', 'meanABP', 'stdSpontRR', 'pulseox', 'stdPulseox',
       'temp', 'heartRate', 'stdHeartRate', 'weight', 'height', 'anchor_age',
       'time_on_vent']


# In[4]:


st.set_option('deprecation.showfileUploaderEncoding', False)


# In[5]:


st.title('Extu-Mate')
st.header('Helping ICU doctors predict successful removal of mechanical ventilation')


# In[6]:


csv_file = st.file_uploader(
    label="Upload a csv file containing your patient's data.", type=["csv"], encoding="utf-8"
)

if csv_file is not None:
    df = pd.read_csv(csv_file)
    x = df[mask]
    x_scaled = scaler.transform(x)
    sample_df = df.copy()
    sample_df[mask] = x_scaled.flatten()
    sample_test = sample_df.drop(labels=['re_intub_class'],axis=1).values
    logmodel.predict(sample_test)
    prediction_percent = logmodel.predict_proba(sample_test)[0,0]
    st.write('There is a ', prediction_percent,
    '% likelihood that extubation will be successful')
    #st.dataframe(df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




