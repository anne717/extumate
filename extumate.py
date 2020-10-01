#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
import joblib


# In[2]:


preprocessor = joblib.load('reintubate_preprocessor_strip.sav')
clf = joblib.load("reintubate_model_strip.sav")


# In[3]:


df_columns=['time_on_vent', 'anchor_age', 'heartrate', 'weight', 'hco3',
       'creatinine', 'bun', 'height', 'tidalvolume', 'temp', 're_intub_class', 'gender','tidal_weight']


# In[4]:


st.set_option('deprecation.showfileUploaderEncoding', False)


# In[5]:


st.title('Extu-Mate')
st.header('Helping ICU doctors predict successful removal of mechanical ventilation')

time_on_vent = st.number_input(label = 'How long has the patient already been on the ventilator? (hours):')
anchor_age = st.number_input(label = 'Patient age (years):')
gender = st.radio(label = 'Patient gender:', options  = ['M', 'F'])
weight = st.number_input(label = 'Patient weight (lb):')
height = st.number_input(label = 'Patient height (inches):')
heartrate = st.number_input(label = 'Heart rate (bpm):')
tidalvolume = st.number_input(label = 'Tidal volume (mL):')
temp = st.number_input(label = 'Temperature (Celcius):')
hco3 = st.number_input(label = 'HCO3 (units):')
creatinine = st.number_input(label = 'Creatining (units):')
bun = st.number_input(label = 'Blood urea nitrogen ():')

#tidal_weight = tidalvolume/weight
tidal_weight = 5/2

re_intub_class = 0

test_data = np.array([[time_on_vent, anchor_age, heartrate, weight, hco3,
       creatinine, bun, height, tidalvolume, temp, re_intub_class, gender,tidal_weight]])

df = pd.DataFrame(data = test_data, columns=df_columns)
#x = df[df.columns.drop(['re_intub_class'])]
#x_columns = x.columns

df.drop('re_intub_class',axis=1,inplace=True)
df_scaled = preprocessor.transform(df)
sample_df = df_scaled.copy()
sample_test = df_scaled.flatten().reshape(1,-1)
#sample_test = sample_df.drop(labels=['re_intub_class'],axis=1).values

clf.predict(sample_df)
prediction_percent = clf.predict_proba(sample_test)[0,0]
st.write('There is a ', prediction_percent,
'% likelihood that extubation will be successful')

# In[6]:


# =============================================================================
# csv_file = st.file_uploader(
#     label="Upload a csv file containing your patient's data.", type=["csv"], encoding="utf-8"
# )
# 
# if csv_file is not None:
#     df = pd.read_csv(csv_file)
#     x = df[mask]
#     x_scaled = scaler.transform(x)
#     sample_df = df.copy()
#     sample_df[mask] = x_scaled.flatten()
#     sample_test = sample_df.drop(labels=['re_intub_class'],axis=1).values
#     logmodel.predict(sample_test)
#     prediction_percent = logmodel.predict_proba(sample_test)[0,0]
#     st.write('There is a ', prediction_percent,
#     '% likelihood that extubation will be successful')
#     #st.dataframe(df)
# =============================================================================


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




