#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import joblib
import lime
import lime.lime_tabular


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
st.header('Helping ICU doctors decide when to extubate')

time_on_vent = st.number_input(label = 'How long has the patient already been on the ventilator? (hours):',value=91)
anchor_age = st.number_input(label = 'Patient age (years):', value = 62)
gender = st.radio(label = 'Patient gender:', options  = ['M', 'F'])
weight = st.number_input(label = 'Patient weight (lb):', value = 182)
height = st.number_input(label = 'Patient height (inches):',value = 67)
heartrate = st.number_input(label = 'Heart rate (bpm):', value = 86)
tidalvolume = st.number_input(label = 'Tidal volume (mL):', value = 200)
temp = st.number_input(label = 'Temperature (Celcius):', value = 37.06)
hco3 = st.number_input(label = 'HCO3 (mEq/L):', value = 25.15)
creatinine = st.number_input(label = 'Creatinine (mg/dL):', value = 1.24)
bun = st.number_input(label = 'Blood urea nitrogen (mg/dL):',value = 26.35)

#tidal_weight = tidalvolume/weight
tidal_weight = tidalvolume/weight

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
prediction_percent = clf.predict_proba(sample_test)[0][0]
st.write('If you take your patient off the ventilator now, there is a ', prediction_percent,
'% chance that they will need to be reintubated')


X_train = pd.read_feather("strip_train_data")
X_scaled = preprocessor.transform(X_train)

categs= preprocessor.named_transformers_['cat']['onehot']
onehot_features = categs.get_feature_names()
numeric_features = preprocessor.transformers[0][2]
feature_names = np.concatenate((numeric_features.tolist(),onehot_features))

explainer = lime.lime_tabular.LimeTabularExplainer(X_scaled,  
                              feature_names=feature_names,  
                              #class_names=['re_intub_class'], 
                              #categorical_features=categorical_features ,
                              verbose=True, 
                              mode='classification',
                              discretize_continuous=True)

explog = explainer.explain_instance(sample_test[0,:], clf.predict_proba, num_features=5,top_labels=1)
#explog.show_in_notebook(show_table=True)

feature_list = explog.as_list()
num_top_feats = len(feature_list)

j = 0
for j in np.arange(num_top_feats):
    salient_feature = feature_list[j][0].split(' ')
    j = j+1
    for i in salient_feature:
        if i in feature_names:
            st.write(i)
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




