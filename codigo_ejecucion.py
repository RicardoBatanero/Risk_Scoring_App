#!/usr/bin/env python
# coding: utf-8

# # CODIGO DE EJECUCION

# Nota: para poder usar este código, hay que lanzarlo con el mismo entorno que se ha creado

# In[ ]:


# 1. PAQUETES
import numpy as np
import pandas as pd
import pickle


from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler
from category_encoders import TargetEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


# 2. FUNCIONES
def calidad_datos(temp):
    
    temp['antigüedad_empleo'] = temp['antigüedad_empleo'].fillna('desconocido')

    temp['dti'] = np.where(temp.dti<0,0,np.where(temp.dti>100,100,temp.dti))

    temp['porc_uso_revolving'] = np.where(temp.porc_uso_revolving>100,100,temp.porc_uso_revolving)

    temp['porc_uso_revolving'] = temp['porc_uso_revolving'].fillna(temp.porc_uso_revolving.median())
    
    temp['dti'] = temp['dti'].fillna(temp.dti.median())

    a_imputar_cero = temp.columns.difference(['porc_uso_revolving','dti'])

    temp[a_imputar_cero] = temp[a_imputar_cero].fillna(0)
    
    return(temp)


def creacion_variables(df):
    temp = df.copy()
    temp.vivienda = temp.vivienda.replace(['ANY','NONE','OTHER'],'MORTGAGE')
    temp.finalidad = temp.finalidad.replace(['wedding','educational','renewable_energy'],'otros')
    return(temp)

def ejecutar_modelos(df):
#3.CALIDAD Y CREACION DE VARIABLES
    x_pd = creacion_variables(calidad_datos(df))
    x_ead = creacion_variables(calidad_datos(df))
    x_lgd = creacion_variables(calidad_datos(df))


    with open('pipe_ejecucion_pd.pickle', mode='rb') as file:
       pipe_ejecucion_pd = pickle.load(file)

    with open('pipe_ejecucion_ead.pickle', mode='rb') as file:
       pipe_ejecucion_ead = pickle.load(file)

    with open('pipe_ejecucion_lgd.pickle', mode='rb') as file:
       pipe_ejecucion_lgd = pickle.load(file)


#5.EJECUCION
    scoring_pd = pipe_ejecucion_pd.predict_proba(x_pd)[:, 1]
    ead = pipe_ejecucion_ead.predict(x_ead)
    lgd = pipe_ejecucion_lgd.predict(x_lgd)


#6.RESULTADO
    principal = x_pd.principal
    EL = pd.DataFrame({'principal':principal,
                       'pd':scoring_pd,
                       'ead':ead,
                       'lgd':lgd                   
                       })
    EL['perdida_esperada'] = round(EL.pd * EL.principal * EL.ead * EL.lgd,2)

    return(EL)

