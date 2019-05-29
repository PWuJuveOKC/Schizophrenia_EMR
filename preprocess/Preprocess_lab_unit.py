import pandas as pd
import warnings
import numpy as np
import os
os.chdir("C:/Users/Peter Xu/Desktop/Yuanjia/schi/trac_ 3772_schizophrenia")
warnings.filterwarnings("ignore", category=DeprecationWarning)
desired_width = 300
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)

# Load Data
file1 = pd.read_csv('trac_ 3772_schizophrenia_measurement.csv',delimiter='|',nrows=300,
                    dtype={'visit_occurrence_id':object},parse_dates=True)
file = pd.read_csv('trac_ 3772_schizophrenia_measurement.csv',delimiter='|',
                    dtype={'visit_occurrence_id':object},parse_dates=True)

# personId = file.person_id.unique()

parse_dates = ['valid_start_date', 'valid_end_date']
concept = pd.read_csv('trac_ 3772_schizophrenia_concept.csv', delimiter='|', date_parser=parse_dates, low_memory=False)

concept_sub = concept[['concept_id', 'concept_name']]

# lab_concept1 = pd.merge(file1, concept_sub, left_on='measurement_concept_id',right_on='concept_id', how='left')
lab_concept = pd.merge(file, concept_sub, left_on='measurement_concept_id',right_on='concept_id', how='left')
# lab_concept1 = lab_concept1[lab_concept1.measurement_concept_id != 0]
lab_concept = lab_concept[lab_concept.measurement_concept_id != 0]


to_take = ['measurement_id','value_as_number','unit_concept_id','unit_source_value']
def to_cover(org, new):
    tt = pd.merge(org[['measurement_id','value_as_number','unit_concept_id','unit_source_value']], new, on='measurement_id', how='left')
    org.loc[~tt.value_as_number_y.isnull().values,'value_as_number']=tt.loc[~tt.value_as_number_y.isnull(),'value_as_number_y'].tolist()
    org.loc[~tt.value_as_number_y.isnull().values,'unit_concept_id']=tt.loc[~tt.value_as_number_y.isnull(),'unit_concept_id_y'].tolist()
    org.loc[~tt.value_as_number_y.isnull().values,'unit_source_value']=tt.loc[~tt.value_as_number_y.isnull(),'unit_source_value_y'].tolist()
    return org
# Body temperature
tmp=lab_concept.query('concept_name=="Body temperature"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8653.0')
concept_sub.query('concept_id==9289.0')
tmp['unit_source_value'] = 'degree Celsius'
tmp['unit_concept_id'] = 8653.0
tmp.loc[(tmp.value_as_number<122) & (tmp.value_as_number>86),'value_as_number'] = \
    (tmp.loc[(tmp.value_as_number<122) & (tmp.value_as_number>86),'value_as_number']-32)*5/9
wrong_id = tmp.loc[~((tmp.value_as_number<50) & (tmp.value_as_number>30)),'measurement_id']
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
tp1 = tmp.copy()
w_id = wrong_id.copy()

# Respiratory rate
tmp=lab_concept.query('concept_name=="Respiratory rate"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
wrong_id = tmp.loc[tmp.value_as_number.isnull(),'measurement_id']
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
w_id = w_id.append(wrong_id)

# BP systolic
tmp=lab_concept.query('concept_name=="BP systolic"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8876')
tmp['unit_source_value'] = 'millimeter mercury column'
wrong_id = tmp.loc[tmp.value_as_number.isnull(),'measurement_id']
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# BP diastolic
tmp=lab_concept.query('concept_name=="BP diastolic"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8876')
tmp['unit_source_value'] = 'millimeter mercury column'
wrong_id = tmp.loc[tmp.value_as_number.isnull(),'measurement_id']
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Glucose [Mass/volume] in Blood | Glucose lab
tmp=lab_concept.query('concept_name=="Glucose [Mass/volume] in Blood" | concept_name=="Glucose lab"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8840.0')
tmp['unit_source_value'] = 'mg/dL'
tmp['unit_concept_id'] = 8840.0
wrong_id = tmp.loc[tmp.value_as_number.isnull(),'measurement_id']
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# eGFR with normals for black | eGFR with normals for non-black
tmp=lab_concept.query('concept_name=="eGFR with normals for black" | concept_name=="eGFR with normals for non-black"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
tmp['unit_source_value'] = 'mL/min/1.73m2'
wrong_id = tmp.loc[tmp.value_as_number.isnull(),'measurement_id']
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Body height Measured
tmp=lab_concept.query('concept_name=="Body height Measured"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8582.0')
concept_sub.query('concept_id==9327.0')
tmp['unit_source_value'] = 'centimeter'
tmp['unit_concept_id'] = 8582.0
tmp.loc[(tmp.value_as_number<80) & (tmp.value_as_number>25),'value_as_number'] = \
    tmp.loc[(tmp.value_as_number<80) & (tmp.value_as_number>25),'value_as_number']*2.54
wrong_id = tmp.loc[~((tmp.value_as_number<203) & (tmp.value_as_number>63.5)),'measurement_id']
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Body weight
tmp=lab_concept.query('concept_name=="Body weight"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==9529.0')
concept_sub.query('concept_id==8739.0')
tmp.loc[tmp.unit_concept_id==8739.0 ,'value_as_number'] = \
    tmp.loc[tmp.unit_concept_id==8739.0 ,'value_as_number']/2.205
tmp.loc[tmp.unit_source_value=='pound' ,'value_as_number'] = \
    tmp.loc[tmp.unit_source_value=='pound' ,'value_as_number']/2.205
tmp.loc[tmp.unit_source_value=='lb' ,'value_as_number'] = \
    tmp.loc[tmp.unit_source_value=='lb' ,'value_as_number']/2.205
wrong_id = tmp.loc[(tmp.value_as_number.isnull())|((tmp.unit_concept_id.isnull()) & tmp.unit_source_value.isnull()),'measurement_id']
tmp['unit_source_value'] = 'kilogram'
tmp['unit_concept_id'] = 9529.0
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Body mass index
tmp=lab_concept.query('concept_name=="Body mass index"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==9531.0')
wrong_id = tmp.loc[(tmp.value_as_number>70) | tmp.value_as_number.isnull(),'measurement_id']
tmp['unit_source_value'] = 'kilogram per square meter'
tmp['unit_concept_id'] = 9531.0
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Cholesterol [Mass/volume] in Serum or Plasma
tmp=lab_concept.query('concept_name=="Cholesterol [Mass/volume] in Serum or Plasma"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8840.0')
tmp['unit_source_value'] = 'mg/dL'
tmp['unit_concept_id'] = 8840.0
wrong_id = tmp.loc[tmp.value_as_number.isnull(),'measurement_id']
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Cholesterol in HDL [Mass/volume] in Serum or Plasma
tmp=lab_concept.query('concept_name=="Cholesterol in HDL [Mass/volume] in Serum or Plasma"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8840.0')
tmp['unit_source_value'] = 'mg/dL'
tmp['unit_concept_id'] = 8840.0
wrong_id = tmp.loc[tmp.value_as_number.isnull(),'measurement_id']
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Cholesterol in LDL [Mass/volume] in Serum or Plasma
tmp=lab_concept.query('concept_name=="Cholesterol in LDL [Mass/volume] in Serum or Plasma"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8840.0')
wrong_id = tmp.loc[tmp.value_as_number.isnull(),'measurement_id']
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(w_id)]
lab_concept = to_cover(lab_concept, tp1)