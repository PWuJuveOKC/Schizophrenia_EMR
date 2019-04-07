import pandas as pd
pd.options.mode.chained_assignment = None
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

dataPath = "./Data/trac_ 3772_schizophrenia/"
# dataPath = r"C:/Users/Peter Xu/Desktop/Yuanjia/schi/trac_ 3772_schizophrenia/"
prefix = 'trac_ 3772_schizophrenia_'
dataName = 'visit'

parse_dates = ['visit_start_date', 'visit_start_datetime', 'visit_end_date', 'visit_end_datetime']
file = pd.read_csv(dataPath + prefix + dataName + '.csv', delimiter='|', date_parser=parse_dates, low_memory=False)

# (1183223, 12)
print(file.shape)
print(file.dtypes)

# 17499
personId = file.person_id.unique()
# check missing portion
file.isnull().sum()

for column in file.columns:
    value_count = file[column].value_counts()
    print('--------------------------------')
    print(value_count.head())
    print('--------------------------------')


# In patient
file_in = file[file.visit_concept_id == 9201] # 57026
file_in_id = file_in.person_id.unique() # 13331
file_in['visit_start_datetime_parsed'] = pd.to_datetime(file_in['visit_start_datetime'])
file_in['visit_end_datetime_parsed'] = pd.to_datetime(file_in['visit_end_datetime']).values
file_in['hospitalization'] = file_in['visit_end_datetime_parsed'].subtract(file_in['visit_start_datetime_parsed'])
file_in['hospitalization_hours'] = file_in['hospitalization'].apply(lambda x: x.days * 24 + x.components.hours)
file_in = file_in[file_in.hospitalization_hours > 0]
file_in = file_in.groupby('person_id').apply(lambda x: x.sort_values(['visit_start_datetime_parsed']))
#13291
file_in.drop_duplicates(subset=['person_id'], keep='first', inplace=True)

# check hospitalization
print('95% percentile hospitalization (hours): ', np.percentile(file_in.hospitalization_hours,95))
file_in['hospitalization_hours'].describe()
file_in['hospitalization_hours'] = np.where(file_in['hospitalization_hours'] > 1000, 1000, file_in['hospitalization_hours'])

# visualization hospitalization
hosp = file_in['hospitalization_hours'].values
ax = sns.distplot(hosp, kde=True, rug=False)
ax.set(xlabel='Lengh of Hospitalization (Hours)')
plt.savefig('Results/Descriptive_figures/hosp_hist.pdf', dpi=600)

# create dataset with window (10 days)
file_in_window = file_in.copy()
window_size = 10
file_in_window['visit_start_date'] = pd.to_datetime(file_in_window['visit_start_date'])
file_in_window['start_date_min'] = file_in_window['visit_start_date'].apply(lambda x: x - pd.DateOffset(days = window_size))
file_in_window['visit_end_date'] = pd.to_datetime(file_in_window['visit_end_date'])
file_in_window['end_date_max'] = file_in_window['visit_end_date'].apply(lambda x: x + pd.DateOffset(days = window_size))

# file_in_window.to_csv(dataPath+'/visit/hospitalization_window_' + str(window_size) + '.csv', index=None)
file_in_window.to_csv('src/int_data/hospitalization_window_' + str(window_size) + '.csv', index=None)
