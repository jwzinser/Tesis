from sanitization_tools import *
import math
income_dataset_path = "census_level_0.csv"
data = pn.read_csv(income_dataset_path)

sdb = pn.read_csv("../data/hist_python/sanitized_census_2ttt.csv")

lb = preprocessing.LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
lb = preprocessing.LabelBinarizer()


data_cols = data.columns
cat_columns = [u'workclass', u'education', u'marital-status', u'occupation',
           u'race', u'sex', u'native-country']

rdb = pn.get_dummies(data)
