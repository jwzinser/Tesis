from operator_model import operator_model
import pandas as pn
from sklearn import preprocessing

cases = list()
for pr in range(1,10):
    for true_prob in [None, .2, .4, .6, .8]:
        cases += [[pr, True, False, true_prob], [pr, False, False, true_prob],
                 [pr, False, True, true_prob], [pr, True, True, true_prob]]

processed_cases = list()
for case in cases:
    case_name = str(case[0])+("t" if case[1] else "f")+("t" if case[2] else "f")+(str(case[3]) if (case[1] and
                                                                                  not case[2]) else "")
    if case_name not in processed_cases:
        data = pn.read_csv("../data/census/census_level_0.csv")

        data_cols = data.columns
        cat_columns = [u'workclass', u'education', u'marital-status', u'occupation',
                   u'race', u'sex', u'native-country']

        oh = preprocessing.OneHotEncoder()
        le = preprocessing.LabelEncoder()
        all_columns = ["age"]
        for col in cat_columns:
            field_dict = operator_model(data[col], *case)
            data.drop(col, axis=1, inplace=True)
            for field in field_dict.keys():
                data.loc[:, field] = field_dict[field]
                if field not in all_columns:
                    all_columns += [field]

        std_cols = ["age"]

        std_scaler = preprocessing.StandardScaler()
        for col in std_cols:
            data.loc[:, col] = std_scaler.fit_transform(data[col].reshape(-1,1))

        data = data[all_columns + ["salary-class"]]
        data.to_csv("../data/census/negative_census_"+case_name+".csv")
        print(case_name)
        processed_cases.append(case_name)


