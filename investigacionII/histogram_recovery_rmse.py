from operator_model import operator_model
import pandas as pn
from sklearn import preprocessing
from collections import Counter

cases = list()
for pr in range(1,10):
    for true_prob in [None, .2, .4, .6, .8]:
        cases += [[pr, True, False, true_prob], [pr, False, False, true_prob],
                 [pr, False, True, true_prob], [pr, True, True, true_prob]]

processed_cases = list()
reco_df = pn.DataFrame(columns=["case", "column", "class", "CIS", "NIS"])
reco_list = list()
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
            cis = pn.DataFrame.from_dict(Counter(data[col]), "index").reset_index()
            cis.columns = ["class", "CIS"]
            field_dict = operator_model(data[col], *case)
            real_col = data[col]
            data.drop(col, axis=1, inplace=True)
            nis_rmse = dict()
            for field in field_dict.keys():
                data.loc[:, field] = field_dict[field]
                rmse = ((real_col == field) - field_dict[field]).map(lambda x: x*x).sum()
                nis_rmse[field] = rmse
                if field not in all_columns:
                    all_columns += [field]

            nis = pn.DataFrame.from_dict(data.loc[:, field_dict.keys()].sum().to_dict(), "index").reset_index()
            nis = pn.DataFrame.from_dict(nis_rmse, "index").reset_index()
            nis.columns = ["class", "NIS"]

            tmp_df = cis.merge(nis, how="left")
            tmp_df["column"] = col
            tmp_df["case"] = case_name
            reco_list.append(tmp_df)
        std_cols = ["age"]

        std_scaler = preprocessing.StandardScaler()
        for col in std_cols:
            data.loc[:, col] = std_scaler.fit_transform(data[col].reshape(-1,1))

        data = data[all_columns + ["salary-class"]]
        print(case_name)
        processed_cases.append(case_name)


reco_df = pn.concat(reco_list)
reco_df.to_csv("../data/census/negative_census_reconstruction_rmse_df.csv")

