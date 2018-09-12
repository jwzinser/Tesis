import pandas as pn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import Counter
import math
import redis
import multiprocessing


def expo_weights(nclasses):
    weights = list()
    curr_weight = 1.
    for i in range(nclasses):
        curr_weight /= 2.
        weights.append(curr_weight)
    return weights_to_probabilities(weights)


def weights_to_probabilities(weights_vector, sum_to=1.):
    if sum(weights_vector) > 0:
        return np.array([sum_to * float(i) / sum(weights_vector) for i in weights_vector])
    else:
        return weights_vector


def operator_model(original_list, privacy=3, include_real=True, uniform=True, real_prob=.2, maybe=False):
    """
    :param original_list:
    :param privacy:
    :param include_real:
    :param uniform:
    :param real_prob: if uniform is false and include_real true, the real value will be given this probability
    :param maybe: if the maybe is true, include real is ignored
    :return:
    """
    # gets the real frequencies and calculates the changes for the new_value for each possible case
    counts = Counter(original_list)
    total = sum(counts.values())
    class_length = len(counts)
    privacy = min(privacy, class_length)
    if (privacy - include_real) >= class_length:
        include_real = True
    privacy_fraction = 1. / privacy
    key_to_order = dict(zip(sorted(counts.keys()), range(class_length)))
    order_exception = dict()
    order_weights = dict()
    ordered_weights = [float(counts[key]) / total for key in sorted(counts.keys())]

    for key in range(class_length):
        all_non_entry = list(range(class_length))
        all_non_entry.pop(key)
        all_non_entry_ordered_weights = [ordered_weights[i] for i in all_non_entry]
        order_exception[key] = all_non_entry
        order_weights[key] = all_non_entry_ordered_weights

    negative_list = list()

    # for entry in original_list:
    def entry_sanitization(entry, real_prob):
        entry_vector = np.zeros(class_length)
        weights = [1. / (class_length - 1)] * (class_length - 1) if uniform else order_weights[key_to_order[entry]]
        weights = weights_to_probabilities(weights)
        non_real_weights = np.random.choice(order_exception[key_to_order[entry]], privacy - include_real, False,
                                            p=weights)
        entry_vector[non_real_weights] = privacy_fraction if uniform else [ordered_weights[i] for i in non_real_weights]
        real_prob = ordered_weights[key_to_order[entry]] if real_prob is None else real_prob
        real_value = (privacy_fraction if uniform else real_prob) if include_real else 0
        entry_vector = weights_to_probabilities(entry_vector, 1 - real_value)
        if not maybe:
            entry_vector[key_to_order[entry]] = real_value
        entry_vector = weights_to_probabilities(entry_vector)
        return entry_vector

    # negative_list.append(entry_vector)
    negative_list = [entry_sanitization(i, real_prob) for i in original_list]

    result_dict = dict()
    for idx, field in enumerate(sorted(counts.keys())):
        result_dict[field] = [i[idx] for i in negative_list]

    return result_dict


column_size=1000
nsim_case = 10
cases = list()
#for nclasses in range(1, 30)[::1]:
for nclasses in ["NA"]:
    #for true_prob in [None, .2, .4, .6, .8]:
    for true_prob in [None]:
        # define the privacy level as a percentage of the number of classes, since it will be variable for each column
        for pr in range(1,11): # si lo llevamos hasta 16 cubrimos de forma correcta otro par de columnas
            #for class_dist in ['uniform','exponential']:
            for nsim in range(nsim_case):
                cases += [[pr, True, False, true_prob, False],
                          [pr, False, False, true_prob, False],
                          [pr, False, True, true_prob, False],
                          [pr, True, True, true_prob, False],
                          [pr, False, False, true_prob, True],
                          [pr, False, True, true_prob, True]]

processed_cases = list()
"""
for case in cases:
    case_name = str(case[0])+("m" if case[4] else "t" if case[1] else "f") +\
                ("t" if case[2] else "f")+(str(case[3]) if (case[1] and not case[2]) else "")
    if case_name not in processed_cases:
        data = pn.read_csv("../data/census/census_level_0.csv")

        data_cols = data.columns
        cat_columns = [u'workclass', u'education', u'marital-status', u'occupation',
                   u'race', u'sex', u'native-country']

        oh = preprocessing.OneHotEncoder()
        le = preprocessing.LabelEncoder()
        all_columns = ["age"]
        case2 = case
        for col in cat_columns:
            rel_privacy = math.ceil(float(case[0])/10*len(data[col].unique()))
            case[0] = rel_privacy
            field_dict = operator_model(data[col], *case2)
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
        data.to_csv("../data/census/maybe/negative_census_"+case_name+".csv")
        print(case_name)
        processed_cases.append(case_name)
"""

processed_cases = list()
reco_df = pn.DataFrame(columns=["case", "column", "class", "CIS", "NIS"])
reco_list = list()
for case in cases:
    case_name = str(case[0])+("t" if case[1] else "f")+("t" if case[2] else "f")+(str(case[3]) if (case[1] and
                                                                                  not case[2]) else "")
    case_name = str(case[0])+("m" if case[4] else "t" if case[1] else "f") +\
                ("t" if case[2] else "f")+(str(case[3]) if (case[1] and not case[2]) else "")
    if case_name not in processed_cases:
        data = pn.read_csv("../data/census/census_level_0.csv")

        data_cols = data.columns
        cat_columns = [u'workclass', u'education', u'marital-status', u'occupation',
                   u'race', u'sex', u'native-country']

        oh = preprocessing.OneHotEncoder()
        le = preprocessing.LabelEncoder()
        all_columns = ["age"]
        for col in cat_columns:
            rel_privacy = math.ceil(float(case[0])/10*len(data[col].unique()))
            case[0] = rel_privacy
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
reco_df.to_csv("../data/census/negative_census_reconstruction_rmse_df_maybe.csv")


# line plots of supervised rmse by privacy level and casse
import re
import math
reco_df["privacy"] = reco_df["case"].map(lambda x: re.findall("\d+", x)[0])
reco_df["real"] = reco_df["case"].map(lambda x: re.findall("[^\d]",x)[0])
reco_df["uniform"] = reco_df["case"].map(lambda x: int(re.findall("[^\d]",x)[1] == "t"))
reco_df["rmse"] = (reco_df["CIS"].map(float) - reco_df["NIS"]).map(lambda x: x*x)

reco_gb = reco_df.groupby(["privacy", "real", "uniform"])["rmse"].agg(lambda x: math.sqrt(sum(x))).reset_index()


def sup_rmse_plot(data, gb_param, reals, uniforms):
    fig, ax = plt.subplots()
    labels = []
    for real in reals:
        for uniform in uniforms:
            data_f = data.query("real=='{real}'".format(real=real)) if real is not None else data
            data_f = data_f.query("uniform=={uniform}".format(uniform=uniform)) if uniform is not None else data_f
            data_f[gb_param] = data_f[gb_param].map(int)
            gb = data_f.sort_values(by="privacy", ascending=True)
            print(gb)
            x = gb[gb_param]
            y = gb["rmse"]
            ax.plot(x, y)
            lines, _ = ax.get_legend_handles_labels()
            tt = "real=" + str(real) + ", uniform=" + str(uniform)

            labels.append(tt)

    ax.legend(lines, labels, loc='best')
    tt = "Supervised RMSE real=" + str(real) + ", uniform="+str(uniform)
    tt_save = "supervised_privacy_gb_" + "".join(reals) + "_" + "".join([str(x) for x in uniforms])
    ax.set_title(tt)
    ax.set_xlabel(gb_param)
    plt.savefig("plots/" + tt_save+".png")
    plt.show()

sup_rmse_plot(reco_gb, "privacy", ["t", "m", "f"], [0, 1])
sup_rmse_plot(reco_gb,  "privacy", ["t", "m", "f"], [0])
sup_rmse_plot(reco_gb, "privacy", ["t", "m", "f"], [1])

sup_rmse_plot(reco_gb, "privacy", ["t"], [0, 1])
sup_rmse_plot(reco_gb, "privacy", ["m"], [0, 1])
sup_rmse_plot(reco_gb,  "privacy", ["f"], [0, 1])