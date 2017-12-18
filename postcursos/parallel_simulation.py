# from postcursos.operator_model import *
import pandas as pn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import Counter
import math


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


def operator_model(original_list, privacy=3, include_real=True, uniform=True, real_prob=.2):
    """
    :param original_list:
    :param privacy:
    :param include_real:
    :param uniform:
    :param real_prob: if uniform is false and include_real true, the real value will be given this probability
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
        entry_vector[key_to_order[entry]] = real_value
        entry_vector = weights_to_probabilities(entry_vector)
        return entry_vector

    # negative_list.append(entry_vector)
    negative_list = [entry_sanitization(i, real_prob)  for i in  original_list]

    result_dict = dict()
    for idx, field in enumerate(sorted(counts.keys())):
        result_dict[field] = [i[idx] for i in negative_list]

    return result_dict





column_size=1000
cases = list()
for nclasses in range(10, 200)[::1]:
    #for true_prob in [None, .2, .4, .6, .8]:
    for true_prob in [None]:
        for pr in range(1,nclasses):
            for class_dist in ['uniform','exponential']:
                cases += [[pr, nclasses, class_dist, True, False, true_prob],
                          [pr, nclasses, class_dist, False, False, true_prob],
                          [pr, nclasses, class_dist, False, True, true_prob],
                          [pr, nclasses, class_dist, True, True, true_prob]]

processed_cases = list()
reco_df = pn.DataFrame(columns=["case", "class", "CIS", "NIS"])
rmse_by_case = dict()
def process_case(case):
    case_name = str(case[0])+("t" if case[3] else "f")+ ("t" if case[4] else "f")+(str(case[5]) if (case[3] and not case[4]) else "")
    case_name += '_' + str(case[1]) + '_' + str(case[2])
    nclasses = case[1]
    class_dist = case[2]
    p = [1./nclasses]*nclasses if class_dist=='uniform' else expo_weights(nclasses)
    sim_data = np.random.choice(range(nclasses), column_size, p=p)
    #if case_name not in processed_cases:
    cis = pn.DataFrame.from_dict(Counter(sim_data), "index").reset_index()
    cis.columns = ["class", "CIS"]
    case2 = case
    case2.pop(1)
    case2.pop(1)
    field_dict = operator_model(sim_data, *case2)
    #print(field_dict)
    try:
        nis = pn.DataFrame.from_dict(field_dict).sum(axis=0).reset_index()
        nis.columns = ["class", "NIS"]
        tmp_df = cis.merge(nis, how="left")
        tmp_df['RMSE'] = (tmp_df['CIS'] - tmp_df['NIS']).map(lambda x: x*x)
        rmse_one = math.sqrt(sum(tmp_df['RMSE'].values))
        print('\t hihihihih \t' + case_name)
    except:
        print(case_name)
        rmse_one = None
    # rmse_by_case[case_name] = math.sqrt(sum(tmp_df['RMSE'].values))
    # print(case_name)
    #processed_cases.append(case_name)
    return (case_name, rmse_one)





import multiprocessing
def __main__():
    pool = multiprocessing.Pool()
    cases_results = pool.map(process_case, cases)