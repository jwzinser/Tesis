import pandas as pn
import numpy as np
from collections import Counter


def weights_to_probabilities(weights_vector, sum_to=1.):
    return np.array([sum_to*float(i)/sum(weights_vector) for i in weights_vector])


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
    privacy_fraction = 1./privacy
    key_to_order = dict(zip(sorted(counts.keys()), range(class_length)))
    order_exception = dict()
    order_weights = dict()
    ordered_weights = [float(counts[key])/total for key in sorted(counts.keys())]

    for key in range(class_length):
        all_non_entry = list(range(class_length))
        all_non_entry.pop(key)
        all_non_entry_ordered_weights = [ordered_weights[i] for i in all_non_entry]
        order_exception[key] = all_non_entry
        order_weights[key] = all_non_entry_ordered_weights

    negative_list = list()
    for entry in original_list:
        entry_vector = np.zeros(class_length)
        weights = [1. / (class_length - 1)] * (class_length - 1) if uniform else order_weights[key_to_order[entry]]
        weights = weights_to_probabilities(weights)
        non_real_weights = np.random.choice(order_exception[key_to_order[entry]], privacy-include_real, False, p=weights)
        entry_vector[non_real_weights] = privacy_fraction if uniform else [ordered_weights[i] for i in non_real_weights]
        real_prob = ordered_weights[key_to_order[entry]] if real_prob is None else real_prob
        real_value = (privacy_fraction if uniform else real_prob) if include_real else 0
        entry_vector = weights_to_probabilities(entry_vector, 1-real_value)
        entry_vector[key_to_order[entry]] = real_value

        negative_list.append(entry_vector)

    result_dict = dict()
    for idx, field in enumerate(sorted(counts.keys())):
        result_dict[field] = [i[idx] for i in negative_list]

    return result_dict


def __main__():
    data = pn.read_csv("../data/census/census_db.csv")
    education = data.education

    cases = [[education, 5, True, False, None]]

    # probar los diferentes casos
    for case in cases:
        sanitized_dict = operator_model(*case)
        sanitized_df = pn.DataFrame(sanitized_dict)
        # checar que sumen uno en cada renglon
        check_df = sanitized_df.sum(1).map(lambda x: round(x,2) == 1.)
        print(sum(check_df)/float(len(check_df)))

