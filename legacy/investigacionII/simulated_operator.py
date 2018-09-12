import pandas as pn
import numpy as np
from collections import Counter


def anonymization_operator(original_list, privacy=3, prob_real=0.2):
    """
    :param original_list: original list to anonymize
    :param prob_real: the assigned probability for the real value of the entry
    :return: anonymized list, Probability matrix and estimated lambda
    """
    # gets the real frequencies and calculates the changes for the new_value for each possible case
    counts = Counter(original_list)
    total = sum(counts.values())
    probability_mass = {k: float(v)/total for k,v in counts.items()}
    probability_mass_by_entry = dict()
    for key, probability in probability_mass.items():
        prob_scaler = (1-prob_real)/(1-probability)
        probability_mass_by_entry[key] = {k: float(v)*prob_scaler if k!=key else prob_real
                                          for k, v in probability_mass.items()}
        #probability_mass_by_entry[key] = {k: float(1)/(len(probability_mass)-1) if k!=key else prob_real
        #                                 for k, v in probability_mass.items()}
    # gets the transformed list
    anonymized_list = list()
    for entry in original_list:
        values = probability_mass_by_entry[entry].keys()
        weights = probability_mass_by_entry[entry].values()
        generalized_entry = np.random.choice(values, 1, weights)
        anonymized_list += list(generalized_entry)

    # gets the probability matrix, where entries are ordered alphabetically
    P = np.matrix([[probability_mass_by_entry[key][k] for k in sorted(probability_mass_by_entry[key].keys())]
                  for key in sorted(probability_mass_by_entry.keys())])

    # lambda, frequencies of transformed vector
    counts_estimator = Counter(anonymized_list)
    total_estimator = sum(counts_estimator.values())
    lambda_estimator = {k: float(v)/total_estimator for k,v in counts_estimator.items()}

    return anonymized_list, P, lambda_estimator


def deidentification_operator(P, lambda_estimator):
    """
    :param P: Probability matrix
    :param lambda_estimator: dictionary with estimator for every lambda
    :return:
    """
    # Singular Value descomposition for inverse
    U, Sigma, V_t = np.linalg.svd(P)
    P_inverse = np.matmul(V_t.transpose(), np.matmul(np.diag(1/Sigma), U.transpose()))
    lambda_vector = np.array([lambda_estimator[k] for k in sorted(lambda_estimator.keys()) ])
    order_vector = [k for k in sorted(lambda_estimator.keys())]
    #pi_vector = np.matmul(P_inverse, lambda_vector.transpose()).tolist()[0]
    pi_vector = np.matmul(np.linalg.inv(P), lambda_vector.transpose()).tolist()[0]
    # dictionary with the estimated real probabilities matched with their labels
    pi_estimated = dict(zip(list(order_vector), pi_vector))

    return pi_estimated




# estimadorWEEQ
# descomposicion


