import pandas as pn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing, metrics, linear_model, metrics, svm, naive_bayes, tree
from collections import Counter


def expo_weights(nclasses):
    weights = list()
    curr_weight = 1.
    for i in range(nclasses):
        curr_weight /= 2.
        weights.append(curr_weight)
    return weights_to_probabilities(weights)


def weights_to_probabilities(weights_vector, sum_to=1.):
    if sum(weights_vector) > 0:
        return np.array([sum_to * float(i) / sum(weights_vector) for
                         i in weights_vector])
    else:
        return weights_vector


def entry_sanitization(entry, real_prob, class_length,
                       maybe, uniform, uniform2, include_real,
                       privacy, order_weights, key_to_order,
                       order_exception, ordered_weights):
    """
    Sanitizes a single record

    :param entry:
    :param real_prob:
    :param class_length:
    :param maybe:
    :param uniform:
    :param uniform2:
    :param include_real:
    :param privacy:
    :param order_weights:
    :param key_to_order:
    :param order_exception:
    :param ordered_weights:
    :return:
    """
    # initializes the entry_vector, same size as the
    # total number of classes
    privacy_fraction = 1. / privacy
    entry_vector = np.zeros(class_length)
    if not maybe:
        # gets the weights of each class excluding the real
        # value class
        weights = [1. / (class_length - 1)] * \
                  (class_length - 1) if uniform else \
            order_weights[key_to_order[entry]]

        # makes the weights sum one
        weights = weights_to_probabilities(weights)

        # get sample of the indexes that will have a
        # non zero weight (real not considered)
        non_real_weights = np.random.choice(
            order_exception[key_to_order[entry]],
            privacy - include_real, False, p=weights)

        # save the corresponding weights into their
        # corresponding index for all the sampled
        # indexes in the previous step
        entry_vector[non_real_weights] = privacy_fraction if \
            uniform2 else [ordered_weights[i] for
                           i in non_real_weights]

        # if real prob is None set to the proportional weight
        real_prob = ordered_weights[key_to_order[entry]] if \
            real_prob is None else real_prob

        # gets the weight that will be assigned to
        # the real value
        real_value = (privacy_fraction if uniform2 else
                      real_prob) if include_real else 0
        entry_vector = weights_to_probabilities(
            entry_vector, 1 - real_value)

        entry_vector[key_to_order[entry]] = real_value
        entry_vector = weights_to_probabilities(entry_vector)
    else:
        # gets the weights of each class excluding the
        # real value class
        weights = [1. / class_length] * class_length if \
            uniform else ordered_weights

        # get sample of the indexes that will have a non
        # zero weight
        selected_weights = np.random.choice(
            list(range(class_length)), privacy,
            False, p=weights)

        # save the corresponding weights into their
        # corresponding index
        # for all the sampled indexes in the previous step
        entry_vector[selected_weights] = privacy_fraction if \
            uniform2 else [ordered_weights[i]
                           for i in selected_weights]
        entry_vector = weights_to_probabilities(entry_vector)

    return entry_vector


def operator_model(original_list, privacy=3, include_real=True,
                   uniform=True, uniform2=True, real_prob=None,
                   maybe=False):
    """
    :param original_list:
    :param privacy:
    :param include_real:
    :param uniform:
    :param uniform2:
    :param real_prob: if uniform is false and include_real true,
    the real value will be given this probability
    :param maybe: if the maybe is true, include real is ignored
    :return:
    """
    # gets the real frequencies and calculates the changes
    # for the new_value for each possible case
    counts = Counter(original_list)
    total = sum(counts.values())
    class_length = len(counts)
    privacy = min(privacy, class_length)
    if (privacy - include_real) >= class_length:
        include_real = True

    # correspondence of the ordered index of each of the classes
    key_to_order = dict(zip(sorted(counts.keys()),
                            range(class_length)))
    order_exception = dict()
    order_weights = dict()
    ordered_weights = [float(counts[key]) / total for
                       key in sorted(counts.keys())]

    # gets two dictionaries, order exception and ordered weights
    for key in range(class_length):
        all_non_entry = list(range(class_length))
        all_non_entry.pop(key)
        all_non_entry_ordered_weights = [ordered_weights[i] for
                                         i in all_non_entry]

        # order exception has a list off all the indexes
        # other than the one of the real value, after
        # being ordered
        order_exception[key] = all_non_entry
        # order weights contains the equivalent to order
        # exception but with the corresponding weights instead
        order_weights[key] = all_non_entry_ordered_weights

    negative_list = [entry_sanitization(
        i, real_prob, class_length, maybe, uniform, uniform2,
        include_real, privacy, order_weights, key_to_order,
        order_exception, ordered_weights)
        for i in original_list]

    result_dict = dict()
    for idx, field in enumerate(sorted(counts.keys())):
        result_dict[field] = [i[idx] for i in negative_list]

    return result_dict


def get_auc_score_of_model(df, model):
    """
    returns both the prediction error and the auc of the given model applied to the dataset
    
    param df: data with the `y` value placed in the last column and corresponds to a binary
    class
    param model: classification model 
    """
    X = df.loc[:,df.columns[:-1]]
    y = (df.loc[:,df.columns[-1]] != " <=50K").astype(int)
    msk = np.random.rand(len(y)) < 0.8
    Xtrain = X[msk]
    ytrain = y[msk]
    Xtest = X[~msk]
    ytest = y[~msk]
    model.fit(Xtrain, ytrain)
    predicted_score = model.predict(Xtest)
    predicted = (predicted_score >= .5).astype(int)
    prediction_error = 1 - sum((ytest == predicted).astype(int))/float(len(ytest))
    roc_auc = metrics.roc_auc_score(ytest, predicted_score)
    roc_curve = metrics.roc_curve(ytest, predicted_score)
    return prediction_error, roc_auc, roc_curve


def get_label_name(param_dict):
    """
    Gets the name of the label from the parameters being used.

    """
    param_list = [("real", " R={val}"), ("uniform", " U={val}"), ("uniform2"," U2={val}"), ("model"," M={val}")]
    label_name = ""
    for param, valpar in param_list:
        if param_dict.get(param) is not None:
            label_name += valpar.format(val=str(param_dict.get(param)))
    return label_name  


def get_single_filter_df(df, k, v):
    """
    Applies a filter to a pandas DataFrame, which might me a multiple condition
    """
    if v:
        v = [v] if not isinstance(v, list) else v
        if k in df.columns:
            if np.issubdtype(df[k].dtype , np.number):
                cond = " | ".join(["{k} == {val}".format(k=k, val=v0) for v0 in v])
                df = df.query(cond)
            else:
                cond = " | ".join(["{k} == '{val}'".format(k=k, val=v0) for v0 in v])
                df = df.query(cond)
    return df


def get_base_filtered_df(df, base_filter=None):
    """
    filters a database with its corresponding filters
    """
    if isinstance(base_filter, dict):
        for k, v in base_filter.items():
            df = get_single_filter_df(df, k, v)
        
    return df


def plot_intervals(df, gb_param, base_filter, savefig=False,  title=None, save_name=None):
    """
    Returns a line plot with quantile intervals of the RMSE of different levels of either privacy or number of classes.
    Works only for the non-supervised datasets since there are multiples simulations for provacy levels and numberr of classes.

    """
    pt = base_filter.get("privacy")
    if pt is not None:
        base_filter.pop("privacy")
        df = df.query("privacy < {pt}".format(pt=pt))
    df = get_base_filtered_df(df, base_filter)
    gb = df.groupby([gb_param])["rmse"].quantile([.1,.25,.5,.75,0.9]).reset_index()
    fig, ax = plt.subplots()
    labels = []
    x = gb[gb_param].unique()
    y1 = gb.query("level_1 == 0.25")["rmse"]
    y2 = gb.query("level_1 == 0.50")["rmse"]
    y3 = gb.query("level_1 == 0.75")["rmse"]
    lines, _ = ax.get_legend_handles_labels()
    ax.fill_between(x, y1, y3, color='grey', alpha='0.5')
    ax.plot(x,y2)
    ax.legend(lines, labels, loc='best')
    ax.set_title(title)
    ax.set_xlabel(gb_param)
    ax.set_ylabel("RMSE")
    if savefig:
        plt.savefig("/home/juanzinser/Documents/plots" + save_name + ".png")
    plt.show()


def rocs_by_case(df, base_filter, lines_cases, savefig=False, title=None, save_name=None):
    """
    Gets the ROC plots for all privacy levels and for the sliced frame with the desired parameters 
    """

    fig, ax = plt.subplots()
    labels = []

    df = get_base_filtered_df(df, base_filter)

    for k, v in lines_cases.items():
        v = [v] if not isinstance(v, list) else v
        for v0 in v:
            dfc = get_single_filter_df(df, k, v0)

            roc_x = dfc.loc[:, "roc_x"].map(lambda x: eval(x)).values 
            roc_y = dfc.loc[:, "roc_y"].map(lambda x: eval(x)).values 
            xs = []
            ys = []
            for x, y in zip(roc_x, roc_y):
                xs.extend(x)
                ys.extend(y)

            df_roc = pn.DataFrame({"fpr": xs, "tpr": ys}).sort_values(by="fpr", ascending=True)
            df_roc.loc[:, "fpr_dis"] = df_roc["fpr"].map(lambda x: round(x,2))
            gb = df_roc.groupby("fpr_dis")["tpr"].mean().reset_index().sort_values(by="fpr_dis", ascending=True)

            x = gb.fpr_dis
            y = gb.tpr.rolling(window=3, center=False).mean() if len(gb) > 10 else gb.tpr
            ax.plot(x, y)
            lines, _ = ax.get_legend_handles_labels()
            labels.append(v0)

    ax.legend(lines, labels, loc='best')
    tt = "Income DB " + title
    ax.set_title(tt)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    if savefig:
        plt.savefig("/home/juanzinser/Documents/plots/" + save_name + ".png")
    plt.show()  
        

def rmse_auc_plot_no_intervals(df, gb_param, yaxis, reals, uniforms, uniforms2, models, combined_cond=None, savefig=False, title=None, save_name=None):
    """
    Gets the supervised RMSE plot, since non supervised also has RMSE, pending is to check what is the difference between 
    this function and the plot_params, which plots the RMSE for the non supervised case. Both rmse and auc are merged now
    the only pending situation is the plot_params separation from this function.
    The difference is that plot params has quantile intervals and this one doesn't. This one can be used for rmse in the 
    supervised case in case quantiles are not needed. The AUC with confidence intervals where can it go? there is not enough 
    simulations.

    yaxis: is either rmse or auc, both lower cased
    """
    df_cols_gb = list(set(["privacy", "real", "uniform", "uniform2", "model"]).intersection(set(df.columns)))
    #df = df.groupby(df_cols_gb)[yaxis].agg(lambda x: np.mean(x)).reset_index()

    fig, ax = plt.subplots()
    labels = []
    for real in reals:
        for uniform in uniforms:
            for uniform2 in uniforms2:
                for model in models:
                    if combined_cond is not None and isinstance(combined_cond, dict): 
                        for tp, vl in combined_cond.items():
                            param_dict = {"real":real, "uniform":uniform, "uniform2": uniform2, "model":model}
                            for col, val in zip([tp]*len(vl), vl):
                                dfc = df
                                print(col)
                                print(val)
                                for i, j in enumerate(col):
                                    dfc = get_single_filter_df(dfc, j, val[i])
                                    param_dict[j] = val[i]
                                dfc = get_single_filter_df(dfc, "real", real)
                                dfc = get_single_filter_df(dfc, "uniform", uniform)
                                dfc = get_single_filter_df(dfc, "uniform2", uniform2)
                                dfc = get_single_filter_df(dfc, "model", model)
                                
                                dfc.loc[:, gb_param] = dfc[gb_param].map(int)
                                gb = dfc.sort_values(by="privacy", ascending=True)
                                gb = gb.groupby(gb_param)[yaxis].agg(lambda x: np.mean(x)).reset_index()
                                x = gb[gb_param]
                                y = gb[yaxis]
                                ax.plot(x, y)
                                lines, _ = ax.get_legend_handles_labels()
                                tt = get_label_name(param_dict)
                                print(tt)
                                print(len(gb))
                                print(param_dict)
                                labels.append(tt)
                    else:
                        param_dict = {"real":real, "uniform":uniform, "uniform2": uniform2, "model":model}
                        dfc = get_single_filter_df(df, "real", real)
                        dfc = get_single_filter_df(dfc, "uniform", uniform)
                        dfc = get_single_filter_df(dfc, "uniform2", uniform2)
                        dfc = get_single_filter_df(dfc, "model", model)

                        dfc.loc[:, gb_param] = dfc[gb_param].map(int)
                        gb = dfc.sort_values(by="privacy", ascending=True)
                        gb = gb.groupby(gb_param)[yaxis].agg(lambda x: np.mean(x)).reset_index()
                        x = gb[gb_param]
                        y = gb[yaxis]
                        ax.plot(x, y)
                        lines, _ = ax.get_legend_handles_labels()
                        tt = get_label_name(param_dict)

                        labels.append(tt)

    ax.legend(lines, labels, loc='best')
    ax.set_title(title)
    ax.set_xlabel(gb_param)
    ax.set_ylabel(yaxis)
    if savefig:
        plt.savefig("/home/juanzinser/Documents/plots" + save_name + ".png")
    plt.show()