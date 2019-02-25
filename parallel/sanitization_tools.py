import pandas as pn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing, metrics, linear_model, metrics, svm, naive_bayes, tree
from collections import Counter
import sys
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, PercentFormatter
import math
from scipy.interpolate import  spline

figures_path = "/home/juanzinser/Documents/plots/" if sys.platform == "linux" \
    else "/Users/juanzinser/Documents/plots/"

y_tick_format = '%.2f'
y_tick_format_int = FuncFormatter(lambda x, pos: '{:,}'.format(x) )
x_tick_percent = PercentFormatter(xmax=10)

def meta_results(ds_name):
    ds_namee = "".join(ds_name)
    #supervised_results = pd.concat([pn.read_csv("model_scores_roc_census06_mbe.csv"), pd.read_csv("model_scores_roc_census611_mbe.csv")])
    supervised_results = pd.read_csv("model_scores_roc_"+ds_namee+".csv")
    # rocs_by_case(supervised_results, {},{"real":["t","f"]}, savefig=True, title="by IF REAL", save_name=ds_name[0]+"_roc_privacy_grouped_tmf",language="spanish")
    # rocs_by_case(supervised_results, {"real":"m"},{"privacy":[1,2,3,4,5,7,9]}, savefig=True, title="by IF REAL", save_name=ds_name+"_roc_privacy_grouped_tmf",language="spanish")

    # print(supervised_results[supervised_results.real=="m"].groupby("privacy")["auc"].agg(lambda x:np.average(x)))

    result_dict = {}
    result_dict["benchmark_auc"] = supervised_results[supervised_results.privacy==0].groupby("privacy")["auc"].agg(lambda x:np.average(x))[0]
    intersect = plot_intervals_smooth_info(supervised_results, "privacy","auc", {"uniform":[0], "uniform2":[0]}, 
                   {"real":["t", "f"]}, savefig=True, 
                   title="AUC Privacy Proportional", save_name=ds_name[0]+"_auc_real_proportional")
    intersection_pnt = [intersect[0][0][i]  for i in range(len(intersect[0][0])) if (0 < abs(intersect[0][1][i]-intersect[1][1][i]) < .004)][0]
    result_dict["proportional"] = intersection_pnt
    
    intersect = plot_intervals_smooth_info(supervised_results, "privacy","auc", {"uniform":[1], "uniform2":[1]}, 
                   {"real":["t", "f"]}, savefig=True, 
                   title="AUC Privacy Uniform", save_name=ds_name[0]+"_auc_real_uniform")
    intersection_pnt = [intersect[0][0][i]  for i in range(len(intersect[0][0])) if (abs(intersect[0][1][i]-intersect[1][1][i]) < .004)][0]
    result_dict["uniform"] = intersection_pnt
    return result_dict


def plot_intervals_smooth_info(df, gb_param, yaxis, base_filter, lines_cases, savefig=False,  title=None, save_name=None,
                   language="english"):
    """
    Returns a line plot with quantile intervals of the RMSE of different levels of either privacy or number of classes.
    Works only for the non-supervised datasets since there are multiples simulations for provacy levels and numberr of classes.
    
    """
    fig, ax = plt.subplots()
    pt = base_filter.get("privacy")
    if pt is not None:
        base_filter.pop("privacy")
        df = df.query("privacy < {pt}".format(pt=pt))
    df = get_base_filtered_df(df, base_filter)
    labels = []
    if "uniform" in df.columns:    
        df = df[df.uniform == df.uniform2]
    y_max = 0
    intersect_line = []
    if len(lines_cases)>0:
        for k, v in lines_cases.items():
            v = [v] if not isinstance(v, list) else v
            for v0 in v:
                dfc = get_single_filter_df(df, k, v0)
                gb = dfc.groupby([gb_param])[yaxis].quantile([.1,.25,.5,.75,0.9]).reset_index()
                x = gb[gb_param].unique()
                xnew = np.linspace(x.min(),x.max(),300) #300 represents number of points to make between T.min and T.max
                xneww = np.linspace(0,x.max(),300) #300 represents number of points to make between T.min and T.max
                y1 = spline(x, gb.query("level_1 == 0.25")[yaxis], xnew)
                y11 = spline(x, gb.query("level_1 == 0.25")[yaxis], xneww)
                y2 = spline(x, gb.query("level_1 == 0.50")[yaxis], xnew)
                y3 = spline(x, gb.query("level_1 == 0.75")[yaxis], xnew)
                intersect_line.append((xneww, y11))
                ax.fill_between(xnew, y1, y3, color='grey', alpha='0.5')
                ax.plot(xnew, y2)
                param_dict = {k: v0}
                tt = get_label_name(param_dict, True, language)
                labels.append(tt)
                lines, _ = ax.get_legend_handles_labels()
                y_max = max(y_max, max(y2))
    else:
        gb = df.groupby([gb_param])[yaxis].quantile([.1,.25,.5,.75,0.9]).reset_index()
        x = gb[gb_param].unique()
        xnew = np.linspace(x.min(),x.max(),300) #300 represents number of points to make between T.min and T.max
        y1 = spline(x, gb.query("level_1 == 0.25")[yaxis], xnew)
        y2 = spline(x, gb.query("level_1 == 0.50")[yaxis], xnew)
        y3 = spline(x, gb.query("level_1 == 0.75")[yaxis], xnew)
        ax.fill_between(xnew, y1, y3, color='grey', alpha='0.5')
        ax.plot(xnew,y2)
        tt = get_label_name(base_filter, True, language)
        labels.append(tt)
        lines, _ = ax.get_legend_handles_labels()
        y_max = max(y_max, max(y2))

    ax.legend(lines, labels, loc='best')
    ax.set_title(title)
    #ax.set_ylim([0, y_max*1.5])
    dict_use = english_dict if language == "english" else spanish_dict
    gb_param = dict_use.get(gb_param.lower(), gb_param)
    yaxis = dict_use.get(yaxis.lower(), yaxis)
    ax.set_xlabel(gb_param.upper())
    ax.set_ylabel(yaxis.upper())
    if yaxis=="auc":
        ax.yaxis.set_major_formatter(FormatStrFormatter(y_tick_format))
    else:
        ax.yaxis.set_major_formatter(y_tick_format_int)
    if gb_param == "privacy":
        #ax.xaxis.set_major_formatter(x_tick_percent)
        pass
    plt.tight_layout()
    if savefig:
        plt.savefig(figures_path + save_name + ".png")
    plt.show()
    return intersect_line

def meta_ds_results(df_path, ds_name, y_col, exclude_cols):
    # benchmark
    privacy, include_real, uniform, uniform2, maybe = 0, True, True, True, False

    data = pd.read_csv(dataset_path)#.sample()
    data = data.sample(int(len(data)/2))
    data = data.dropna()
    data = data[[col for col in data.columns if col not in exclude_cols]]
    data_cols = data.columns

    # selects categorical data
    std_cols = list(set(data.select_dtypes(["number"]).columns).difference({y_col}))
    for col in std_cols:
        data[col] = pd.cut(data[col], bins=10, labels=False).map(str)
    cat_columns = list(set(data.select_dtypes(["bool_", "object_","flexible"],["number"]).columns).difference({y_col}))

    meta_info = get_meta_info_pandas(data, cat_columns, privacy, include_real, uniform, uniform2, maybe, True)
    
    return meta_info

def sanitize_df(df_path, ds_name, y_col, exclude_cols, cases):

    model_dict = dict()
    model_dict["linear_regression"] = linear_model.LinearRegression()
    model_dict["svm"] = svm.SVC(gamma=0.001, C=100.)
    model_dict["naive_bayes"] = naive_bayes.GaussianNB()
    model_dict["tree"] = tree.DecisionTreeRegressor()

    processed_cases = list()
    case_model_scores = dict()
    reco_list = list()
    
    iterator = 0
    for case in cases:

        if iterator > 0:
            privacy, include_real, uniform, uniform2, maybe = case[0], case[1], case[2], case[3], case[4]
            case_name = str(privacy)+("m" if maybe else "t" if include_real else "f") + ("t" if uniform else "f")+("t" if uniform2 else "f")
        else:
            # benchmark
            privacy, include_real, uniform, uniform2, maybe = 0, True, True, True, False
            case_name = "0.0ttt"
        print(case_name)
        if case_name not in processed_cases:
            for rand_num in range(10):
                case_name_rand = case_name +"_"+ str(rand_num)
                display(case_name_rand)

                data = pd.read_csv(df_path)#.sample()
                data = data.sample(int(len(data)/2))
                data = data.dropna()
                data = data[[col for col in data.columns if col not in exclude_cols]]
                data_cols = data.columns
                data_y = data[y_col]

                # selects categorical data
                std_cols = list(set(data.select_dtypes(["number"]).columns).difference({y_col}))
                for col in std_cols:
                    data[col] = pd.cut(data[col], bins=10, labels=False).map(str)
                cat_columns = list(set(data.select_dtypes(["bool_", "object_","flexible"],["number"]).columns).difference({y_col}))

                if iterator > 0:
                    meta_info = get_meta_info_pandas(data, cat_columns, privacy, include_real, uniform, uniform2, maybe, False)
                else:
                    meta_info = get_meta_info_pandas(data, cat_columns, privacy, include_real, uniform, uniform2, maybe, True)

                asd = {}
                rmse_dict = {}
                for col in cat_columns:
                    if len(meta_info["columns"][col]["counter"])<50:
                        asd[col] = np.matrix([entry_sanitization_with_maybe(entry=x, **meta_info["algorithm"], **meta_info["columns"][col]) for x in data[col]])
                        rmse_dict[col] = sum([np.power(x-y,2) for x,y in zip(meta_info["columns"][col]["counter"].values(), asd[col].sum(axis=0))])
                nis = pd.DataFrame.from_dict(rmse_dict, orient="index").reset_index()
                nis.columns = ["class", "rmse"]
                nis["case"] = case_name_rand
                reco_list.append(nis)

                dataa =  pd.concat([pd.DataFrame(v, columns=[k+"/"+i for i in meta_info["columns"][k]["key_to_order"].keys()]) for k, v in asd.items()], axis=1)
                if len(data_y.unique())==2:
                    dataa["y"] = (data_y==data_y.unique()[0]).astype(int).values
                else: # in case there is numeric take the median
                    dataa["y"] = (data_y>=sorted(data_y)[int(len(data_y)/2)]).astype(int).values
                dataa = dataa.dropna()
                display(dataa.head())
                case_model_scores[case_name_rand] = dict()
                for model_name, model in model_dict.items():
                    try:
                        case_model_scores[case_name_rand][model_name] = get_auc_score_of_model(dataa, model)#[:2]
                    except exception as e:
                        print(e)
                    display(model_name + str(case_model_scores[case_name_rand][model_name][:2]))
                clear_output(wait=True)
            processed_cases.append(case_name)
            iterator += 1
            
    timestr = time.strftime("%Y%m%d")

    reco_df = pd.concat(reco_list)
    reco_df.to_csv("supervised_rmse_df_"+ds_name+timestr+".csv")

    df_models_scores = pd.DataFrame.from_dict(case_model_scores, orient="index").reset_index().rename(columns={"index":"case"})
    df_models_scores = df_models_scores.melt(id_vars=df_models_scores.columns[0], value_vars=df_models_scores.columns[1:], value_name="models")
    df_models_scores = pn.DataFrame.from_dict(case_model_scores, orient="index").reset_index().rename(columns={"index":"case"})
    df_models_scores = df_models_scores.melt(id_vars=["case"]).rename(columns={"variable":"model"})


    df_models_scores["privacy"] = df_models_scores["case"].map(lambda x: int("".join(re.findall("\d+", x)[:2])))
    df_models_scores["real"] = df_models_scores["case"].map(lambda x: re.findall("[^\d]",x)[1])
    df_models_scores["uniform"] = df_models_scores["case"].map(lambda x: int(re.findall("[^\d]",x)[2] == "t"))
    df_models_scores["uniform2"] = df_models_scores["case"].map(lambda x: int(re.findall("[^\d]",x)[3] == "t"))

    df_models_scores["error"] = df_models_scores["value"].map(lambda x: x[0])
    df_models_scores["auc"] = df_models_scores["value"].map(lambda x: x[1])

    def all_entries_vector(x):
        xs = ""
        for xi in x:
            xs += str(xi) + ","
        return xs[:-1]

    df_models_scores["roc_x"] = df_models_scores["value"].map(lambda x: all_entries_vector(x[2][0]))
    df_models_scores["roc_y"] = df_models_scores["value"].map(lambda x: all_entries_vector(x[2][1]))
    df_models = df_models_scores[["case", "model", "privacy", "real", "uniform", "uniform2", "error", "auc", "roc_x", "roc_y"]]
    df_models.columns = [["case", "model", "privacy", "real", "uniform", "uniform2", "error", "auc", "roc_x", "roc_y"]]
    df_models.to_csv("model_scores_roc_"+ds_name+timestr+".csv")

def check_include_real(include_real, privacy, class_length):
    if (privacy - include_real) >= class_length:
        include_real = True
    return include_real


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
                       order_exception, ordered_weights, counter):
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
    #print(entry)

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

    return entry_vector




def entry_sanitization_with_maybe(entry, real_prob, class_length,
                       maybe, uniform, uniform2, include_real,
                       privacy, order_weights, key_to_order,
                       order_exception, ordered_weights, counter):
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
    #print(entry)
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

def get_owoe(class_length, key_to_order, ordered_weights):
    
    order_exception = dict()
    order_weights = dict()

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
        
    return order_exception, order_weights






def get_meta_info(data, cat_columns, privacy, include_real, uniform, uniform2, maybe, client, benchmark=False):
    
    if not benchmark:
        meta_data = client.gather(client.compute({col: {"privacy": math.ceil(privacy*len(data[col].unique())),
                                                     "counter": data[col].value_counts()} for col in cat_columns}))
    else:
        meta_data = client.gather(client.compute({col: {"privacy": 1,
                                                     "counter": data[col].value_counts()} for col in cat_columns}))

    meta_data = {col:{"privacy":min(val["privacy"],len(val["counter"])), 
                      "class_length":len(val["counter"]),
                      "counter":val["counter"].to_dict()}
                 for col, val in meta_data.items()}

    meta_info = {"df":{"n":len(data)},
                 "columns":meta_data,
                "algorithm":{"uniform":uniform,
                            "uniform2":uniform2,
                            "real_prob":None,
                            "maybe":maybe}}

    # the meta info should include de get_owoe() information
    for col, col_info in meta_info["columns"].items():
        meta_info["columns"][col]["include_real"] = check_include_real(include_real, col_info["privacy"], col_info["class_length"])
        key_to_order =  dict(zip(sorted(col_info["counter"].keys()), range(col_info["class_length"])))
        ordered_weights = [float(col_info["counter"][key]) / meta_info["df"]["n"] for key in sorted(col_info["counter"].keys())]
        meta_info["columns"][col]["ordered_weights"] = ordered_weights
        meta_info["columns"][col]["key_to_order"] = key_to_order
        order_exception, order_weights = get_owoe(col_info["class_length"], key_to_order, ordered_weights)
        meta_info["columns"][col]["order_exception"] = order_exception
        meta_info["columns"][col]["order_weights"] = order_weights
    
    return meta_info


def get_meta_info_pandas(data, cat_columns, privacy, include_real, uniform, uniform2, maybe, benchmark=False):
    
    if not benchmark:
        meta_data = {col: {"privacy": math.ceil(privacy*len(data[col].unique())),
                                                     "counter": dict(Counter(data[col]))} for col in cat_columns}
    else:
        meta_data = {col: {"privacy": 1,"counter": dict(Counter(data[col]))} for col in cat_columns}

    meta_data = {col:{"privacy":min(val["privacy"],len(val["counter"])), 
                      "class_length":len(val["counter"]),
                      "counter": dict(val["counter"])}
                 for col, val in meta_data.items()}

    meta_info = {"df":{"n":len(data)},
                 "columns":meta_data,
                "algorithm":{"uniform":uniform,
                            "uniform2":uniform2,
                            "real_prob":None,
                            "maybe":maybe}}

    # the meta info should include de get_owoe() information
    for col, col_info in meta_info["columns"].items():
        meta_info["columns"][col]["include_real"] = check_include_real(include_real, col_info["privacy"], col_info["class_length"])
        key_to_order =  dict(zip(sorted(col_info["counter"].keys()), range(col_info["class_length"])))
        ordered_weights = [float(col_info["counter"][key]) / meta_info["df"]["n"] for key in sorted(col_info["counter"].keys())]
        meta_info["columns"][col]["ordered_weights"] = ordered_weights
        meta_info["columns"][col]["key_to_order"] = key_to_order
        order_exception, order_weights = get_owoe(col_info["class_length"], key_to_order, ordered_weights)
        meta_info["columns"][col]["order_exception"] = order_exception
        meta_info["columns"][col]["order_weights"] = order_weights
    
    return meta_info

def get_auc_score_of_model(df, model):
    """
    returns both the prediction error and the auc of the given model applied to the dataset
    
    param df: data with the `y` value placed in the last column and corresponds to a binary
    class
    param model: classification model 
    """
    data_x = df[[col for col in df.columns if col != "y"]]
    data_y = df["y"].astype(str)
    
    X = df.loc[:,[col for col in df.columns if col != "y"]]
    y = df.loc[:,"y"]
    # this could be done using cross validation for better confidence
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




english_dict = {"t": "include",
                "f": "not-include",
                "m": "maybe",
                "privacy": "privacy",
                "nclasses": "nclasses",
                "real": "include real"}

spanish_dict = {"t": "incluido",
                "f": "no-incluido",
                "m": "tal vez",
                "nclasses": "Total Clases",
                "privacy": "dispersión",
                "real": "incluir real",
                "auc": "area bajo la curva"}


def label_rename(label_list, language="english"):

    relabel_dict = english_dict if language == "english" else spanish_dict
    relabel_list = list()
    for lab in label_list:
        new_lab = relabel_dict.get(lab) if relabel_dict.get(lab) is not None else lab
        relabel_list.append(new_lab)
    return relabel_list


def get_label_name(param_dict, l_name=False, language="english"):
    """
    Gets the name of the label from the parameters being used.

    """
    param_list = [("privacy", "P={val}"), ("real", " R={val}"), ("uniform", " U={val}"),
                  ("uniform_original", " UO={val}"), ("model", " M={val}")]
    label_name = ""
    label_values = ""
    dict_use = english_dict if language=="english" else spanish_dict
    for param, valpar in param_list:
        if param_dict.get(param) is not None:
            original_value = str(param_dict.get(param))
            std_value = dict_use.get(original_value) if dict_use.get(original_value) else original_value
            label_name += valpar.format(val=std_value)
            label_values += std_value
    if l_name:
        return label_name
    else:
        return label_values


def get_single_filter_df(df, k, v):
    """
    Applies a filter to a pandas DataFrame, which might me a multiple condition
    """
    if v is not None:
        v = [v] if not isinstance(v, list) else v
        if k in df.columns:
            if np.issubdtype(df[k].dtype , np.number):
                cond = " | ".join(["{k} == {val}".format(k=k, val=float(v0)) for v0 in v])
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


def plot_bars(df, gb_param, yaxis, base_filter, lines_cases, savefig=False,  title=None, save_name=None,
              width_delta=.2, language="english"):
    """
    Returns a line plot with quantile intervals of the RMSE of different levels of either privacy or number of classes.
    Works only for the non-supervised datasets since there are multiples simulations for provacy levels and numberr of classes.

    """
    fig, ax = plt.subplots()
    pt = base_filter.get("privacy")
    if pt is not None:
        base_filter.pop("privacy")
        df = df.query("privacy < {pt}".format(pt=pt))
    if "uniform" in gb_param:
        df = df[df.uniform == df.uniform2]
    df = get_base_filtered_df(df, base_filter)
    ps = list()
    labels = list()
    width = 0
    if len(lines_cases)>0:
        for k, v in lines_cases.items():
            v = [v] if not isinstance(v, list) else v
            for v0 in v:
                dfc = get_single_filter_df(df, k, v0)

                gb = dfc.groupby([gb_param])[yaxis].mean().reset_index()
                gb2 = dfc.groupby([gb_param])[yaxis].std().reset_index()

                x = gb[gb_param].unique()
                ind = np.arange(len(x))
                curr_p = ax.bar(ind + width, gb[yaxis], width_delta, color=np.random.rand(3,),
                                bottom=0, yerr=gb2[yaxis])
                ps.append(curr_p)
                param_dict = {k: v0}
                tt = get_label_name(param_dict, True, language)
                labels.append(tt)
                width += width_delta
    else:
        gb = df.groupby([gb_param])[yaxis].mean().reset_index()
        gb2 = df.groupby([gb_param])[yaxis].std().reset_index()

        x = gb[gb_param].unique()
        ind = np.arange(len(x))
        curr_p = ax.bar(ind+width, gb[yaxis], width_delta, color=np.random.rand(3,),
                        bottom=0, yerr=gb2[yaxis])
        ps.append(curr_p)
        tt = get_label_name(base_filter, True, language)
        labels.append(tt)
        width += width_delta

    ax.set_title(title)
    ax.set_xticks(ind + width_delta / 2)
    ax.set_ylabel(yaxis)
    x = label_rename(x, language)
    ax.set_xticklabels(x, rotation=45, ha="right")
    ax.legend([list(p)[0] for p in ps], labels)

    dict_use = english_dict if language == "english" else spanish_dict
    gb_param = dict_use.get(gb_param.lower()) if dict_use.get(gb_param.lower()) else gb_param
    yaxis = dict_use.get(yaxis.lower()) if dict_use.get(yaxis.lower()) else yaxis
    ax.set_xlabel(gb_param.upper())
    ax.set_ylabel(yaxis.upper())
    ax.yaxis.set_major_formatter(FormatStrFormatter(y_tick_format))
    plt.tight_layout()
    if savefig:
        plt.savefig(figures_path + save_name + ".png")
    plt.show()


def plot_bars_single_chunk(df, gb_param, yaxis, base_filter, lines_cases, savefig=False, title=None, save_name=None,
                  width_delta=.2, language="english"):
    """
    Returns a line plot with quantile intervals of the RMSE of different levels of either privacy or number of classes.
    Works only for the non-supervised datasets since there are multiples simulations for provacy levels and numberr of classes.
    """
    colors2 = {0:"b",1:"r", 2:"g", 3:"m", "t":"g", "f":"r","m":"b", "linear_regression":"b", "svm":"r", "naive_bayes":"g", "tree":"m"}

    fig, ax = plt.subplots()
    pt = base_filter.get("privacy")
    if pt is not None:
        base_filter.pop("privacy")
        df = df.query("privacy < {pt}".format(pt=pt))
    if "uniform" in df.columns:    
        df = df[df.uniform == df.uniform2]
    df = get_base_filtered_df(df, base_filter)
    ps = list()
    labels = list()
    width = 0
    xticks = list()
    xticks_locs = list()
    citer=0
    tendency_points = list()
    if len(lines_cases) > 0:
        for k, v in lines_cases.items():
            v = [v] if not isinstance(v, list) else v
            for v0 in v:

                dfc = get_single_filter_df(df, k, str(v0))

                gb = dfc.groupby([gb_param])[yaxis].mean().reset_index()
                gb2 = dfc.groupby([gb_param])[yaxis].std().reset_index()

                x = gb[gb_param].unique()
                xticks.extend(x)
                ind = np.arange(len(x))
                xticks_locs.extend(ind+width)
                tendency_points.append((ind + width, gb[yaxis]))
                curr_p = ax.bar(ind + width, gb[yaxis], width_delta, color=colors2[citer % 2],
                                bottom=0, yerr=gb2[yaxis]) if gb_param == "privacy" else \
                    ax.bar(ind + width, gb[yaxis], width_delta, color=colors2[v0],
                                bottom=0, yerr=gb2[yaxis])
                citer += 1
                ps.append(curr_p)
                param_dict = {k: v0}
                tt = get_label_name(param_dict, True, language)
                labels.append(tt)
                width += width_delta
    ax.plot([t1[0] for t1 in tendency_points], [t1[1] for t1 in tendency_points], lw=5, c="k")
    ax.set_title(title)
    #ax.set_xticks(ind + width_delta / 2)
    ax.set_xticks(xticks_locs)
    ax.set_ylabel(yaxis)
    xticks = label_rename(xticks, language)
    ax.set_xticklabels(xticks, rotation=45, ha="right")
    #ax.legend([p[0] for p in ps], labels)
    dict_use = english_dict if language == "english" else spanish_dict
    gb_param = dict_use.get(gb_param.lower()) if dict_use.get(gb_param.lower()) else gb_param
    yaxis = dict_use.get(yaxis.lower()) if dict_use.get(yaxis.lower()) else yaxis
    ax.set_xlabel(gb_param.upper())
    ax.set_ylabel(yaxis.upper())
    ax.yaxis.set_major_formatter(FormatStrFormatter(y_tick_format))
    plt.tight_layout()
    if savefig:
        plt.savefig(figures_path + save_name + ".png")
    plt.show()


def plot_bars_single_chunk_no_tendency(df, gb_param, yaxis, base_filter, lines_cases, savefig=False, title=None, save_name=None,
                  width_delta=.2, language="english"):
    """
    Returns a line plot with quantile intervals of the RMSE of different levels of either privacy or number of classes.
    Works only for the non-supervised datasets since there are multiples simulations for provacy levels and numberr of classes.

    """
    colors2 = {0:"b",1:"r","t":"g", "f":"r","m":"r"}

    fig, ax = plt.subplots()
    pt = base_filter.get("privacy")
    if pt is not None:
        base_filter.pop("privacy")
        df = df.query("privacy < {pt}".format(pt=pt))
    if "uniform" in df.columns:    
        df = df[df.uniform == df.uniform2]
    df = get_base_filtered_df(df, base_filter)
    ps = list()
    labels = list()
    width = 0
    xticks = list()
    xticks_locs = list()
    citer=0
    if len(lines_cases) > 0:
        for k, v in lines_cases.items():
            print(v)
            v = [v] if not isinstance(v, list) else v
            for v0 in v:
                print(v0)
                dfc = get_single_filter_df(df.copy(), k, v0)

                gb = dfc.groupby([gb_param])[yaxis].mean().reset_index()
                gb2 = dfc.groupby([gb_param])[yaxis].std().reset_index()

                x = gb[gb_param].unique()
                xticks.extend(x)
                ind = np.arange(len(x))
                xticks_locs.extend(ind+width)
                curr_p = ax.bar(ind + width, gb[yaxis], width_delta, color=colors2[citer % 2],
                                bottom=0, yerr=gb2[yaxis]) if gb_param == "privacy" else \
                    ax.bar(ind + width, gb[yaxis], width_delta, color=colors2[v0],
                                bottom=0, yerr=gb2[yaxis])
                citer += 1
                ps.append(curr_p)
                param_dict = {k: v0}
                tt = get_label_name(param_dict, True, language)
                labels.append(tt)
                width += width_delta
    ax.set_title(title)
    #ax.set_xticks(ind + width_delta / 2)
    ax.set_xticks(xticks_locs)
    ax.set_ylabel(yaxis)
    xticks = label_rename(xticks, language)
    ax.set_xticklabels(xticks, rotation = 45, ha="right")
    #ax.legend([p[0] for p in ps], labels)
    dict_use = english_dict if language == "english" else spanish_dict
    gb_param = dict_use.get(gb_param.lower()) if dict_use.get(gb_param.lower()) else gb_param
    yaxis = dict_use.get(yaxis.lower()) if dict_use.get(yaxis.lower()) else yaxis
    ax.set_xlabel(gb_param.upper())
    ax.set_ylabel(yaxis.upper())
    ax.yaxis.set_major_formatter(FormatStrFormatter(y_tick_format))
    plt.tight_layout()
    if savefig:
        plt.savefig(figures_path + save_name + ".png")
    plt.show()


def plot_intervals(df, gb_param, yaxis, base_filter, lines_cases, savefig=False,  title=None, save_name=None,
                   language="english"):
    """
    Returns a line plot with quantile intervals of the RMSE of different levels of either privacy or number of classes.
    Works only for the non-supervised datasets since there are multiples simulations for provacy levels and numberr of classes.

    """
    fig, ax = plt.subplots()
    pt = base_filter.get("privacy")
    if pt is not None:
        base_filter.pop("privacy")
        df = df.query("privacy < {pt}".format(pt=pt))
    df = get_base_filtered_df(df, base_filter)
    labels = []
    if "uniform" in df.columns:    
        df = df[df.uniform == df.uniform2]
    y_max = 0
    if len(lines_cases)>0:
        for k, v in lines_cases.items():
            v = [v] if not isinstance(v, list) else v
            for v0 in v:
                dfc = get_single_filter_df(df, k, v0)
                gb = dfc.groupby([gb_param])[yaxis].quantile([.1,.25,.5,.75,0.9]).reset_index()
                x = gb[gb_param].unique()
                y1 = gb.query("level_1 == 0.25")[yaxis]
                y2 = gb.query("level_1 == 0.50")[yaxis]
                y3 = gb.query("level_1 == 0.75")[yaxis]
                ax.fill_between(x, y1, y3, color='grey', alpha='0.5')
                ax.plot(x, y2)
                param_dict = {k: v0}
                tt = get_label_name(param_dict, True, language)
                labels.append(tt)
                lines, _ = ax.get_legend_handles_labels()
                y_max = max(y_max, max(y2))
    else:
        gb = df.groupby([gb_param])[yaxis].quantile([.1,.25,.5,.75,0.9]).reset_index()
        x = gb[gb_param].unique()
        y1 = gb.query("level_1 == 0.25")[yaxis]
        y2 = gb.query("level_1 == 0.50")[yaxis]
        y3 = gb.query("level_1 == 0.75")[yaxis]
        ax.fill_between(x, y1, y3, color='grey', alpha='0.5')
        ax.plot(x,y2)
        tt = get_label_name(base_filter, True, language)
        labels.append(tt)
        lines, _ = ax.get_legend_handles_labels()
        y_max = max(y_max, max(y2))

    ax.legend(lines, labels, loc='best')
    ax.set_title(title)
    #ax.set_ylim([0, y_max*1.5])
    dict_use = english_dict if language == "english" else spanish_dict
    gb_param = dict_use.get(gb_param.lower()) if dict_use.get(gb_param.lower()) else gb_param
    yaxis = dict_use.get(yaxis.lower()) if dict_use.get(yaxis.lower()) else yaxis
    ax.set_xlabel(gb_param.upper())
    ax.set_ylabel(yaxis.upper())
    if yaxis=="auc":
        ax.yaxis.set_major_formatter(FormatStrFormatter(y_tick_format))
    else:
        ax.yaxis.set_major_formatter(y_tick_format_int)
    if gb_param == "privacy":
        ax.xaxis.set_major_formatter(x_tick_percent)
    plt.tight_layout()
    if savefig:
        plt.savefig(figures_path + save_name + ".png")
    plt.show()

    
def plot_intervals_smooth(df, gb_param, yaxis, base_filter, lines_cases, savefig=False,  title=None, save_name=None,
                   language="english"):
    """
    Returns a line plot with quantile intervals of the RMSE of different levels of either privacy or number of classes.
    Works only for the non-supervised datasets since there are multiples simulations for provacy levels and numberr of classes.
    
    """
    fig, ax = plt.subplots()
    pt = base_filter.get("privacy")
    if pt is not None:
        base_filter.pop("privacy")
        df = df.query("privacy < {pt}".format(pt=pt))
    df = get_base_filtered_df(df, base_filter)
    labels = []
    if "uniform" in df.columns:    
        df = df[df.uniform == df.uniform2]
    y_max = 0
    if len(lines_cases)>0:
        for k, v in lines_cases.items():
            v = [v] if not isinstance(v, list) else v
            for v0 in v:
                dfc = get_single_filter_df(df, k, v0)
                gb = dfc.groupby([gb_param])[yaxis].quantile([.1,.25,.5,.75,0.9]).reset_index()
                x = gb[gb_param].unique()
                xnew = np.linspace(x.min(),x.max(),300) #300 represents number of points to make between T.min and T.max
                y1 = spline(x, gb.query("level_1 == 0.25")[yaxis], xnew)
                y2 = spline(x, gb.query("level_1 == 0.50")[yaxis], xnew)
                y3 = spline(x, gb.query("level_1 == 0.75")[yaxis], xnew)
                ax.fill_between(xnew, y1, y3, color='grey', alpha='0.5')
                ax.plot(xnew, y2)
                param_dict = {k: v0}
                tt = get_label_name(param_dict, True, language)
                labels.append(tt)
                lines, _ = ax.get_legend_handles_labels()
                y_max = max(y_max, max(y2))
    else:
        gb = df.groupby([gb_param])[yaxis].quantile([.1,.25,.5,.75,0.9]).reset_index()
        x = gb[gb_param].unique()
        xnew = np.linspace(x.min(),x.max(),300) #300 represents number of points to make between T.min and T.max
        y1 = spline(x, gb.query("level_1 == 0.25")[yaxis], xnew)
        y2 = spline(x, gb.query("level_1 == 0.50")[yaxis], xnew)
        y3 = spline(x, gb.query("level_1 == 0.75")[yaxis], xnew)
        ax.fill_between(xnew, y1, y3, color='grey', alpha='0.5')
        ax.plot(xnew,y2)
        tt = get_label_name(base_filter, True, language)
        labels.append(tt)
        lines, _ = ax.get_legend_handles_labels()
        y_max = max(y_max, max(y2))

    ax.legend(lines, labels, loc='best')
    ax.set_title(title)
    #ax.set_ylim([0, y_max*1.5])
    dict_use = english_dict if language == "english" else spanish_dict
    gb_param = dict_use.get(gb_param.lower()) if dict_use.get(gb_param.lower()) else gb_param
    yaxis = dict_use.get(yaxis.lower()) if dict_use.get(yaxis.lower()) else yaxis
    ax.set_xlabel(gb_param.upper())
    ax.set_ylabel(yaxis.upper())
    if yaxis=="auc":
        ax.yaxis.set_major_formatter(FormatStrFormatter(y_tick_format))
    else:
        ax.yaxis.set_major_formatter(y_tick_format_int)
    if gb_param == "privacy":
        ax.xaxis.set_major_formatter(x_tick_percent)
    plt.tight_layout()
    if savefig:
        plt.savefig(figures_path + save_name + ".png")
    plt.show()


def plot_intervals_std(df, gb_param, yaxis, base_filter, lines_cases, savefig=False,  title=None, save_name=None,
                       language="english"):
    """
    Returns a line plot with quantile intervals of the RMSE of different levels of either privacy or number of classes.
    Works only for the non-supervised datasets since there are multiples simulations for provacy levels and numberr of classes.

    """
    fig, ax = plt.subplots()
    pt = base_filter.get("privacy")
    if pt is not None:
        base_filter.pop("privacy")
        df = df.query("privacy < {pt}".format(pt=pt))
    df = get_base_filtered_df(df, base_filter)
    labels = []
    if "uniform" in df.columns:    
        df = df[df.uniform == df.uniform2]
    y_max = 0
    if len(lines_cases)>0:
        for k, v in lines_cases.items():
            v = [v] if not isinstance(v, list) else v
            for v0 in v:
                dfc = get_single_filter_df(df, k, v0)
                gb = dfc.groupby([gb_param])[yaxis].mean().reset_index()
                x = gb[gb_param].unique()
                gb_std = dfc.groupby([gb_param])[yaxis].std().reset_index()
                y2 = gb[yaxis]
                y1 = (gb[yaxis] - gb_std[yaxis]).map(lambda x: max(x,0))
                y3 = gb[yaxis] + gb_std[yaxis]
                ax.fill_between(x, y1, y3, color='grey', alpha='0.5')
                ax.plot(x,y2)
                param_dict = {k: v0}
                tt = get_label_name(param_dict, True, language)
                labels.append(tt)
                lines, _ = ax.get_legend_handles_labels()
                y_max = max(y_max, max(y2))
    else:
        gb = df.groupby([gb_param])[yaxis].mean().reset_index()
        x = gb[gb_param].unique()
        gb_std = df.groupby([gb_param])[yaxis].std().reset_index()
        y2 = gb[yaxis]
        y1 = (gb[yaxis] - gb_std[yaxis]).map(lambda x: max(x, 0))
        y3 = gb[yaxis] + gb_std[yaxis]
        ax.fill_between(x, y1, y3, color='grey', alpha='0.5')
        ax.plot(x,y2)
        tt = get_label_name(base_filter, True, language)
        labels.append(tt)
        lines, _ = ax.get_legend_handles_labels()
        y_max = max(y_max, max(y2))

    ax.legend(lines, labels, loc='best')
    ax.set_title(title)
    #ax.set_ylim([0, y_max*1.5])
    dict_use = english_dict if language == "english" else spanish_dict
    gb_param = dict_use.get(gb_param.lower()) if dict_use.get(gb_param.lower()) else gb_param
    yaxis = dict_use.get(yaxis.lower()) if dict_use.get(yaxis.lower()) else yaxis
    ax.set_xlabel(gb_param.upper())
    ax.set_ylabel(yaxis.upper())
    ax.yaxis.set_major_formatter(y_tick_format_int)
    if gb_param == "privacy":
        ax.xaxis.set_major_formatter(x_tick_percent)
    plt.tight_layout()
    if savefig:
        plt.savefig(figures_path + save_name + ".png")
    plt.show()


def rocs_by_case(df, base_filter, lines_cases, savefig=False, title=None, save_name=None, language="english"):
    """
    Gets the ROC plots for all privacy levels and for the sliced frame with the desired parameters 
    """

    fig, ax = plt.subplots()
    labels = []
    if "uniform" in df.columns:    
        df = df[df.uniform == df.uniform2]
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
            gb_std = df_roc.groupby("fpr_dis")["tpr"].std().reset_index().sort_values(by="fpr_dis", ascending=True)

            x = gb.fpr_dis
            y = gb.tpr.rolling(window=3, center=False).mean() if len(gb) > 10 else gb.tpr
            y_std = gb_std.tpr.rolling(window=3, center=False).mean() if len(gb) > 10 else gb_std.tpr
            y_std = gb_std.tpr
            y1 = y - y_std
            y3 = y + y_std
            ax.fill_between(x, y1, y3, color='grey', alpha='0.5')
            ax.plot(x, y)

            lines, _ = ax.get_legend_handles_labels()
            param_dict = {k:v0}
            tt = get_label_name(param_dict, True, language)
            labels.append(tt)

    ax.legend(lines, labels, loc='best')
    tt = "Income DB " + title
    ax.set_title(tt)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    plt.tight_layout()
    if savefig:
        plt.savefig(figures_path + save_name + ".png")
    plt.show()  
        

def rmse_auc_plot_no_intervals(df, gb_param, yaxis, reals, uniforms, uniforms2, uniform_original,
                               models, combined_cond=None, savefig=False, title=None, save_name=None,
                               language="english"):
    """
    Gets the supervised RMSE plot, since non supervised also has RMSE, pending is to check what is the difference between 
    this function and the plot_params, which plots the RMSE for the non supervised case. Both rmse and auc are merged now
    the only pending situation is the plot_params separation from this function.
    The difference is that plot params has quantile intervals and this one doesn't. This one can be used for rmse in the 
    supervised case in case quantiles are not needed. The AUC with confidence intervals where can it go? there is not enough 
    simulations.

    yaxis: is either rmse or auc, both lower cased
    """
    df = df.query("privacy < 11")
    fig, ax = plt.subplots()
    labels = []
    if "uniform" in df.columns:    
        df = df[df.uniform == df.uniform2]
    for real in reals:
        for uniform in uniforms:
            for uniform2 in uniforms2:
                for uo in uniform_original:
                    for model in models:
                        if combined_cond is not None and isinstance(combined_cond, dict):
                            for tp, vl in combined_cond.items():
                                param_dict = {"real":real, "uniform":uniform, "uniform_original":uo,
                                              "uniform2": uniform2, "model":model}
                                for col, val in zip([tp]*len(vl), vl):
                                    dfc = df
                                    for i, j in enumerate(col):
                                        dfc = get_single_filter_df(dfc, j, val[i])
                                        param_dict[j] = val[i]
                                    dfc = get_single_filter_df(dfc, "real", real)
                                    dfc = get_single_filter_df(dfc, "uniform", uniform)
                                    dfc = get_single_filter_df(dfc, "uniform2", uniform2)
                                    dfc = get_single_filter_df(dfc, "uniform_original", uo)
                                    dfc = get_single_filter_df(dfc, "model", model)

                                    dfc.loc[:, gb_param] = dfc[gb_param].map(int)
                                    gb = dfc.sort_values(by="privacy", ascending=True)
                                    gb = gb.groupby(gb_param)[yaxis].agg(lambda x: np.mean(x)).reset_index()
                                    x = gb[gb_param]
                                    y = gb[yaxis]
                                    ax.plot(x, y)
                                    if len(gb) > 0:
                                        lines, _ = ax.get_legend_handles_labels()
                                        tt = get_label_name(param_dict, False, language)
                                        labels.append(tt)
                        else:
                            param_dict = {"real": real, "uniform": uniform, "uniform_original": uo,
                                          "uniform2": uniform2, "model": model}
                            dfc = get_single_filter_df(df, "real", real)
                            dfc = get_single_filter_df(dfc, "uniform", uniform)
                            dfc = get_single_filter_df(dfc, "uniform2", uniform2)
                            dfc = get_single_filter_df(dfc, "uniform_original", uo)
                            dfc = get_single_filter_df(dfc, "model", model)

                            dfc.loc[:, gb_param] = dfc[gb_param].map(int)
                            gb = dfc.sort_values(by="privacy", ascending=True)
                            gb = gb.groupby(gb_param)[yaxis].agg(lambda x: np.mean(x)).reset_index()
                            x = gb[gb_param]
                            y = gb[yaxis]
                            ax.plot(x, y)
                            if len(gb) > 0:
                                lines, _ = ax.get_legend_handles_labels()
                                tt = get_label_name(param_dict, False, language)
                                labels.append(tt)
    ax.legend(lines, labels, loc='best')
    ax.set_title(title)
    dict_use = english_dict if language == "english" else spanish_dict
    gb_param = dict_use.get(gb_param.lower()) if dict_use.get(gb_param.lower()) else gb_param
    yaxis = dict_use.get(yaxis.lower()) if dict_use.get(yaxis.lower()) else yaxis
    ax.set_xlabel(gb_param.upper())
    ax.set_ylabel(yaxis.upper())
    ax.yaxis.set_major_formatter(FormatStrFormatter(y_tick_format))
    plt.tight_layout()
    if savefig:
        plt.savefig(figures_path + save_name + ".png")
    plt.show()


def rmse_auc_plot_with_intervals(df, gb_param, yaxis, reals, uniforms, uniforms2, uniform_original,
                               models, combined_cond=None, savefig=False, title=None, save_name=None,
                                 language="english"):
    """
    Gets the supervised RMSE plot, since non supervised also has RMSE, pending is to check what is the difference between
    this function and the plot_params, which plots the RMSE for the non supervised case. Both rmse and auc are merged now
    the only pending situation is the plot_params separation from this function.
    The difference is that plot params has quantile intervals and this one doesn't. This one can be used for rmse in the
    supervised case in case quantiles are not needed. The AUC with confidence intervals where can it go? there is not enough
    simulations.

    yaxis: is either rmse or auc, both lower cased
    """
    df = df.query("privacy < 11")
    fig, ax = plt.subplots()
    labels = []
    if "uniform" in df.columns:    
        df = df[df.uniform == df.uniform2]
    for real in reals:
        for uniform in uniforms:
            for uniform2 in uniforms2:
                for uo in uniform_original:
                    for model in models:
                        if combined_cond is not None and isinstance(combined_cond, dict):
                            for tp, vl in combined_cond.items():
                                param_dict = {"real":real, "uniform":uniform, "uniform_original":uo,
                                              "uniform2": uniform2, "model":model}
                                for col, val in zip([tp]*len(vl), vl):
                                    dfc = df
                                    for i, j in enumerate(col):
                                        dfc = get_single_filter_df(dfc, j, val[i])
                                        param_dict[j] = val[i]
                                    dfc = get_single_filter_df(dfc, "real", real)
                                    dfc = get_single_filter_df(dfc, "uniform", uniform)
                                    dfc = get_single_filter_df(dfc, "uniform2", uniform2)
                                    dfc = get_single_filter_df(dfc, "uniform_original", uo)
                                    dfc = get_single_filter_df(dfc, "model", model)

                                    dfc.loc[:, gb_param] = dfc[gb_param].map(int)
                                    gb = dfc.sort_values(by="privacy", ascending=True)
                                    gb = gb.groupby(gb_param)[yaxis].agg(lambda x: np.mean(x)).reset_index()
                                    x = gb[gb_param]
                                    y = gb[yaxis]
                                    ax.plot(x, y)

                                    gb_std = dfc.groupby([gb_param])[yaxis].std().reset_index()
                                    y1 = (gb[yaxis] - gb_std[yaxis]).map(lambda x: max(x, 0))
                                    y3 = gb[yaxis] + gb_std[yaxis]
                                    ax.fill_between(x, y1, y3, color='grey', alpha='0.5')

                                    if len(gb) > 0:
                                        lines, _ = ax.get_legend_handles_labels()
                                        tt = get_label_name(param_dict, False, language)
                                        labels.append(tt)
                        else:
                            param_dict = {"real": real, "uniform": uniform, "uniform_original": uo,
                                          "uniform2": uniform2, "model": model}
                            dfc = get_single_filter_df(df, "real", real)
                            dfc = get_single_filter_df(dfc, "uniform", uniform)
                            dfc = get_single_filter_df(dfc, "uniform2", uniform2)
                            dfc = get_single_filter_df(dfc, "uniform_original", uo)
                            dfc = get_single_filter_df(dfc, "model", model)

                            dfc.loc[:, gb_param] = dfc[gb_param].map(int)
                            gb = dfc.sort_values(by="privacy", ascending=True)
                            gb = gb.groupby(gb_param)[yaxis].agg(lambda x: np.mean(x)).reset_index()
                            x = gb[gb_param]
                            y = gb[yaxis]
                            ax.plot(x, y)

                            gb_std = dfc.groupby([gb_param])[yaxis].std().reset_index()
                            y1 = (gb[yaxis] - gb_std[yaxis]).map(lambda x: max(x, 0))
                            y3 = gb[yaxis] + gb_std[yaxis]
                            ax.fill_between(x, y1, y3, color='grey', alpha='0.5')

                            if len(gb) >0:
                                lines, _ = ax.get_legend_handles_labels()
                                tt = get_label_name(param_dict, False, language)
                                labels.append(tt)
    ax.legend(lines, labels, loc='best')
    ax.set_title(title)
    dict_use = english_dict if language == "english" else spanish_dict
    gb_param = dict_use.get(gb_param.lower()) if dict_use.get(gb_param.lower()) else gb_param
    yaxis = dict_use.get(yaxis.lower()) if dict_use.get(yaxis.lower()) else yaxis
    ax.set_xlabel(gb_param.upper())
    ax.set_ylabel(yaxis.upper())
    ax.yaxis.set_major_formatter(FormatStrFormatter(y_tick_format))
    if gb_param == "privacy":
        ax.xaxis.set_major_formatter(x_tick_percent)
    plt.tight_layout()
    if savefig:
        plt.savefig(figures_path + save_name + ".png")
    plt.show()

