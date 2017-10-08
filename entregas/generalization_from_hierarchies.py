import pandas as pn
from os import listdir
from os.path import isfile, join

census_name = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
               "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
               "salary-class"]

hierarchies_path = "/home/juanzinser/Workspace/Tesis/data/census/hierarchies/"
hier_files = [f for f in listdir(hierarchies_path) if isfile(join(hierarchies_path, f))]

hierarchies_dict = dict()
hier_columns = list()

for attr in census_name:
    if attr + ".csv" in hier_files:
        hier_columns.append(attr)
        hier_data = pn.read_csv(hierarchies_path + attr + ".csv", sep=";", names=range(10))
        # check which column has all entries as *
        hier_data = hier_data.dropna(axis=1, how="all")
        hier_data.index = hier_data.ix[:, 0]
        hierarchies_dict[attr] = hier_data.to_dict()




# generar bases con nivel 0 (original), 1, 2 , 3 y 4 y guardarlas en
tables_path = "/home/juanzinser/Workspace/Tesis/data/census/"
census_name = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
               "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
               "salary-class"]

census_data = pn.read_csv("/home/juanzinser/Workspace/Tesis/data/census/census.csv", names=census_name)

# do not change the sensitive column
for h in range(4):
    curr_data = census_data[hier_columns]
    for col in hier_columns:
        if col != "salary-class":
            generalizer = hierarchies_dict[col]
            max_level = max(generalizer.keys())
            curr_data.loc[:,col + "_prep"] = curr_data[col].map(lambda x: x.strip() if isinstance(x,basestring) else x)
            curr_data.loc[:,col] = curr_data[col + "_prep"].map(lambda x: generalizer[min(h, max_level)][x] if
            x in generalizer[min(h, max_level)].keys() else x)
            curr_data = curr_data.drop(col + "_prep" ,axis=1)
    curr_data.to_csv(tables_path + "census_level_" + str(h) + ".csv", index=None)

