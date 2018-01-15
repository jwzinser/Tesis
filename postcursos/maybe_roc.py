import pandas as pn
import matplotlib as plt
import re


data = pn.read_csv("~/Workspace/Tesis/data/census/maybe/roc/df.csv")
df = data[data.columns[1:]]

df["privacy"] = df["case"].map(lambda x: re.findall("\d+", x)[0])
df["real"] = df["case"].map(lambda x: re.findall("[^\d]",x)[0])
df["uniform"] = df["case"].map(lambda x: int(re.findall("[^\d]",x)[1] == "t"))



# make one roc curve for each case

fig, ax = plt.subplots()

