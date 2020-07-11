"""example of density peak cluster algorithm."""
import pandas as pd

from dpca import DensityPeakCluster

# file name
file = "dataset4"

# load data
# data = pd.read_csv(r"data/data/%s.txt" % file, sep="\t", header=None)
data = pd.read_csv(r"data/data/dataNoNoise.csv", sep=",", header=None)

# dpca model
# plot decision graph to set params `density_threshold`, `distance_threshold`.
# dpca = DensityPeakCluster(density_threshold=120, distance_threshold=0.2, anormal=False)
# 数据集5
# dpca = DensityPeakCluster(density_threshold=140, distance_threshold=0.20, anormal=False)
# 数据集4
# dpca = DensityPeakCluster(density_threshold=40, distance_threshold=0.6, anormal=False)
# 数据集5
# dpca = DensityPeakCluster(density_threshold=7, distance_threshold=7, anormal=False)
# dnsData
dpca = DensityPeakCluster(density_threshold=0, distance_threshold=50, anormal=False)

# fit model
dpca.fit(data.iloc[:, [0, 1]])

# print predict label
# print(dpca.labels_)

# plot cluster
dpca.plot("all", title=file, save_path="data/result")
