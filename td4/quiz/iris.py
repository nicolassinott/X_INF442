import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

fname = '../csv/iris.csv'
data = pd.read_csv(fname, header = 0)

tree = linkage(data[['a', 'b', 'c', 'd']])
plt.figure(figsize=(12,12))
D = dendrogram(tree, labels = data['name'].to_numpy(), orientation = 'left')
plt.show()
