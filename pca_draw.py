import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

DATA_DIC = "./model_data/fedavg_alpha10000"

def load_data(dic, num):
    res = []
    for i in range(num):
        file_name = dic + "/%d.npy" % (i)
        # print(file_name)
        a = np.load(file_name)
        res.append(a.reshape(a.shape[0]* a.shape[1]))
    return res

def draw_pic(pca_data, x_idx, y_idx):
    print(pca_data)
    X = []
    Y = []
    for d in pca_data:
        X.append(d[x_idx])
        Y.append(d[y_idx])

    plt.figure()
    plt.plot(X, Y, linewidth=1)
    plt.show()

if __name__ == "__main__":
    data = load_data(DATA_DIC, 300)
    pca = PCA(n_components=10)
    pca.fit(data)
    draw_pic(pca.transform(data), 1, 2)
