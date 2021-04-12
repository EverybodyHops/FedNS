import fed_mnist as fed_data
import split_dataset
import mnist_model as fed_model
import tensorflow as tf
import random
from tqdm import tqdm
import numpy as np
import argparse

class FedAvg:
    def __init__(self, client_num, batch_size, batch_num, alpha, C):
        self.model = fed_model.model(sess=tf.InteractiveSession())
        self.global_model = self.model.get_model_values()
        self.local_models = []
        self.client_num = client_num
        for i in range(client_num):
            self.local_models.append(self.model.get_model_values())
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.alpha = alpha
        self.C = C

        self.test = fed_data.get_test()
        self.train = fed_data.get_train()
        self.split = split_dataset.split_dataset(self.train[0], self.train[1])
        self.all_data = self.split.get_all_data(client_num=self.client_num, alpha=self.alpha)

        self.trace_data = []
        for i in range(self.client_num):
            self.trace_data.append([])
        self.global_acc = []

    def select_client(self):
        num = int(self.C * self.client_num)
        ret = []
        while len(ret) < num:
            t = random.randint(0, self.client_num - 1)
            if t not in ret:
                ret.append(t)
        return ret

    def train_one(self, client_idx):
        self.model.set_model_values(self.global_model)
        now_dataset = split_dataset.my_dataset(self.all_data[client_idx])
        print("Start train client ", client_idx)
        for i in range(self.batch_num):
            batch = now_dataset.next_batch(self.batch_size)
            self.model.train(batch[0], batch[1])
        self.local_models[client_idx] = self.model.get_model_values()
        print("End train client ", client_idx, ", now local acc: ", self.model.vaild(self.test[0], self.test[1]))

    def train_all(self, to_train):
        self.model.set_model_values(self.global_model)        
        g_acc = self.model.vaild(self.test[0], self.test[1])
        self.global_acc.append(g_acc)
        print("Start new Round! now global acc:", g_acc)
        for idx in to_train:
            self.train_one(idx)
        print("End Round!")

    def aggregation(self, trained):
        tt = trained[0]
        for i in range(len(self.global_model.v_list)):
            self.global_model.v_list[i] = self.local_models[tt].v_list[i]

        t = len(trained)
        for k in range(1, t):
            i = trained[k]
            for j in range(len(self.global_model.v_list)):
                self.global_model.v_list[j] = np.add(self.global_model.v_list[j], self.local_models[i].v_list[j])

        for i in range(len(self.global_model.v_list)):
            self.global_model.v_list[i] = np.divide(self.global_model.v_list[i], t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fedavg.")
    parser.add_argument('-bn', '--batch_num', default=12, type=int)
    parser.add_argument('-a', '--alpha', default=1, type=float)
    parser.add_argument('-f', '--file', default="exp_data/fedavg_default.txt")
    args = parser.parse_args()

    fedavg = FedAvg(100, 50, args.batch_num, args.alpha, 0.1)
    for i in range(100):
        print("Now round ", i)
        now = fedavg.select_client()
        fedavg.train_all(now)
        fedavg.aggregation(now)
        print(fedavg.global_model.v_list[4].shape)
        np.save("./model_data/fedavg/%d" % (i), fedavg.global_model.v_list[4])
    np.savetxt(args.file, np.array(fedavg.global_acc))