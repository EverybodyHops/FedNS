from get_dirichlet import my_dirichlet
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
import math
import random
from get_noniid import noniid

class split_dataset:
    def __init__(self, image, label):
        self.cate_num = len(label[0])
        self.data_num = len(image)
        self.data = []
        for i in range(self.cate_num):
            self.data.append([])

        for i, l in tqdm(enumerate(label)):
            for j, t in enumerate(l):
                if t == 1:
                    self.data[j].append(tuple((image[i], l)))
                    break

        self.number = np.zeros(10, dtype="int32")
        for i in range(self.cate_num):
            self.number[i] = len(self.data[i])
        print(self.number)
        self.now = np.zeros(10, dtype="int32")

    def get_distribution(self, d, num):
        t = num * d
        differ = []
        ret = []
        s = 0
        for i in range(self.cate_num):
            ret.append(math.floor(t[i]))
            differ.append(t[i] - ret[i])
            s += ret[i]

        while s < num:
            m = 0
            m_ind = 0
            for i in range(self.cate_num):
                if differ[i] > m:
                    m = differ[i]
                    m_ind = i
            ret[m_ind] += 1
            differ[m_ind] = 0
            s += 1
        return ret

    def get_data(self, category, num):
        ret = []
        while num > 0:
            ret.append(self.data[category][self.now[category]])
            self.now[category] += 1
            if self.now[category] == self.number[category]:
                self.now[category] = 0
            num -= 1
        return ret

    def get_one_data_set(self, distribution):
        ret = []
        for i in range(10):
            ret.extend(self.get_data(i, distribution[i]))
        random.shuffle(ret)
        return ret

    def get_all_data(self, client_num, alpha):
        one_client_num = int(self.data_num / client_num)
        ret = []
        md = my_dirichlet()
        d = md.get_dirichlet(alpha, self.number / self.data_num, client_num)
        # nd = noniid()
        # d = nd.get_noniid(alpha)
        for i in range(client_num):
            to_append = []
            distribution = self.get_distribution(d[i], one_client_num)
            print(distribution, sum(distribution))
            to_append.append(self.get_one_data_set(distribution))
            to_append.append(self.get_one_data_set(distribution))
            ret.append(to_append)
        return ret

class my_dataset:
    def __init__(self, data):
        self.images = []
        self.labels = []
        self.test_images = []
        self.test_labels = []
        self.now = 0
        for d in data[0]:
            self.images.append(d[0])
            self.labels.append(d[1])

        for d in data[1]:
            self.test_images.append(d[0])
            self.test_labels.append(d[1])

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.test_images = np.array(self.test_images)
        self.test_labels = np.array(self.test_labels)

    def next_batch(self, size):
        ret = []
        if (self.now + size) <= self.images.shape[0]:
            ret.append(self.images[self.now: self.now + size])
            ret.append(self.labels[self.now: self.now + size])
            self.now += size
        else:
            ret.append(np.concatenate((self.images[self.now:], self.images[: self.now + size - self.images.shape[0]]), axis=0))
            ret.append(np.concatenate((self.labels[self.now:], self.labels[: self.now + size - self.images.shape[0]]), axis=0))
            self.now = self.now + size - self.images.shape[0]
        # print(self.now)
        # print(ret[0].shape, ret[1].shape)
        return ret

if __name__ == "__main__":
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    a = split_dataset(mnist.train.images, mnist.train.labels)
    b = a.get_all_data(10, 10)
    c = my_dataset(b[0])
    c.next_batch(50)
