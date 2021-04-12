from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def get_train():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    images = np.concatenate((mnist.train.images, mnist.validation.images), axis=0)
    labels = np.concatenate((mnist.train.labels, mnist.validation.labels), axis=0)
    print(images.shape, labels.shape)
    return tuple((images, labels))

def get_test():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print(mnist.test.images.shape, mnist.test.labels.shape)
    return tuple((mnist.test.images, mnist.test.labels))

if __name__ == "__main__":
    get_train()
    get_test()