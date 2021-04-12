from scipy.stats import dirichlet
import numpy as np

class my_dirichlet:
    def __init__(self, random_state=17):
        self.random_state = random_state
    
    def set_random_state(self, random_state):
        self.random_state = random_state

    def get_random_state(self):
        return self.random_state

    def get_dirichlet(self, alpha, p, size=1):
        q = alpha * p
        return dirichlet.rvs(q, size, self.random_state)

    def test(self, alpha, p, size=1):
        q = alpha * p
        d = dirichlet.rvs(q, size, self.random_state)
        print(d.shape)
        mean = np.zeros(q.shape)
        for i in d:
            mean = mean + i
            # print(i)
        mean = mean / size
        print(mean)

if __name__ == "__main__":
    a = my_dirichlet(1)
    print(a.get_random_state())
    _alpha = 1
    _p = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    a.test(_alpha, _p, 100)
