import numpy as np

class noniid:
    def __init__(self):
        pass
    
    def get_noniid(self, alpha):
        ret = []
        for i in range(10):
            t = np.zeros(10)
            for j in range(10):
                if j == i:
                    t[j] = alpha
                else:
                    t[j] = (1 - alpha) / 9
            for k in range(10):
                ret.append(t)
        return ret

if __name__ == "__main__":
    a = noniid()
    print(a.get_noniid(0.91))
