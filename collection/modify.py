
import numpy as np
from scipy.stats import zipf

def modify_theta(theta, Dt_array, num_docs): 
    "балансировка тем, Dt_array[i] - число документов с преобладающей темой i"
    arr = np.arange(num_docs)
    np.random.shuffle(arr)
    current = 0
    for i in range(len(Dt_array)):
        for j in range(Dt_array[i]):
            x = np.argmax(theta[:, arr[current]])
            theta[x,arr[current]],theta[i,arr[current]] = theta[i,arr[current]],theta[x,arr[current]]
            current = current + 1
    return theta

def add_general(phi, theta, num_docs, num_words, deg = 0.5):
    "добавляет фоновую тему, deg - её вероятность появления в каждом документе, 0 < deg < 1"
    "num_docs и num_words обычно совпадают с соответствующими размерами phi, theta"
    gen_topic = zipf.rvs(1.01, size = num_words, random_state = 1)
    gen_topic = gen_topic / np.sum(gen_topic)
    phi = np.append(phi, gen_topic.reshape(num_words, 1), axis = 1)
    theta = theta * (1 - deg)
    theta = np.append(theta, (np.ones(num_docs) * deg).reshape(1, num_docs), axis = 0)
    return (phi, theta)
