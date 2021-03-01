

import numpy as np
import metrics_definition

def count_metrics_for_phi(phi_0, phi_1, simil_fun, num_topics):
    distance = np.zeros((num_topics,num_topics))
    for i in range(num_topics):
        for j in range(num_topics):
            distance[i][j] = simil_fun(phi_0[:,i],phi_1[:,j])
    ax0 = np.argmin(distance, axis = 0)
    ax1 = np.argmin(distance, axis = 1)
    uni0 = np.unique(ax0, return_counts = True)
    uni1 = np.unique(ax1, return_counts = True)
    f1 = pairsim(ax0, ax1)
    f2 = (np.sum((uni0[1] > 1))) # 
    f3 = (np.sum((uni1[1] > 1))) #
    f4 = (num_topics - len(uni0[0]))
    f5 = (num_topics - len(uni1[0]))
    return(f1, f2, f3, f4, f5)

def check_metrics(phi_true, phi_test, num_topics):
    "print стоит потому, что обычно в эксперименте сохраняются сами матрицы, а не метрики"
    "метрики же часто просто просматриваются"
    print(count_metrics_for_phi(phi_true, phi_test, simil_fun_cdist, num_topics))
    print(count_metrics_for_phi(phi_true, phi_test, simil_fun_euclid, num_topics)) 
    print(count_metrics_for_phi(phi_true, phi_test, simil_fun_jenson_shennon, num_topics)) 
    print(count_metrics_for_phi(phi_true, phi_test, simil_fun_hellinger, num_topics))
    return 0
    
def get_metrics(phi_true, phi_test, num_topics):
    metrics = []
    metrics.append(count_metrics_for_phi(phi_true, phi_test, simil_fun_cdist, num_topics))
    metrics.append(count_metrics_for_phi(phi_true, phi_test, simil_fun_cdist, num_topics))
    metrics.append(count_metrics_for_phi(phi_true, phi_test, simil_fun_cdist, num_topics))
    metrics.append(count_metrics_for_phi(phi_true, phi_test, simil_fun_cdist, num_topics))
    return metrics
    

