import numpy as np
import itertools



def get_ecost(net_probs, network, min_centers):
    n = network.shape[0]
    z = network.shape[1]
    dis = np.zeros((n, z))
    for i in range(n):
        dis[i] = np.sqrt(np.sum((network[i]-min_centers[i])**2, axis=1))
    cost = 0.0
    for i1 in range(n):
        for j1 in range(z):
            point_cost = dis[i1][j1] * net_probs[i1][j1]
            for i2 in range(n):
                if i2 != i1:
                    p = 0.0
                    for j2 in range(z):
                        if dis[i1][j1] > dis[i2][j2]:
                            p += net_probs[i2][j2]
                    point_cost *= p
                    print(point_cost, p)
            cost += point_cost
    return cost

def find_center(probs, points, centers):
    min_center = None
    min_dis = np.inf
    for i in range(centers.shape[0]):
        center = centers[i]
        dis_vector = np.sqrt(np.sum((points-center)**2, axis=1))
        dis = dis_vector.dot(probs)
        if dis < min_dis:
            min_center = center
            min_dis = dis
    return min_center


def exact_kcenter(net_probs, network, k):
    n = network.shape[0]
    z = network.shape[1]
    min_cost = np.inf
    min_center = None
    candidates = set([(i,j) for i in range(n) for j in range(z)])
    for subset in itertools.combinations(candidates, k):
        centers = np.array(list(subset))
        min_centers = np.zeros((n,0))
        for i in range(n):
            min_centers[i] = find_center(net_probs[i], network[i], centers)
        cost = get_ecost(net_probs, network, min_center)
        if cost < min_cost:
            min_center = min_centers
            min_cost = cost
    return min_center

def run():
    print(get_ecost(np.array([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]),
        np.array([[[0,0], [1,1], [2,2], [3,3]], [[1,1], [2,2], [3,3], [4,4]]]),
        np.array([[0, 0], [1,1]])))

if __name__ == '__main__':
    run()
