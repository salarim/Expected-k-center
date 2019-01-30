import numpy as np
import pandas as pd
import itertools


class Grid:
    def __init__(self, name, min_x, max_x, min_y, max_y, last_x, last_y):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.last_x = last_x
        self.last_y = last_y
        self.name = name
        self.points = list()
        self.freq = 0
        self.prob = 0.0
        
    def get_center(self):
        if self.freq == 0:
            return [(self.min_x+self.max_x)/2, (self.min_y+self.max_y)/2]
        return [sum(i)/len(self.points) for i in zip(*self.points)]
    
    def is_in(self, lat, lng):
        if lat < self.min_x or not self.last_x and lat >= self.max_x or self.last_x and lat > self.max_x:
            return False
        if lng < self.min_y or not self.last_y and lng >= self.max_y or self.last_y and lng > self.max_y:
            return False
        return True

def convert_pokemon_data(path, n, t):
    data = pd.read_csv(path)[['name', 'lat', 'lng']]
    names = pd.unique(data[['name']].values.ravel('K'))[:n]
    data = data[data['name'].isin(names)]
    min_lat, max_lat, min_lng, max_lng = min(data['lat']), max(data['lat']), min(data['lng']), max(data['lng'])
    
    lats = []
    for i in range(t):
        lats.append(min_lat + ((max_lat-min_lat)/t)*i)
    lats.append(max_lat)
    lngs = []
    for i in range(t):
        lngs.append(min_lng + ((max_lng-min_lng)/t)*i)
    lngs.append(max_lng)
    
    grids = {}
    for name in names:
        grids[name] = []
        for i in range(len(lats)-1):
            for j in range(len(lngs)-1):
                last_x = (i == len(lats)-2)
                last_y = (j == len(lngs)-2)
                grids[name].append(Grid(name, lats[i], lats[i+1], lngs[j], lngs[j+1], last_x, last_y))
                
    cnt = data.groupby('name').size()
    for _, row in data.iterrows():
        for grid in grids[row['name']]:
            if grid.is_in(row['lat'], row['lng']):
                grid.freq += 1
                grid.points.append([row['lat'], row['lng']])
                break
    
    for name in grids.keys():
        for grid in grids[name]:
            grid.prob = (float)(grid.freq) / cnt[name]
            
    net_probs = np.zeros((n, t*t))
    network = np.zeros((n, t*t, 2))
    for i in range(n):
        name = names[i]
        for j in range(t*t):
            grid = grids[name][j]
            net_probs[i][j] = grid.prob
            center = grid.get_center()
            network[i][j][0] = center[0]
            network[i][j][1] = center[1]
    
    return net_probs, network
        
    def get_center(self):
        if self.freq == 0:
            return [(self.min_x+self.max_x)/2, (self.min_y+self.max_y)/2]
        return [sum(i)/len(self.points) for i in zip(*self.points)]
    
    def is_in(self, lat, lng):
        if lat < self.min_x or not self.last_x and lat >= self.max_x or self.last_x and lat > self.max_x:
            return False
        if lng < self.min_y or not self.last_y and lng >= self.max_y or self.last_y and lng > self.max_y:
            return False
        return True

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
                        if dis[i1][j1] > dis[i2][j2] or dis[i1][j1] == dis[i2][j2] and i1 < i2:
                            p += net_probs[i2][j2]
                    point_cost *= p
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
    return min_center, min_dis


def exact_kcenter(net_probs, network, k):
    n = network.shape[0]
    z = network.shape[1]
    min_cost = np.inf
    min_center = None
    candidates = set([(i,j) for i in range(n) for j in range(z)])
    for subset in itertools.combinations(candidates, k):
        centers = np.zeros((len(subset), 2))
        for i in range(len(subset)):
            centers[i] = network[subset[i][0]][subset[i][1]]
        min_centers = np.zeros((n,2))
        for i in range(n):
            min_centers[i], _ = find_center(net_probs[i], network[i], centers)
        cost = get_ecost(net_probs, network, min_centers)
        if cost < min_cost:
            min_center = centers
            min_cost = cost
    return min_center

def get_expected_point(probs, points):
    return probs.dot(points)

def approx_kcenter(net_probs, network, k):
    n = network.shape[0]
    z = network.shape[1]
    centers = np.array([get_expected_point(net_probs[0], network[0])])
    while centers.shape[0] < k:
        max_dis = -1.0
        max_index = None
        for i in range(n):
            _, min_dis = find_center(net_probs[i], network[i], centers)
            if min_dis > max_dis:
                max_index = i
                max_dis = min_dis
        new_center = get_expected_point(net_probs[max_index], network[max_index])
        centers = np.vstack([centers, new_center])
    return centers

def test(net_probs, network, k):
    exact_center = exact_kcenter(net_probs, network, k)
    approx_center = approx_kcenter(net_probs, network, k)
    n = network.shape[0]

    exact_min_center = np.zeros((n,2))
    for i in range(n):
        exact_min_center[i], _ = find_center(net_probs[i], network[i], exact_center)
    approx_min_center = np.zeros((n,2))
    for i in range(n):
        approx_min_center[i], _ = find_center(net_probs[i], network[i], approx_center)

    exact_cost = get_ecost(net_probs, network, exact_min_center)
    approx_cost = get_ecost(net_probs, network, approx_min_center)

    return exact_center, approx_center, exact_cost, approx_cost


def run():
    # net_probs = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    # network = np.array([[[0,0], [3,3]], [[2,1], [1,3]], [[3,1], [0,3]]])

    net_probs, network = convert_pokemon_data('pokemon-spawns.csv', 2, 2)

    print(test(net_probs, network, 2))

if __name__ == '__main__':
    run()
