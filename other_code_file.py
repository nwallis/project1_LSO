from provided_functions import *
import numpy as np

instance_path = "./instances/VLSI_8_6_7.json"

instance = load_instance(instance_path)

MAX_ITER = 1

## Implementing Algorithm 1:

def route_cost(u_mat, route):
    if route is None:
        return np.inf
    else:
        val = 0
        for node in route:
            val += u_mat[node]
        return val

# Initializations:
n = instance.width * instance.height * 2
u_mat = np.ones((instance.width, instance.height, 2)) / n
t = 0
while t < MAX_ITER:
    routes = cheapest_grid_routes(instance.width, instance.height, u_vec, instance.contact_pairs)
    
    t += 1