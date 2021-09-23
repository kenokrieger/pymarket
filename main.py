#! /usr/bin/env python3

from traders import init_traders, update
from datetime import datetime
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states


def read_config_file(filename):
    config = dict()
    with open(filename, 'r') as f:
        args = f.readlines()
        for arg in args:
            if arg == '\n' or arg[0] == '#':
                continue
            else:
                key, value = arg.split("=")
                config[key.strip()] = value.strip()
    return config


if __name__ == "__main__":
    config = read_config_file("pymarket.conf")
    grid_height = int(config["grid_height"])
    grid_width = int(config["grid_width"])
    grid_depth = int(config["grid_depth"])
    shape = (grid_height, grid_width // 2, grid_depth)

    for param in shape:
        if (param % 8):
            print("Please specify grid dimensions to be multiple of 8")
            exit()

    black = np.ones(shape, dtype=np.int32)
    d_black = cuda.to_device(black)
    white = np.ones(shape, dtype=np.int32)
    d_white = cuda.to_device(white)


    alpha = float(config["alpha"])
    j = float(config["j"])
    beta = float(config["beta"])
    total_updates = int(config["total_updates"])
    seed = int(config["seed"])

    reduced_alpha = -2 * beta * alpha
    reduced_neighbor_coupling = -2 * beta * j

    threads_per_block = (8, 8, 8)
    blocks = (16, 16, 16)
    total_number_of_threads = (8 ** 3) * (16 ** 3)
    rng_states = create_xoroshiro128p_states(total_number_of_threads, seed=seed)
    magnetisation = np.empty((total_updates, ))

    init_traders[blocks, threads_per_block](True, rng_states, d_black, shape, 0.5)
    init_traders[blocks, threads_per_block](False, rng_states, d_white, shape, 0.5)

    start = datetime.now()
    for iteration in range(total_updates):
        global_market = update(rng_states, d_black, d_white, reduced_neighbor_coupling, reduced_alpha, shape)
        print(global_market)
        magnetisation[iteration] = global_market

    elapsed_time = (datetime.now() - start)
    flips_per_ns = total_updates * (grid_depth * grid_width * grid_height) / (elapsed_time.seconds * 1e9 + elapsed_time.microseconds * 1e3)
    print("time: {}.{}".format(elapsed_time.seconds, elapsed_time.microseconds))
    print("Spin updates per nanosecond: {:.4E}".format(flips_per_ns))
    np.savetxt("magnetisation.dat", magnetisation / (grid_width * grid_height * grid_depth))
    with open("log", "w") as f:
        f.write(f"updates/ns: {flips_per_ns}\n")
    cuda.close()
