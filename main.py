#! /usr/bin/env python3

from traders import init_traders, update
from datetime import datetime


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

    alpha = float(config["alpha"])
    j = float(config["j"])
    beta = float(config["beta"])
    total_updates = int(config["total_updates"])

    reduced_alpha = -2 * beta * alpha
    reduced_neighbor_coupling = -2 * beta * j
    shape = (grid_height, grid_width, grid_depth)
    black, white = init_traders(shape)

    start = datetime.now()
    for iter in range(total_updates):
        global_market = update(black, white, reduced_neighbor_coupling, reduced_alpha)
        print(global_market)

    elapsed_time = (datetime.now() - start)
    flips_per_ns = total_updates * (grid_depth * grid_width * grid_height) / (elapsed_time.seconds * 1e9 + elapsed_time.microseconds * 1e3)
    print("time: {}.{}".format(elapsed_time.seconds, elapsed_time.microseconds))
    print("Spin updates per nanosecond: {:.4E}".format(flips_per_ns))
