#! /usr/bin/env python3

from traders import init_traders, update
from datetime import datetime
from numpy import empty, nan, savetxt
from sys import argv

MAX_FILE_SIZE = 100_000


def read_config_file(filename):
    """
    Read the parameters from the configuration file.

    :param filename: The location of the configuration file.
    :type filename: str or path

    :return: The parameters from the configuration file stored in a dictionary.
    :rtype: dict

    """
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


def main():
    """Main function of the script"""

    # Read the configuration file
    if len(argv) > 1:
        config_filename = argv[1]
    else:
        config_filename = "multising.conf"
    config = read_config_file(config_filename)

    # Cast the configuration input to the correct types
    grid_height = int(config["grid_height"])
    grid_width = int(config["grid_width"])
    alpha = float(config["alpha"])
    j = float(config["j"])
    total_updates = int(config["total_updates"])
    init_up = float(config["init_up"])
    beta = float(config["beta"])
    reduced_alpha = -2 * beta * alpha
    reduced_neighbour_coupling = -2 * beta * j

    # assign magnetisation invalid values that will be overwritten during the
    # update
    magnetisation = empty((min(total_updates, MAX_FILE_SIZE), ), dtype=float)
    magnetisation[:] = nan

    shape = (grid_height, grid_width)
    black, white = init_traders(shape, init_up=init_up)

    start = datetime.now()
    for ii in range(total_updates):
        magnetisation[ii % MAX_FILE_SIZE] = update(
            black, white, reduced_neighbour_coupling, reduced_alpha
        )

        if ii and not ii % MAX_FILE_SIZE:
            savetxt(f"magnetisation_{ii}.dat", magnetisation)
            magnetisation[:] = nan
    elapsed_time = (datetime.now() - start)
    savetxt(f"magnetisation_{ii}.dat", magnetisation)
    flips_per_ns = total_updates * (grid_width * grid_height)
    flips_per_ns /= (elapsed_time.seconds * 1e9 + elapsed_time.microseconds * 1e3)

    print("Computation time: {}.{}".format(
        elapsed_time.seconds, elapsed_time.microseconds)
    )
    print("Spin updates per nanosecond: {:.4E}".format(flips_per_ns))


if __name__ == "__main__":
    main()
