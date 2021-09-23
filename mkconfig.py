#!/usr/bin/env python3
from random import randrange
from numpy import arange
from sys import argv

TSTART = 0.2
TEND = 0.6
TSTEP = 0.02
TESTRUNS = 5


if __name__ == "__main__":
    with open("pymarket.conf", "r") as f:
        multising_config = f.read()

    for run in range(TESTRUNS):
        seed = f"\nseed = {randrange(1, 1000000)}"
        for temp in arange(TSTART, TEND, TSTEP):
            config_temp = "\nbeta = {}".format(temp)

            file_id = "{}/run={}_temp={:.3f}.conf".format(argv[1], run, temp)
            with open(file_id, "w") as f:
                f.write(multising_config + seed + config_temp)
