# Pymarket

Pymarket is the implementation of the Bornholdt Ising Model in Python. All
background information can be found at the
[main page](https://github.com/kenokrieger/multising).

## Usage

To run simulations with this script, execute **main.py** in the source
directory.

The program expects a file "multising.conf" in the directory it is called from.
This file contains all the values for the simulation like grid size and parameters.
The path to the file can also be passed as the first argument in the terminal.

Example configuration file:

```
grid_height = 512
grid_width = 512
grid_depth = 512
total_updates = 100000
seed = 2458242462
alpha = 15.0
j = 1.0
beta = 0.6
init_up = 0.5
```

The parameter **grid_depth** is only needed for 3 dimensional lattices.
For **alpha = 0** this model resembles the standard Ising model.

## License

This project is licensed under MIT License (see LICENSE.txt).
