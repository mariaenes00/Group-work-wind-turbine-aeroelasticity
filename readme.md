# Example Python code for Exercise 1

## Installation

The guide below describes how to install and run the code through a command line interface.

1. Clone the repository. E.g., change your working directory in a command line interface to where you want to create a new directory with this code and then `git clone https://github.com/DTUWind-46310-2026/Python-Example`.
2. From the same command line interface, create a virtual environment using `python -m venv .venv`.
3. Again from the command line interface, activate the environment by `.\.venv\Scripts\activate`.
4. Install the requirements by `pip install -r .\requirements.txt`
5. Test the installation by running `python 1.py`.

## Documentation

This briefly outlines some of the choices made. The code structure is based on the slides `coding_structure` from the class. File `1.py` has solutions to the exercise 1; use this file to for a top-level understanding of how to run the simulations.

### `structure.py`

The coordinate systems are slightly different compared to what we learned in class. This should probably be updated. Currently, the coordinate systems are:

1. Ground reference.
2. In the hub, yawed with respect to 1.
3. At the end of the shaft, tilted with respect to 2.
4. At the base of one of the blades, rotated in the azimuth to the corresponding blade with respect to 3.
5. Along the blade, coned with respect to 4.

The addition here is the coordinate system between 3 and 5 that selects a single blade. In class, this is skipped and we go straight to the blades.

### `recorders.py`

This pre-defines some functionalties to record data during a simulation. Use the `Recorder` class to define your custom recorders. Some examples (your custom ones don't have to be classes) are:

1. `BladePosition1Recorder`: Record the position of a blade element in coordinate system 1.
2. `BladeVelocity5Recorder`: Record the blade velocity (without wind!) of a blade element in coordinate system 5.
3. `Wind5Recorder`: Record the wind velocity of a blade element in coordinate system 5.

## Scary Python code

If you look under the hood, some code might seem daunting. Here's what the (probably) scariest code does:

### `@abstractmethod`

This is used in parent classes to specify functions that the children classes need to implement. This is very useful to define a blueprint. If some code somewhere receives a child class of said parent class, the code knows for certain that the child class has the functions that were specified by `@abstractmethod`.

Functions that have this `@abstractmethod` do not do anything themselves; the children define what they do. However, the define that they need to be there and what they receive.

### `__init__(self, ...)`

If you have a class

```python
class MyClass:
    def __init__(self, ...):
        <some code>
```

then this `__init__()` function is used if you run `MyClass(...)`.

```python
my_object = MyClass(...)
```

### `__call__(self, ...)`

Similar to `__init__()`, but this time for the instance `my_object`. I.e., `__call__()` is used when you do

```python
my_object(...)
```

### `self`

Coming later today.
