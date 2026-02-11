"""
Recorder class for storing time-series data during simulation.
"""

from collections.abc import Callable

import numpy as np


class Recorder:
    """
    Records simulation data in a pre-allocated numpy array.

    The recorder allocates memory based on dt and T.
    """

    def __init__(self, func: Callable, name: str, func_returns: tuple[str, ...] | str):
        """
        Create a recorder instance.

        Example
        ----------
        Wanted: Position of blade element at index 10 in coordinate system 1.
        First: Write a function that receives `simulation` and returns the position:
        >>> def get_blade_pos_in_1(simulation):
        >>>     return simulation.structure.blade_x1(blade_idx=0)[10]

        Then: Create the recorder
        >>> pos_recorder = Recorder(get_blade_pos_in_1, "position_recorder", ("x", "y", "z"))

        Where "position_recorder" becomes the name of the recorder (when you use `simulation.get_recorders()`) and
        `("x", "y", "z")` are the coordinates that the `get_blade_pos_in_1()` returns.

        This example is already implemented as the `BladePosition1Recorder`.

        Parameters
        ----------
        func : Callable
            A function that receives only `simulation` as input and returns a 1D list or 1D numpy array of values.
        name : str
            The name for the recorded data. Important when using `simulation.get_recorders()`.
        func_returns : tuple[str, ...] | str
            Specify what the `func` returns, i.e., if it returns a xyz position, `func_returns = ("x", "y", "z")`.
        """
        self.func = func
        self.name = name
        self.func_returns = func_returns if isinstance(func_returns, tuple) else (func_returns,)
        self._data = np.ndarray((0,))
        self._steps_udpated = False

    def update_n_steps(self, n_steps: int):
        self._data = np.zeros((n_steps, len(self.func_returns)))
        self._steps_udpated = True

    def __call__(self, simulation):
        if not self._steps_udpated:
            raise RuntimeError(f"Need to use `update_n_steps` before using the recorder '{self.name}'.")
        self._data[simulation.step_idx] = self.func(simulation)

    @property
    def data(self) -> np.ndarray:
        return self._data.squeeze()


class TimeRecorder(Recorder):

    def __init__(self, n_steps):
        def time(simulation):
            return simulation.time

        super().__init__(time, "time", ("time",))
        self.update_n_steps(n_steps)


class BladePosition1Recorder(Recorder):

    def __init__(self, name: str, blade_idx: int, element_idx: int):
        def blade_pos(simulation):
            return simulation.structure.blade_x1(blade_idx)[element_idx]

        super().__init__(blade_pos, name, ("x", "y", "z"))


class BladeVelocity5Recorder(Recorder):

    def __init__(self, name: str, blade_idx: int, element_idx: int | None = None):

        def blade_rel_vel(simulation):
            vel5 = simulation.structure.blade_u5(blade_idx)[element_idx]

            blade_pos1 = simulation.structure.blade_x1(blade_idx)[element_idx]
            wind1 = simulation.wind(blade_pos1)
            wind5 = simulation.structure.x15(wind1, blade_idx)
            return vel5 + wind5

        super().__init__(blade_rel_vel, name, ("u", "v", "w"))


class Wind5Recorder(Recorder):

    def __init__(self, name, blade_idx: int, element_idx: int):

        def wind5(simulation):
            blade_pos1 = simulation.structure.blade_x1(blade_idx)[element_idx]
            wind1 = simulation.wind(blade_pos1)
            return simulation.structure.x15(wind1, blade_idx)

        super().__init__(wind5, name, ("u", "v", "w"))
