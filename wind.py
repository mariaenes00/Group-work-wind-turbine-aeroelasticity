from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import interp1d


class Wind(ABC):
    """
    Base (parent) class for the wind. This is not supposed to be used during the simulations (and also doesn't do
    anything). Using the @abstractmethod line defines which methods the children classes need to implement.
    """

    @abstractmethod
    def __call__(self, xyz) -> np.ndarray:
        pass

    @abstractmethod
    def step(self, simulation) -> None:
        pass


class ConstantWind(Wind):
    def __init__(self, ws: float) -> None:
        """
        Initialises a wind instance that returns a constant wind speed everywhere.

        Parameters
        ----------
        ws : float
            The wind speed.
        """
        self.ws = ws

    def __call__(self, xyz):
        xyz = np.atleast_2d(xyz)
        return (np.c_[np.zeros_like(xyz[:, 0]), np.zeros_like(xyz[:, 0]), np.full_like(xyz[:, 0], self.ws)]).squeeze()

    def step(self, simulation):
        # Nothing needs to happen here; the wind speed simply stays constant everywhere.
        pass


class NoWind(ConstantWind):
    def __init__(self) -> None:
        """
        Initialises an instance that returns a wind speed of zero everywhere.
        """
        super().__init__(0)


class ShearWind(Wind):
    def __init__(self, x_ref: float, v_ref: float, exponent: float) -> None:
        """
        Initialises an instance that returns wind speeds based on the defined shear.

        Parameters
        ----------
        x_ref : float
            The x coordinate at which the reference wind speed `v_ref` is defined.
        v_ref : float
            The reference wind speed at height `x_ref`
        exponent : float
            The exponent of for the shear.
        """
        self.shear = lambda x: v_ref * (x / x_ref) ** exponent

    def __call__(self, xyz):
        xyz = np.atleast_2d(xyz)
        return np.c_[np.zeros_like(xyz[:, 0]), np.zeros_like(xyz[:, 0]), self.shear(xyz[:, 0])]

    def step(self, simulation):
        # Nothing needs to happen here either.
        pass


class WindWithTower(Wind):

    def __init__(self, y_tower: float, z_tower: float, xa: np.ndarray, surrounding_wind: Wind) -> None:
        """
        Initialises and instance that returns wind speeds based on `surrounding_wind` including the
        tower effect.

        Example
        ----------
        To use a shear with `x_ref=119`, `u_ref=10`, `exponent=0.2` that includes the tower effect (tower at
        `y_tower=0`, `z_tower=0`, with a constant radius of 3.32m from the bottom to the top), do this:

        >>> shear_wind = ShearWind(119, 10, 0.2)
        >>> xa = np.asarray([[0, 3.32], [119, 3.32]])
        >>> shear_with_tower = WindWithTower(0, 0, xa, shear_wind)

        Parameters
        ----------
        y_tower : float
            The y coordinate of the tower base.
        z_tower : float
            The z coordiante of the tower base.
        xa : np.ndarray
            A 2D array of shape (n, 2) defining the radius of the tower over height x. Each row defines
            [x coordinate, radius at this x coordinate].
        surrounding_wind : Wind
            An instance of a wind class that has the `Wind` class as parent class.
        """
        self.centre = np.asarray([y_tower, z_tower])
        self.a = interp1d(xa[:, 0], xa[:, 1], fill_value=(0, 0), bounds_error=False)
        self.surrounding_wind = surrounding_wind

    def __call__(self, xyz):
        xyz = np.atleast_2d(xyz)

        x, y, z = xyz.T
        r = np.linalg.norm(xyz[:, 1:3] - self.centre)
        V_0 = np.atleast_2d(self.surrounding_wind(xyz))[:, 2]
        v_r = z / r * V_0 * (1 - (self.a(x) / r) ** 2)
        v_theta = y / r * V_0 * (1 + (self.a(x) / r) ** 2)

        v_y = y / r * v_r - z / r * v_theta
        v_z = z / r * v_r + y / r * v_theta
        return (np.c_[np.zeros_like(v_y), v_y, v_z]).squeeze()

    def step(self, simulation):
        # Nothing needs to change here :)
        pass
