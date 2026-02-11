from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from rotation import Rotation


class Structure(ABC):
    """
    Base (parent) class for the structure. This is not supposed to be used during the simulations. Using the @abstractmethod line defines which methods the children classes need to implement.

    This class defines some functionalities that are useful for the child classes (RigidStructure and at some
    point a flexible structure).
    """

    def __init__(
        self,
        omega_init=0.0,
        file_blade="data/blade_data.csv",
        hub_height=119.0,
        l_shaft=7.1,
        cone=0.0,
        yaw=0.0,
        tilt=0.0,
        pitch_init=[0, 0, 0],
    ) -> None:
        """
        Sets up some instance variables for the child classes.

        Parameters
        ----------
        omega_init : float, optional
            The initial rotational speed of the rotor, by default 0.0
        file_blade : str, optional
            Path to the file defining the blade structure. The path is expected to be a csv file
            with columns `r,c,twist,rel_thickness` for the radial position `r`, chord `c`, twist `twist`, and
            relative thickness `rel_thickness`, by default "data/blade_data.csv"
        hub_height : float, optional
            Hub height of the wind turbine, by default 119.0
        l_shaft : float, optional
            Length of the shaft, by default 7.1
        cone : float, optional
            Coning of the rotor, by default 0.0
        yaw : float, optional
            Yaw of the rotor, by default 0.0
        tilt : float, optional
            Tilt of the shaft, by default 0.0
        pitch_init : list, optional
            The initial pitch angles for each blade. From this, the number of blades are defined, by default [0, 0, 0]
        """
        r = pd.read_csv(file_blade)["r"].to_numpy()
        self.r = r
        self.R = r[-1]
        self.r_hub = r[0]

        self.hub_height = hub_height
        self.l_shaft = l_shaft
        self.cone = np.deg2rad(cone)
        self.yaw = np.deg2rad(yaw)
        self.tilt = np.deg2rad(tilt)
        self.n_blades = len(pitch_init)
        self.pitch = np.deg2rad(pitch_init)

        self.phi_shaft = 0
        self.omega_shaft = omega_init

        self._x5_blade = np.c_[self.r, np.zeros_like(self.r), np.zeros_like(self.r)]

    @abstractmethod
    def step(self, simulation):
        pass

    def blade_azimuth(self, blade_idx):
        if blade_idx > self.n_blades:
            raise ValueError(f"Structure only has '{self.n_blades}' blades, but {blade_idx=}.")
        return self.phi_shaft + blade_idx * 2 * np.pi / self.n_blades


class RigidStructure(Structure):

    def __init__(
        self,
        omega_init=0.0,
        file_blade="data/blade_data.csv",
        hub_height=119,
        l_shaft=7.1,
        cone=0.0,
        yaw=0.0,
        tilt=0.0,
        pitch_init=[0.0, 0.0, 0.0],
        drive_train_dynamics=False,
    ) -> None:
        """
        Creates an instance for a rigid wind turbine.

        Parameters
        ----------
        omega_init : float, optional
            The initial rotational speed of the rotor, by default 0.0
        file_blade : str, optional
            Path to the file defining the blade structure. The path is expected to be a csv file
            with columns `r,c,twist,rel_thickness` for the radial position `r`, chord `c`, twist `twist`, and
            relative thickness `rel_thickness`, by default "data/blade_data.csv"
        hub_height : float, optional
            Hub height of the wind turbine, by default 119.0
        l_shaft : float, optional
            Length of the shaft, by default 7.1
        cone : float, optional
            Coning of the rotor, by default 0.0
        yaw : float, optional
            Yaw of the rotor, by default 0.0
        tilt : float, optional
            Tilt of the shaft, by default 0.0
        pitch_init : list, optional
            The initial pitch angles for each blade. From this, the number of blades are defined, by default [0, 0, 0]
        drive_train_dynamics : bool, optional
            Whether or not to include drive train dynamics, by default False
        """
        super().__init__(omega_init, file_blade, hub_height, l_shaft, cone, yaw, tilt, pitch_init)

        self.drive_train_dynamics = drive_train_dynamics

    def step(self, simulation):
        """
        Advances the structure one time step.

        Parameters
        ----------
        simulation : Simulation
            The simulation object

        Raises
        ------
        NotImplementedError
            Drive train dynamics are not yet implemented.
        """
        if not self.drive_train_dynamics:
            self.phi_shaft += self.omega_shaft * simulation.dt
        else:
            raise NotImplementedError("You'll have to implement the drive train dynamcis at some point :)")

    def blade_x1(self, blade_idx: int) -> np.ndarray:
        """
        Returns the coordinates of blade number `blade_idx` in the coordinate system 1.

        Parameters
        ----------
        blade_idx : int
            Index of blade.

        Returns
        -------
        np.ndarray
            The coordinates of the blade in coordinate system 1 as [x, y, z].
        """
        x4_blade = Rotation.rotate_3d_y(self._x5_blade, self.cone)
        x3_blade = Rotation.rotate_3d_z(x4_blade, self.blade_azimuth(blade_idx))
        x2_blade = Rotation.rotate_3d_y(x3_blade + np.asarray([0, 0, -self.l_shaft]), self.tilt)
        return Rotation.rotate_3d_x(x2_blade + np.asarray([self.hub_height, 0, 0]), self.yaw)

    def blade_u5(self, blade_idx: int) -> np.ndarray:
        """
        The velocities only due to the motion of the blade in the blade coordinate system.

        Parameters
        ----------
        blade_idx : int
            Blade index for which to get the velocities.

        Returns
        -------
        np.ndarray
            Velocities as numpy array as [u, v, w] in coordinate system 5.
        """
        wr = self.omega_shaft * self.r
        v = np.cos(self.pitch[blade_idx]) * wr
        w = np.sin(self.pitch[blade_idx]) * wr
        return np.c_[np.zeros_like(self.r), v, w]

    def x15(self, array: np.ndarray, blade_idx: int) -> np.ndarray:
        """
        Transforms an array from coordinate system 1 into the blade coordinate system 5.

        Parameters
        ----------
        array : np.ndarray
            The array with shape (n, 3) where each row is in the directions [x, y, z]
        blade_idx : int
            Blade index.

        Returns
        -------
        np.ndarray
            The transformed array in the blade coordinate system.
        """
        x2 = Rotation.rotate_3d_x(array, -self.yaw)
        x3 = Rotation.rotate_3d_y(x2, -self.tilt)
        x4 = Rotation.rotate_3d_z(x3, -self.blade_azimuth(blade_idx))
        return Rotation.rotate_3d_y(x4, -self.cone)


if __name__ == "__main__":
    wt_structure = RigidStructure(yaw=0, tilt=0, cone=90)
    # print(wt_structure.blade_azimuth(0))
    # print(wt_structure.blade_azimuth(1))
    # print(wt_structure.blade_azimuth(2))
    # print(wt_structure.blade_x1(0))
    # print(wt_structure.blade_x1(1))
    # print(wt_structure.blade_x1(2))

    wind = np.asarray([0, 0, 10])
    print(wt_structure.x15(wind, 0))
