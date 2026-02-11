import matplotlib.pyplot as plt
import numpy as np

from recorder import BladePosition1Recorder, BladeVelocity5Recorder, Wind5Recorder
from simulation import Simulation
from structure import RigidStructure
from wind import ConstantWind, ShearWind, WindWithTower

do = {
    "1": True,
    "2": True,
    "3": True,
    "4": True,
    "5": True,
}


if do["1"]:
    dt = 0.15
    T = 30
    omega_init = 0.62
    structure = RigidStructure(omega_init)

    recorder_name = "blade_pos_part1"
    blade_pos_recorder = BladePosition1Recorder(name=recorder_name, blade_idx=0, element_idx=10)
    simulation = Simulation(structure, recorders=blade_pos_recorder)
    simulation.simulate(dt, T)
    simulation.save_recorders("sim_data", overwrite=True)

    data = simulation.get_recorders()
    fig, ax = plt.subplots()
    ax.plot(data[recorder_name]["values"][:, 1], data[recorder_name]["values"][:, 0])
    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")
    ax.set_aspect("equal")

    plt.show()

if do["2"]:
    dt = 0.15
    T = 30
    omega_init = 0.62

    fig, ax = plt.subplots()
    for yaw in [0, 20]:
        structure = RigidStructure(omega_init, yaw=yaw)

        recorder_name = "blade_pos_part2"
        blade_pos_recorder = BladePosition1Recorder(name=recorder_name, blade_idx=0, element_idx=10)
        simulation = Simulation(structure, recorders=blade_pos_recorder)
        simulation.simulate(dt, T)

        data = simulation.get_recorders()
        ax.plot(data[recorder_name]["values"][:, 1], data[recorder_name]["values"][:, 0], label=f"{yaw=}$\degree$")

    simulation.save_recorders("sim_data", overwrite=True)
    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")
    ax.legend()
    ax.set_aspect("equal")

    plt.show()

if do["3"]:
    omega_init = 0.62
    T = 2 * np.pi / omega_init
    dt = T / 100

    fig, ax = plt.subplots()
    for yaw in [0, 20]:
        structure = RigidStructure(omega_init, yaw=yaw)

        shear_wind = ShearWind(119, 10, 0.2)
        recorder_name = "vel_recorder_part3"
        vel_recorder = BladeVelocity5Recorder(name=recorder_name, blade_idx=0, element_idx=10)

        simulation = Simulation(structure, shear_wind, vel_recorder)
        simulation.simulate(dt, T)

        data = simulation.get_recorders()
        azimuth = data["time"]["values"] * omega_init / (2 * np.pi) * 360
        ax.plot(azimuth, data[recorder_name]["values"][:, 1], label=f"{yaw=}$\degree$, $V_y$")
        ax.plot(azimuth, data[recorder_name]["values"][:, 2], label=f"{yaw=}$\degree$, $V_z$")

    simulation.save_recorders("sim_data", overwrite=True)
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend()

    plt.show()

if do["4"]:
    omega_init = 0.62
    T = 2 * np.pi / omega_init
    dt = T / 300

    structure = RigidStructure(omega_init)

    tower_radius = np.asarray([[0, 3.32], [119, 3.32]])
    shear_wind = WindWithTower(0, 0, tower_radius, ConstantWind(10))
    recorder_name = "vel_recorder_part4"
    vel_recorder = Wind5Recorder(name=recorder_name, blade_idx=0, element_idx=10)

    simulation = Simulation(structure, shear_wind, vel_recorder)
    simulation.simulate(dt, T)
    simulation.save_recorders("sim_data", overwrite=True)

    data = simulation.get_recorders()
    azimuth = data["time"]["values"] * omega_init / (2 * np.pi) * 360

    fig, ax = plt.subplots()
    ax.plot(azimuth, data[recorder_name]["values"][:, 1], label="$V_y$")
    ax.plot(azimuth, data[recorder_name]["values"][:, 2], label="$V_z$")

    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend()

    plt.show()


if do["5"]:
    # Some needed values
    omega_init = 0.62
    T = 2 * np.pi / omega_init
    dt = T / 200

    # Define structure
    structure = RigidStructure(omega_init, yaw=-20, tilt=-5)

    # Define wind with tower effect
    tower_radius = np.asarray(  # columns are [x, tower radius]
        [
            [0, 3.32],
            [119, 3.32],
        ]
    )
    shear_wind = WindWithTower(y_tower=0, z_tower=0, xa=tower_radius, surrounding_wind=ShearWind(119, 10, 0.2))

    # Define recorder to save wind as seen in blade coordinate system (5)
    recorder_name = "wind_recorder_part5"
    wind_recorder = Wind5Recorder(name=recorder_name, blade_idx=0, element_idx=10)

    # Set up simulation, run, and save wind recorder data
    simulation = Simulation(structure, shear_wind, wind_recorder)
    simulation.simulate(dt, T)
    simulation.save_recorders("sim_data", overwrite=True)

    # Get data (saving above not needed for this) for plotting
    data = simulation.get_recorders()
    azimuth = data["time"]["values"] * omega_init / (2 * np.pi) * 360

    # Plot
    fig, ax = plt.subplots()
    ax.plot(azimuth, data[recorder_name]["values"][:, 1], label="$V_y$")
    ax.plot(azimuth, data[recorder_name]["values"][:, 2], label="$V_z$")

    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend()

    plt.show()
