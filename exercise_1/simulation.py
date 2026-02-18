from pathlib import Path

import numpy as np
import pandas as pd

from recorder import Recorder, time_recorder
from structure import Structure
from wind import NoWind, Wind


class Simulation:

    def __init__(
        self,
        structure: Structure,
        wind: Wind = NoWind(),
        recorders: Recorder | list[Recorder] | None = None,
    ) -> None:
        """
        Creates a simulation instance.

        Parameters
        ----------
        structure : Structure
            The structure instance.
        wind : Wind, optional
            The wind instance., by default NoWind()
        recorders : Recorder | list[Recorder] | None, optional
            Any number of recorders. By default, a recorder is added that saves the times of the simulation.
        """
        self.structure = structure
        self.wind = wind
        self.model_parts = [self.wind, self.structure]
        self.time = 0
        self.dt = 0
        self.step_idx = 0

        recorders = recorders or []
        self.recorders = recorders if isinstance(recorders, list) else [recorders]

    def run(self, dt: float, T: float):
        """
        Run the simulation.

        Parameters
        ----------
        dt : float
            Time step duration.
        T : float
            Time the simulation runs for.
        """
        self.recorders.append(time_recorder())
        self.dt = dt

        n_sim_steps = int(T / dt)
        for recorder in self.recorders:
            recorder.update_n_steps(n_sim_steps)

        for step_idx in range(n_sim_steps):
            self.step_idx = step_idx

            for recorder in self.recorders:
                recorder(self)

            for part in self.model_parts:
                part.step(self)

            self.time += dt

    def get_recorders(self) -> dict[str, dict[str, tuple[str, ...] | np.ndarray]]:
        """
        Returns the data of all the recorders.


        Returns
        -------
        dict[str, dict[str, tuple[str, ...] | np.ndarray]]
            Dictionary with format {<recorder_name>: {"dims": <dimension names of data>, "values": <data of recorder>}}
        """
        return {rec.name: {"dims": rec.func_returns, "values": rec.data} for rec in self.recorders}

    def save_recorders(self, root: str | Path, case_name="", overwrite=False):
        """
        Save the data of the recorders to files in the `root` directory. The files will have the names
        `<recorder_name><case_name>.csv`.

        Parameters
        ----------
        root : str | Path
            The directory into which the files will be saved.
        case_name : str, optional
            What to append to the file name., by default ""
        overwrite : bool, optional
            Whether or not to overwrite if the file exists already, by default False
        """
        recorders = self.get_recorders()
        time = {"time": recorders.pop("time")["values"]}
        (_r := Path(root)).mkdir(parents=True, exist_ok=True)
        for rec_name, data in recorders.items():
            if (save_to := (_r / (rec_name + f"{case_name}.csv"))).is_file() and not overwrite:
                print(f"Skipping '{save_to.as_posix()}' because it already exists and 'overwrite=False'")
                continue
            dims = data["dims"]
            values = data["values"] if len(data["values"].shape) > 1 else np.atleast_2d(data["values"]).T
            pd.DataFrame(time | {dim: values[:, i] for i, dim in enumerate(dims)}).to_csv(save_to, index=False)


if __name__ == "__main__":
    from structure import RigidStructure
    from wind import ShearWind

    sim = Simulation(
        RigidStructure(0.62),
        ShearWind(119, 10, 0.2),
    )
    sim.run(0.1, 30)
    sim.save_recorders("sim_data", overwrite=True)
