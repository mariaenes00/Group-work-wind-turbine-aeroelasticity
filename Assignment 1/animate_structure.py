from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d' projection)
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from rotation import Rotation
from datetime import datetime

plt.rcParams.update(
    {
        "axes.labelsize": 14,
        "figure.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)


def _load_state(state_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    df = pd.read_csv(state_csv)
    required_fixed = {"time", "x_t", "phi_shaft"}
    missing = required_fixed - set(df.columns)
    if missing:
        raise ValueError(f"State CSV {state_csv} is missing columns: {sorted(missing)}")

    q_cols = [c for c in df.columns if c.startswith("q_b") and "_m" in c]
    if len(q_cols) % 3 != 0:
        raise ValueError(f"State CSV {state_csv} has {len(q_cols)} q_b*_m* columns; expected a multiple of 3.")
    n_blades = len(q_cols) // 3

    expected = [f"q_b{b}_m{m}" for b in range(n_blades) for m in range(3)]
    missing_q = set(expected) - set(df.columns)
    extra_q = set(q_cols) - set(expected)
    if missing_q or extra_q:
        raise ValueError(
            f"State CSV {state_csv} blade-mode columns mismatch. "
            f"Missing: {sorted(missing_q)}. Unexpected: {sorted(extra_q)}."
        )

    allowed = required_fixed.union(expected)
    unexpected = set(df.columns) - allowed
    if unexpected:
        raise ValueError(f"State CSV {state_csv} has unexpected columns: {sorted(unexpected)}")

    time = df["time"].to_numpy()
    x_t = df["x_t"].to_numpy()
    phi_shaft = df["phi_shaft"].to_numpy()
    q_modal = np.stack(
        [df[f"q_b{b}_m{m}"].to_numpy() for b in range(n_blades) for m in range(3)],
        axis=1,
    ).reshape(len(df), n_blades, 3)
    return time, x_t, phi_shaft, q_modal, n_blades


def _load_pitch(pitch_csv: Path | None, time_grid: np.ndarray) -> np.ndarray:
    if pitch_csv is None:
        return np.zeros_like(time_grid)
    df = pd.read_csv(pitch_csv)
    if set(df.columns) != {"time", "pitch"}:
        raise ValueError(f"Pitch CSV {pitch_csv} must have exactly columns 'time' and 'pitch', got {list(df.columns)}.")
    if df["time"].shape != time_grid.shape or not np.allclose(time_grid, df["time"]):
        raise ValueError("The times of the pitch and motion timeseries do not match.")
    return df["pitch"].to_numpy()


def _load_geometry(
    blade_data_csv: Path, mode_shapes_csv: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df_blade = pd.read_csv(blade_data_csv)
    r_blade = df_blade["radius"].to_numpy()
    chord = df_blade["chord"].to_numpy()
    rel_thickness = df_blade["rel_thickness"].to_numpy() / 100.0
    twist = np.deg2rad(df_blade["twist"].to_numpy())

    df_modes = pd.read_csv(mode_shapes_csv)
    if not np.allclose(df_modes["r"].to_numpy(), r_blade):
        raise ValueError(f"Mode-shape radii in {mode_shapes_csv} do not match blade radii from {blade_data_csv}.")
    phi_y_org = np.vstack([df_modes["u1fy"].to_numpy(), df_modes["u1ey"].to_numpy(), df_modes["u2fy"].to_numpy()])
    phi_z_org = np.vstack([df_modes["u1fz"].to_numpy(), df_modes["u1ez"].to_numpy(), df_modes["u2fz"].to_numpy()])
    return r_blade, chord, rel_thickness, twist, phi_y_org, phi_z_org


def _blade_positions(
    q_modal_b: np.ndarray,  # (3,)
    pitch: float,
    phi_shaft: float,
    blade_idx: int,
    n_blades: int,
    r: np.ndarray,
    phi_y_org: np.ndarray,  # (3, n_elements)
    phi_z_org: np.ndarray,  # (3, n_elements)
    x_t: float,
    hub_height: float,
    l_shaft: float,
    tilt: float,
    cone: float,
    yaw: float,
    scale_deflections: float,
) -> np.ndarray:
    dy0 = q_modal_b @ phi_y_org * scale_deflections
    dz0 = q_modal_b @ phi_z_org * scale_deflections
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    dy = dz0 * sin_p + dy0 * cos_p
    dz = dz0 * cos_p - dy0 * sin_p

    p5 = np.column_stack([r, dy, dz])
    p4 = Rotation.rotate_3d_y(p5, cone)
    p3 = Rotation.rotate_3d_z(p4, phi_shaft + blade_idx * 2 * np.pi / n_blades)
    p2 = Rotation.rotate_3d_y(p3 + np.asarray([0.0, 0.0, -l_shaft]), tilt)
    p1 = Rotation.rotate_3d_x(p2 + np.asarray([hub_height, 0.0, 0.0]), yaw)
    p1 = p1 + np.asarray([0.0, 0.0, x_t])
    return p1


def _blade_box_positions(
    q_modal_b: np.ndarray,  # (3,)
    pitch: float,
    phi_shaft: float,
    blade_idx: int,
    n_blades: int,
    r: np.ndarray,
    phi_y_org: np.ndarray,
    phi_z_org: np.ndarray,
    half_chord: np.ndarray,  # (n_elements,)
    half_thickness: np.ndarray,  # (n_elements,)
    twist: np.ndarray,  # (n_elements,) radians
    x_t: float,
    hub_height: float,
    l_shaft: float,
    tilt: float,
    cone: float,
    yaw: float,
    scale_deflections: float,
) -> np.ndarray:
    dy0_centerline = q_modal_b @ phi_y_org * scale_deflections
    dz0_centerline = q_modal_b @ phi_z_org * scale_deflections

    # Section-frame corner offsets: (+y,+z), (-y,+z), (-y,-z), (+y,-z)
    sign_y = np.asarray([+1.0, -1.0, -1.0, +1.0])
    sign_z = np.asarray([+1.0, +1.0, -1.0, -1.0])
    c_y = sign_y[:, None] * half_chord[None, :]  # (4, n_el)
    c_z = sign_z[:, None] * half_thickness[None, :]  # (4, n_el)

    # Twist about the radial axis (same convention as the pitch transform below).
    cos_t, sin_t = np.cos(twist), np.sin(twist)
    c_y_pre = c_z * sin_t[None, :] + c_y * cos_t[None, :]
    c_z_pre = c_z * cos_t[None, :] - c_y * sin_t[None, :]

    dy0 = dy0_centerline[None, :] + c_y_pre  # (4, n_el)
    dz0 = dz0_centerline[None, :] + c_z_pre

    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    dy = dz0 * sin_p + dy0 * cos_p
    dz = dz0 * cos_p - dy0 * sin_p

    n_el = r.size
    out = np.empty((4, n_el, 3))
    for c in range(4):
        p5 = np.column_stack([r, dy[c], dz[c]])
        p4 = Rotation.rotate_3d_y(p5, cone)
        p3 = Rotation.rotate_3d_z(p4, phi_shaft + blade_idx * 2 * np.pi / n_blades)
        p2 = Rotation.rotate_3d_y(p3 + np.asarray([0.0, 0.0, -l_shaft]), tilt)
        p1 = Rotation.rotate_3d_x(p2 + np.asarray([hub_height, 0.0, 0.0]), yaw)
        out[c] = p1 + np.asarray([0.0, 0.0, x_t])
    return out


def animate(
    state_csv: str | Path,
    output: str | Path,
    pitch_csv: str | Path | None = None,
    *,
    fps: int = 30,
    scale_deflections=1.0,
    blade_data_csv: str | Path = "data/blade_data.csv",
    mode_shapes_csv: str | Path = "data/blade_mode_shapes.csv",
    hub_height: float = 119.0,
    tip_deflection: bool = True,
    n_tip_deflections: int = 40,
    plot_blade: str | None = "flapwise",
    minmax_deflections: tuple[float, float] = (-2.0, 14.0),
    title: str | None = "DTU 10 MW",
    blade_model: str = "box",
    T=(0, -1),
    n_blades_total=3,
    tilt=-5,
) -> None:
    """
    Animates the solution of a wind turbine (tower fore-aft, blade edge/flapwise, rotor rotation) simulation, flap vs
    edgewise displacements, and the flap- or edgewise bending of the blades.

    The animations of the flap vs edgewise and flap or edgewise bending figures are in the pitched coordinate system of
    the blades.

    The animation of the whole wind turbine is in the global coordinate system.

    Parameters
    ----------
    state_csv : str | Path
        Path to the csv file that stores the displacement time series of
            - t (must be present): times in seconds
            - x_t (must be present): tower fore-aft position
            - phi_shaft (must be present): rotor shaft position in RADIAN
            - q_b<i>_m{j} (can be present): magnitude of mode {i} for blade {b}. At least one blade must be specified,
            but there is no upper limit for the number of blades. If fewer blades are specified than `n_blades_total`,
            then the remaining blades are assumed stiff.
    output : str | Path
        Path to file to which the animation will be saved. Can end in ".mp4" or ".gif". For ".mp4", FFmpeg needs to be
        installed.
    pitch_csv : str | Path | None, optional
        Path to a csv file storing the time series of the collective pitch. If provided, needs to have the same times
        as `state_csv`. Required columns are
            - t (must be present): times in seconds
            - pitch (must be present): pitch in DEGREE
    fps : int, optional
        Frames per second of the animation, by default 30
    scale_deflections : float, optional
        Constant factor by which the tower displacement and modal amplitudes can be scaled, by default 1.0
    blade_data_csv : str | Path, optional
        Path to file describing the blade. Columns must include radius, twist, chord, rel_thickness, by default "data/
        blade_data.csv"
    mode_shapes_csv : str | Path, optional
        Path to file describing the mode shapes. Columns must include r, u1fy, u1fz, u1ey, u1ez, u2fy, u2fz, by default
        "data/blade_mode_shapes.csv"
    hub_height : float, optional
        Hub height, by default 119.0
    tip_deflection : bool, optional
        Whether or not to plot flapwise vs edgewise tip deflection, by default True
    n_tip_deflections : int, optional
        Number of trailing points behind the current tip deflections, by default 40
    plot_blade : str | None, optional
        Whether or not and if so which direction of the bending to plot. Can be "flapwise", "edgewise", None, by
        default "flapwise"
    minmax_deflections : tuple[float, float], optional
        Axis limits for the tip deflection and blade bending plots, by default (-2.0, 14.0)
    title : str | None, optional
        Figure title. Some additional information is added to the title by default, by default "DTU 10 MW"
    blade_model : str, optional
        How to plot the blade_model, can be "box" (wireframe of cuboids) or "line" ("blade centre"), by default "box"
    T : tuple, optional
        Period for which to animate, by default (0, -1), i.e. the whole timeseries
    n_blades_total : int, optional
        Number of total blades. May be overridden if more flexible blades are defined, by default 3
    tilt : int, optional
        Tilt of the shaft, by default -5
    """

    if plot_blade not in (None, "flapwise", "edgewise"):
        raise ValueError(f"plot_blade must be None, 'flapwise', or 'edgewise', got {plot_blade}.")
    if blade_model not in ("line", "box"):
        raise ValueError(f"blade_model must be 'line' or 'box', got {blade_model}.")

    state_csv = Path(state_csv)
    output = Path(output)

    l_shaft = 7.1
    tilt = np.deg2rad(tilt)
    cone = np.deg2rad(2.5)
    yaw = np.deg2rad(0.0)

    time, x_t, phi_shaft, q_modal, n_blades_flex = _load_state(state_csv)
    n_blades_total = max(n_blades_flex, n_blades_total)
    if n_blades_flex < n_blades_total:
        pad = np.zeros((q_modal.shape[0], n_blades_total - n_blades_flex, 3))
        q_modal = np.concatenate([q_modal, pad], axis=1)
    r, chord, rel_thickness, twist, phi_y_org, phi_z_org = _load_geometry(Path(blade_data_csv), Path(mode_shapes_csv))
    pitch = _load_pitch(Path(pitch_csv) if pitch_csv is not None else None, time)
    radius = float(r[-1])
    half_chord = 0.5 * chord
    half_thickness = 0.5 * chord * rel_thickness

    t_start = time[0] if T[0] == 0 else T[0]
    t_end = time[-1] if T[1] == -1 else T[1]
    duration = float(t_end - t_start)
    n_frames = max(2, int(round(duration * fps)))
    frame_times = np.linspace(t_start, t_end, n_frames)
    frame_idx = np.searchsorted(time, frame_times).clip(0, len(time) - 1)
    unique = np.unique(frame_idx)
    if unique.size < n_frames:
        print(f"Timeseries is too sparse for {fps=}. Using {unique.size} unique frames for {n_frames} frames.")
    else:
        print(f"Using {n_frames} unique frames to animate.")

    fig_w = 12 if tip_deflection else 8
    fig_h = 10 if plot_blade is not None else 8
    fig = plt.figure(figsize=(fig_w, fig_h))

    def get_title(time: float, title=title):
        if title is not None:
            title += f"\nt = {time:0.2f}s"
        else:
            title = f"t = {time:0.2f}s"
        if scale_deflections != 1:
            title += f", deflections scaled by {scale_deflections}x"
        return title

    fig.suptitle(get_title(0), y=0.9)
    height_ratios = [3, 1] if plot_blade is not None else [1]
    width_ratios = [3, 2] if tip_deflection else [1]
    gs = fig.add_gridspec(
        len(height_ratios),
        len(width_ratios),
        height_ratios=height_ratios,
        width_ratios=width_ratios,
    )
    ax: Axes3D = fig.add_subplot(gs[0, 0], projection="3d")
    ax_td = fig.add_subplot(gs[0, 1]) if tip_deflection else None
    ax_blade = fig.add_subplot(gs[1, :]) if plot_blade is not None else None

    ax.view_init(elev=10, azim=-25, vertical_axis="x")
    span = hub_height + radius
    ax.set_xlim(0, span)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)
    yz_ticks = [-80, 0, 80]
    ax.set_yticks(yz_ticks, list(map(str, yz_ticks)))
    ax.set_zticks(yz_ticks, list(map(str, yz_ticks)))
    ax.set_box_aspect((span, 2 * radius, 2 * radius))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    (tower_line,) = ax.plot([0, hub_height], [0, 0], [0, 0], color="black", lw=2)
    (shaft_line,) = ax.plot([], [], [], color="black", lw=2)
    # Always create the line artists so per-blade colors are consistent across modes;
    # for "box" mode they remain empty/hidden and the wireframe collections are drawn instead.
    blade_colors = [None] * n_blades_flex + ["black"] * (n_blades_total - n_blades_flex)
    blade_lines = [
        ax.plot([], [], [], lw=2, color=c)[0] if c is not None else ax.plot([], [], [], lw=2)[0] for c in blade_colors
    ]
    blade_boxes: list[Line3DCollection] = []
    if blade_model == "box":
        # Use a degenerate placeholder segment so add_collection3d's auto-scale
        # has something to consume; segments are replaced on the first frame.
        placeholder = [[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]]
        for b in range(n_blades_total):
            blade_lines[b].set_visible(False)
            coll = Line3DCollection(placeholder, lw=1.0, color=blade_lines[b].get_color())
            ax.add_collection3d(coll)
            blade_boxes.append(coll)
    (hub_marker,) = ax.plot([], [], [], marker="o", color="red", ms=6, linestyle="")

    td_lines: list = []
    td_markers: list = []
    td_dy: list[list[float]] = [[] for _ in range(n_blades_total)]
    td_dz: list[list[float]] = [[] for _ in range(n_blades_total)]
    phi_y_tip = phi_y_org[:, -1]
    phi_z_tip = phi_z_org[:, -1]
    if ax_td is not None:
        ax_td.set_xlim(*minmax_deflections)
        ax_td.set_ylim(*minmax_deflections)
        ax_td.set_aspect("equal")
        ax_td.set_autoscale_on(False)
        ax_td.set_xlabel("Edgewise Tip Deflection (m)")
        ax_td.set_ylabel("Flapwise Tip Deflection (m)")
        ax_td.axhline(0.0, color="0.7", lw=0.5)
        ax_td.axvline(0.0, color="0.7", lw=0.5)
        td_lines = [ax_td.plot([], [], lw=1.5, color=blade_lines[b].get_color())[0] for b in range(n_blades_total)]
        td_markers = [
            ax_td.plot([], [], marker="o", linestyle="", color=blade_lines[b].get_color(), ms=5)[0]
            for b in range(n_blades_total)
        ]

    blade_profile_lines: list = []
    if ax_blade is not None:
        ax_blade.set_xlim(0.0, radius + r[0])
        ax_blade.set_ylim(*minmax_deflections)
        ax_blade.set_autoscale_on(False)
        ax_blade.set_xlabel("r (m)")
        ax_blade.set_ylabel("Flapwise (m)" if plot_blade == "flapwise" else "Edgewise (m)")
        ax_blade.axhline(0.0, color="0.7", lw=0.5)
        blade_profile_lines = [
            ax_blade.plot([], [], lw=1.5, color=blade_lines[b].get_color())[0] for b in range(n_blades_total)
        ]

    def update(frame_no: int):
        i = int(frame_idx[frame_no])
        if frame_no % 100 == 0 and frame_no != 0:
            print(
                f"  At frame number {frame_no}, t_sim = {time[i]:0.2f}s, time: {datetime.now().time().strftime('%H:%M:%S')}"
            )

        fig.suptitle(get_title(time[i]), y=0.9)
        x_t_i = float(x_t[i]) * scale_deflections
        phi_shaft_i = float(phi_shaft[i])
        pitch_i = np.deg2rad(float(pitch[i]))

        tower_line.set_data_3d([0, hub_height], [0, 0], [0, x_t_i])

        if blade_model == "line":
            for b, line in enumerate(blade_lines):
                p1 = _blade_positions(
                    q_modal[i, b],
                    pitch_i,
                    phi_shaft_i,
                    b,
                    n_blades_total,
                    r,
                    phi_y_org,
                    phi_z_org,
                    x_t_i,
                    hub_height,
                    l_shaft,
                    tilt,
                    cone,
                    yaw,
                    scale_deflections,
                )
                line.set_data_3d(p1[:, 0], p1[:, 1], p1[:, 2])
        else:
            n_el = r.size
            for b, coll in enumerate(blade_boxes):
                corners = _blade_box_positions(
                    q_modal[i, b],
                    pitch_i,
                    phi_shaft_i,
                    b,
                    n_blades_total,
                    r,
                    phi_y_org,
                    phi_z_org,
                    half_chord,
                    half_thickness,
                    twist,
                    x_t_i,
                    hub_height,
                    l_shaft,
                    tilt,
                    cone,
                    yaw,
                    scale_deflections,
                )
                segments = []
                # Spanwise edges (one polyline per corner across all stations).
                for c in range(4):
                    for k in range(n_el - 1):
                        segments.append([corners[c, k], corners[c, k + 1]])
                # Section rectangles at each station.
                for k in range(n_el):
                    for c in range(4):
                        segments.append([corners[c, k], corners[(c + 1) % 4, k]])
                coll.set_segments(segments)

        hub2 = Rotation.rotate_3d_y(np.asarray([[0.0, 0.0, -l_shaft]]), tilt)
        hub1 = Rotation.rotate_3d_x(hub2 + np.asarray([hub_height, 0.0, 0.0]), yaw)
        hub1 = hub1 + np.asarray([0.0, 0.0, x_t_i])
        hub_marker.set_data_3d(hub1[:, 0], hub1[:, 1], hub1[:, 2])

        tower_top = np.asarray([hub_height, 0.0, x_t_i])
        shaft_line.set_data_3d(
            [tower_top[0], hub1[0, 0]],
            [tower_top[1], hub1[0, 1]],
            [tower_top[2], hub1[0, 2]],
        )

        if tip_deflection:
            for b in range(n_blades_total):
                dy0_tip_b = float(q_modal[i, b] @ phi_y_tip) * scale_deflections
                dz0_tip_b = float(q_modal[i, b] @ phi_z_tip) * scale_deflections
                td_dy[b].append(dy0_tip_b)
                td_dz[b].append(dz0_tip_b)
                if n_tip_deflections == -1:
                    xs, ys = td_dy[b], td_dz[b]
                else:
                    xs = td_dy[b][-n_tip_deflections:]
                    ys = td_dz[b][-n_tip_deflections:]
                td_lines[b].set_data(xs, ys)
                td_markers[b].set_data([dy0_tip_b], [dz0_tip_b])

        if plot_blade is not None:
            phi_span = phi_z_org if plot_blade == "flapwise" else phi_y_org
            for b in range(n_blades_total):
                blade_profile_lines[b].set_data(r, q_modal[i, b] @ phi_span * scale_deflections)

        return [
            tower_line,
            shaft_line,
            *blade_lines,
            *blade_boxes,
            hub_marker,
            *td_lines,
            *td_markers,
            *blade_profile_lines,
        ]

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".mp4":
        writer = animation.FFMpegWriter(fps=fps)
    elif suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        raise ValueError(f"Unsupported output extension {suffix}; use .mp4 or .gif.")
    anim.save(output, writer=writer)
    plt.close(fig)
    print(f"Wrote animation to {output.resolve()}")
