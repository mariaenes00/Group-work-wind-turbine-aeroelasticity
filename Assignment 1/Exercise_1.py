import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameters (DTU 10 MW)
# ----------------------------
H = 119.0        # hub height [m]
Ls = 7.1         # shaft length [m]
r_blade = 70.0   # point on blade radius [m]
omega = 0.62     # rad/s
dt = 0.15        # s
T = 40.0         # total sim time [s] (change as you like)

# Wind
Vhub = 10.0      # mean wind at hub height [m/s]
nu_shear = 0.2   # shear exponent

# Tower
a_tower_const = 3.32  # tower radius [m] (constant option)

# ----------------------------
# Rotation / transformation matrices (match slides)
# ----------------------------
def A_yaw(theta_yaw):
    """Yaw: from system 1 -> 2, rotation about x-axis (slide 5/6)."""
    c, s = np.cos(theta_yaw), np.sin(theta_yaw)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,  c,  s],
        [0.0, -s,  c],
    ])

def A_tilt(theta_tilt):
    """Tilt: rotation about y-axis (slide 6)."""
    c, s = np.cos(theta_tilt), np.sin(theta_tilt)
    return np.array([
        [ c, 0.0, -s],
        [0.0, 1.0, 0.0],
        [ s, 0.0,  c],
    ])

def A_blade(theta_blade):
    """Blade azimuth matrix a23 (slide 7)."""
    c, s = np.cos(theta_blade), np.sin(theta_blade)
    return np.array([
        [ c,  s, 0.0],
        [-s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])

def A_cone(theta_cone):
    """Cone matrix a34 (slide 7)."""
    c, s = np.cos(theta_cone), np.sin(theta_cone)
    return np.array([
        [ c, 0.0,  s],
        [0.0, 1.0, 0.0],
        [-s, 0.0,  c],
    ])

def A14(theta_yaw, theta_tilt, theta_cone, theta_blade):
    """
    Total transform a14 = a34 * a23 * a12
    where a12 = tilt * yaw (as in slide 6, no roll).
    """
    a12 = A_tilt(theta_tilt) @ A_yaw(theta_yaw)
    return A_cone(theta_cone) @ A_blade(theta_blade) @ a12

# ----------------------------
# Geometry: point on blade in system 1
# (matches idea on slide 8: r1 = rt + rs + rb)
# ----------------------------
def point_position_system1(theta_yaw, theta_tilt, theta_cone, theta_blade, r=r_blade):
    a14 = A14(theta_yaw, theta_tilt, theta_cone, theta_blade)
    a41 = a14.T
    a12 = A_tilt(theta_tilt) @ A_yaw(theta_yaw)
    a21 = a12.T

    # tower top / hub-height reference point in system 1
    r_t1 = np.array([H, 0.0, 0.0])

    # shaft offset (0,0,-Ls) in system 2 -> to system 1 via a21
    r_s2 = np.array([0.0, 0.0, -Ls])
    r_s1 = a21 @ r_s2

    # blade point in system 4 -> to system 1 via a41
    r_b4 = np.array([r, 0.0, 0.0])
    r_b1 = a41 @ r_b4

    return r_t1 + r_s1 + r_b1, a14

# ----------------------------
# Wind models in system 1
# ----------------------------
def wind_shear_system1(x_height, Vhub=Vhub, nu=nu_shear):
    """
    Wind direction aligned with z1 (as in slide 10):
    V0,1 = [0, 0, Vhub*(x/H)^nu]
    """
    x_eff = max(x_height, 0.0) # hvis punktet ligger “under bakken” numerisk (kan skje hvis man gjør feil eller punkt går under 0), så vil vi ikke ha negative høyder i en potensfunksjon
    Vz = Vhub * (x_eff / H) ** nu # beregner vindfarten i z-retning ved høyde x_eff
    return np.array([0.0, 0.0, Vz]) # betyr: ingen vind i x og y, bare i z

def tower_potential_flow_system1(pos1, Vbase, a_tower):
    """
    Very standard potential flow around cylinder (slide 11 structure).
    Here cylinder axis is x (vertical), cross-section in (y,z).
    pos1 = [x,y,z] in system 1.
    Returns wind vector in system 1 including tower influence.
    """
    x, y, z = pos1

    # no tower above tower top (slide note: a(x)=0 for x > H)
    if x > H:
        return np.array([0.0, 0.0, Vbase]) # Over tårnhøyden finnes ikke tårn i modellen → ingen påvirkning

    a = a_tower
    r = np.hypot(y, z) # Dette er avstanden fra tårnets sentrum ut til punktet i tverrsnittet
    if r < 1e-6:
        r = 1e-6

    # Avoid singularity if point is inside tower
    if r <= a:
        return np.array([0.0, 0.0, 0.0])

    # trig definitions (parsed slide text indicates these signs)
    cos_th = z / r
    sin_th = -y / r

    # Potential-flow radial/tangential components
    Vr = Vbase * (1.0 - (a / r) ** 2) * cos_th
    Vt = -Vbase * (1.0 + (a / r) ** 2) * sin_th

    # Back to (y,z) in system 1 (consistent with slide 11 relations)
    Vz = Vr * cos_th - Vt * sin_th
    Vy = -(Vr * sin_th + Vt * cos_th)

    return np.array([0.0, Vy, Vz])

# ----------------------------
# Simulation driver
# ----------------------------
def simulate(theta_yaw_deg=0.0, theta_tilt_deg=0.0, theta_cone_deg=0.0,
             use_shear=True, use_tower=False, a_tower=a_tower_const,
             blade_index=1):
    """
    blade_index: 1,2,3
    Returns time, azimuth, positions in sys1, wind in sys4 for that blade.
    """
    # Konverterer grader til radianer
    theta_yaw = np.deg2rad(theta_yaw_deg)
    theta_tilt = np.deg2rad(theta_tilt_deg)
    theta_cone = np.deg2rad(theta_cone_deg)

    # lager tidsvektor
    t = np.arange(0.0, T + dt, dt)

    # lager azimuth vinkel for hvert blad (bladene er 120 grader forskjøvet)
    theta1 = omega * t
    if blade_index == 1:
        theta = theta1
    elif blade_index == 2:
        theta = theta1 + 2.0 * np.pi / 3.0
    elif blade_index == 3:
        theta = theta1 + 4.0 * np.pi / 3.0
    else:
        raise ValueError("blade_index must be 1, 2, or 3")

    # lagrer posisjon i system 1 og vind i system 4 over tid
    r1_hist = np.zeros((len(t), 3))
    Vo4_hist = np.zeros((len(t), 3))

    for i, th in enumerate(theta):
        pos1, a14 = point_position_system1(theta_yaw, theta_tilt, theta_cone, th, r=r_blade)
        r1_hist[i, :] = pos1

        # Base wind magnitude at this height
        if use_shear:
            V01 = wind_shear_system1(pos1[0], Vhub=Vhub, nu=nu_shear)
        else:
            V01 = np.array([0.0, 0.0, Vhub])

        # Tower influence (optional)
        if use_tower:
            Vbase = V01[2]  # wind along z1
            V01 = tower_potential_flow_system1(pos1, Vbase=Vbase, a_tower=a_tower)

        # Transform wind to blade system (system 4)
        Vo4 = a14 @ V01
        Vo4_hist[i, :] = Vo4

    return t, theta, r1_hist, Vo4_hist

# ----------------------------
# Plots for Exercise 1
# ----------------------------
def plot_part1_positions():
    # Part 1: yaw=0, tilt=0, cone=0, show (x,y) in system 1 for each blade
    plt.figure()
    for b in [1, 2, 3]:
        t, th, r1, _ = simulate(theta_yaw_deg=0, theta_tilt_deg=0, theta_cone_deg=0,
                                use_shear=False, use_tower=False, blade_index=b)
        plt.plot(r1[:, 0], r1[:, 1], label=f"Blade {b}")
    plt.xlabel("x1 [m] (height)")
    plt.ylabel("y1 [m]")
    plt.title("Part 1: Position of point (x1,y1), yaw=0, tilt=0, cone=0")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

def plot_part2_positions_yaw20():
    # Part 2: yaw=20 deg, show blade 1 (or all blades)
    plt.figure()
    for b in [1, 2, 3]:
        t, th, r1, _ = simulate(theta_yaw_deg=20, theta_tilt_deg=0, theta_cone_deg=0,
                                use_shear=False, use_tower=False, blade_index=b)
        plt.plot(r1[:, 0], r1[:, 1], label=f"Blade {b}")
    plt.xlabel("x1 [m] (height)")
    plt.ylabel("y1 [m]")
    plt.title("Part 2: Position of point (x1,y1), yaw=20°")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

def plot_part3_wind_components_shear(blade_index=1):
    # Part 3: Vy and Vz in blade system vs azimuth, yaw=0 and yaw=20
    plt.figure()
    for yaw in [0, 20]:
        t, th, _, Vo4 = simulate(theta_yaw_deg=yaw, theta_tilt_deg=0, theta_cone_deg=0,
                                 use_shear=True, use_tower=False, blade_index=blade_index)
        # Blade system components: [Vx4, Vy4, Vz4]
        plt.plot(np.unwrap(th), Vo4[:, 1], label=f"Vy4, yaw={yaw}°")
        plt.plot(np.unwrap(th), Vo4[:, 2], linestyle="--", label=f"Vz4, yaw={yaw}°")
    plt.xlabel("Azimuth θ [rad]")
    plt.ylabel("Wind components in system 4 [m/s]")
    plt.title(f"Part 3: Wind shear (ν={nu_shear}), Blade {blade_index}")
    plt.grid(True)
    plt.legend()

def plot_part4_tower_shadow(blade_index=1):
    # Part 4: zero shear, zero yaw, include tower
    plt.figure()
    t, th, _, Vo4 = simulate(theta_yaw_deg=0, theta_tilt_deg=0, theta_cone_deg=0,
                             use_shear=False, use_tower=True, a_tower=a_tower_const,
                             blade_index=blade_index)
    plt.plot(np.unwrap(th), Vo4[:, 1], label="Vy4")
    plt.plot(np.unwrap(th), Vo4[:, 2], label="Vz4")
    plt.xlabel("Azimuth θ [rad]")
    plt.ylabel("Wind components in system 4 [m/s]")
    plt.title("Part 4: Tower influence, no shear, yaw=0")
    plt.grid(True)
    plt.legend()

def plot_part5_all_combined(blade_index=1):
    # Part 5: yaw=20, shear + tower
    plt.figure()
    t, th, _, Vo4 = simulate(theta_yaw_deg=20, theta_tilt_deg=0, theta_cone_deg=0,
                             use_shear=True, use_tower=True, a_tower=a_tower_const,
                             blade_index=blade_index)
    plt.plot(np.unwrap(th), Vo4[:, 1], label="Vy4")
    plt.plot(np.unwrap(th), Vo4[:, 2], label="Vz4")
    plt.xlabel("Azimuth θ [rad]")
    plt.ylabel("Wind components in system 4 [m/s]")
    plt.title("Part 5: yaw=20° + shear + tower")
    plt.grid(True)
    plt.legend()

if __name__ == "__main__":
    plot_part1_positions()
    plot_part2_positions_yaw20()
    plot_part3_wind_components_shear(blade_index=1)
    plot_part4_tower_shadow(blade_index=1)
    plot_part5_all_combined(blade_index=1)
    plt.show()
