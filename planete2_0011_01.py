
##################################################################
#
## Simple terrestrial planet temperature energy balance model 2D
#
## Python3, Numba JIT
#
# 10.10.2026 00.11.01
#
#################################################################


import math

import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import zoom
from numba import njit

import noise
from noise import pnoise3


# ============================================================
#  MAIN CONTROL PARAMETERS
# ============================================================

## simulation dimensions

simu_nx = 32
simu_ny = 16
        # simulation
simu_run_years=7
        # save daily

# <-- THIS IS THE MAIN TIMESTEP
simu_dt_hours=1

simu_save_every_hours=24

## accurate map dimensions
height=360*2
width=720*2
height_delta=10000
sea_level=6000

#seed1=4 ## dem map seed
seed1=99




#from scipy import constants

# Vaihtoehtoisesti astrofysiikan astropy-kirjastosta:
# from astropy.constants import M_earth, R_earth


planet_mass_me=1
planet_radius_re=1   
surface_pressure_atm=1
#co2_ppm=280*1000
co2_ppm=420

solar_constant_suns=1

initial_temperature=288.15*math.sqrt(solar_constant_suns)

orbital_period_years=1.0

#rotation_period_days=365.25*1.0
rotation_period_days=1
eccentricity=0.0167
perihelion=283.0
tilt=23.44


# KOI 4878.01

#solar_constant=1365.2*0.98 #0.92-1.04
#solar_constant=1365.2*0.92 #0.92-1.04

#orbital_period_years=449.015/365.25
#rotation_period_days=1.0
#eccentricity=0.0167
#perihelion=283.0
#tilt=23.44


# clouds
#use_clouds=True
clouds_present=2 # 0 none 1 zonal 2 locked hurruicane

cloud_strength=1

# ice
#use_ice=True
use_ice=True


# horizontal heat transport
# start with 0 while testing energy balance
#diffusion=0.1 ## locked planet
#diffusion=1e5 ## fastrot
#diffusion=3e7*pow(surface_pressure_atm, -0.5) ## slowrot 1e7 ... 1e8
#diffusion=1e6*pow(surface_pressure_atm, -0.5) ##  fastrot
diffusion=0


#Eksponentti asettuu tyypillisesti välille \(-0.25\) ... \(-0.5\)  -1 ?

# ============================================================
#  CONSTANTS




## basic const

G_CONST = 6.67430e-11       # m^3 kg^-1 s^-2
G= G_CONST # gravitaatiovakio [m^3 kg^-1 s^-2]
SIGMA = 5.670374419e-8

## other const

M_e = 5.972e24 # Maan massa kilogrammoina (kg)
R_e = 6.371e6  # Maan keskimääräinen säde metreinä (m)

R_AIR = 287.05              # J kg^-1 K^-1
P_REF = 101325.0  ## earth sea-lavel pressure   
ice_temperature=263.0

## SI unit params

planet_radius_m=R_e*planet_radius_re
planet_mass_kg=M_e*planet_mass_me
surface_pressure=101325*surface_pressure_atm
solar_constant=solar_constant_suns*1365.2

# ------------------------------------------------------------
# MATERIAL PARAMETERS
#
# 0 = ocean
# 1 = land
# 2 = sea ice
# 3 = snow / land ice
# ------------------------------------------------------------

@njit
def material_properties(material):

    if material == 0:       # ocean
        rho = 1030.0
        cp = 4184.0
        depth = 10.0
        albedo = 0.06+0.2 ## clouds add albedo

    elif material == 1:     # land
        rho = 2700.0
        cp = 750.0
        depth = 2.0
        albedo = 0.20

    elif material == 2:     # sea ice
        rho = 917.0
        cp = 2100.0
        depth = 3.0
        albedo = 0.60
    elif material == 3:                    # snow / land ice
        rho = 400.0
        cp = 880.0
        depth = 3.0
        albedo = 0.75
    else:     # n/a, land
        rho = 2700.0
        cp = 750.0
        depth = 2.0
        albedo = 0.20

    C = rho * cp * depth

    return C, albedo








# ------------------------------------------------------------
# CLOUD ALBEDO
# ------------------------------------------------------------

@njit
def cloud_albedo(cloud_fraction):

    cloud_percent = cloud_fraction * 100.0

    a = (
        0.0028623 * cloud_percent
        - 0.000009 * cloud_percent * cloud_percent
    )

    if a < 0.0:
        a = 0.0
    if a > 1.0:
        a = 1.0

    return a


# ------------------------------------------------------------
# SIMPLE LATITUDINAL CLOUD BELT
# ------------------------------------------------------------

def cloud_fraction_np(lat, solar_declination):

    # six broad cloud belts
    #x = (lat - solar_declination * 0.5) * math.pi / 180.0
    x = (lat - solar_declination ) * math.pi / 180.0
    c = 0.5 + 0.5 * np.sin(6.0 * x)
    c=np.where(c<0,0,c)
    c=np.where(c>1,1,c)
    #if c < 0.0:
    #    c = 0.0
    #if c > 1.0:
    #    c = 1.0

    return c
    
    
@njit
def cloud_fraction_fastrot(lat, solar_declination):

    # six broad cloud belts
    #x = (lat - solar_declination * 0.5) * math.pi / 180.0
    x = (lat - solar_declination ) * math.pi / 180.0
    c = 0.5 + 0.5 * math.sin(6.0 * x)

    if c < 0.0:
        c = 0.0
    if c > 1.0:
        c = 1.0

    return c

@njit
def cloud_fraction_locked(
    lat,
    lon,
    substellar_lat=0.0,
    substellar_lon=0.0
):

    deg2rad = math.pi / 180.0

    phi1 = lat * deg2rad
    lambda1 = lon * deg2rad

    phi2 = substellar_lat * deg2rad
    lambda2 = substellar_lon * deg2rad

    cos_dist = (
        math.sin(phi1) * math.sin(phi2)
        +
        math.cos(phi1)
        * math.cos(phi2)
        * math.cos(lambda1 - lambda2)
    )

    cos_dist = max(-1.0, min(1.0, cos_dist))

    distance = math.acos(cos_dist) / deg2rad

    eye_radius = 25.0
    storm_radius = 80.0

    if distance <= eye_radius:

        # lähes pilvetön silmä,
        # mutta pilvisyys kasvaa kohti eyewallia
        t = distance / eye_radius

        return 0.10 + 0.80 * t

    elif distance <= storm_radius:

        # paksu pilvivyöhyke
        t = (
            distance - eye_radius
        ) / (
            storm_radius - eye_radius
        )

        # 0.90 -> 0.0
        return 0.90 * (1.0 - t) ** 2

    else:

        return 0.0


@njit
def cloud_fraction_locked2(
    lat,
    lon,
    substellar_lat=0.0,
    substellar_lon=0.0
):

    deg2rad = math.pi / 180.0

    phi1 = lat * deg2rad
    lambda1 = lon * deg2rad

    phi2 = substellar_lat * deg2rad
    lambda2 = substellar_lon * deg2rad

    cos_dist = (
        math.sin(phi1) * math.sin(phi2)
        +
        math.cos(phi1)
        * math.cos(phi2)
        * math.cos(lambda1 - lambda2)
    )

    cos_dist = max(-1.0, min(1.0, cos_dist))

    distance = math.acos(cos_dist) / deg2rad

    eye_radius = 25.0
    storm_radius = 80.0

    if distance <= eye_radius:

        # lähes pilvetön silmä,
        # mutta pilvisyys kasvaa kohti eyewallia
        t = distance / eye_radius

        return 0.10 + 0.80 * t

    elif distance <= storm_radius:

        # paksu pilvivyöhyke
        t = (
            distance - eye_radius
        ) / (
            storm_radius - eye_radius
        )

        # 0.90 -> 0.0
        return 0.90 * (1.0 - t) ** 2

    else:

        return 0.0





@njit
def create_cloud_raster(
    ny,
    nx,
    clouds_present,
    tilt,
    orbital_longitude,
    substellar_lat=0.0,
    substellar_lon=0.0,
):
    cloud = np.zeros((ny, nx), dtype=np.float64)

    if clouds_present == 0:
        return cloud

    # --------------------------------------------------------
    # FAST ROTATING PLANET
    # --------------------------------------------------------

    if clouds_present == 1:

        declination = solar_declination(
            tilt,
            orbital_longitude
        )

        for y in range(ny):

            lat = -90.0 + 180.0 * y / (ny - 1)

            c = cloud_fraction_fastrot(
                lat,
                declination
            )

            for x in range(nx):
                cloud[y, x] = c

    # --------------------------------------------------------
    # TIDALLY LOCKED / SLOW ROTATING PLANET
    # --------------------------------------------------------

    elif clouds_present == 2:

        for y in range(ny):

            lat = -90.0 + 180.0 * y / (ny - 1)

            for x in range(nx):

                lon = -180.0 + 360.0 * x / nx

                cloud[y, x] = cloud_fraction_locked(
                    lat,
                    lon,
                    substellar_lat,
                    substellar_lon
                )

    return cloud

def debug_ocean_currents(
    ocean_u,
    ocean_v,
    dem,
    landmask,
    step=None,
):
    ocean = (dem < 0.0) & (landmask <= 0.5)

    if not np.any(ocean):
        print("OCEAN DEBUG: No ocean cells found!")
        return

    speed = np.sqrt(ocean_u**2 + ocean_v**2)
    s = speed[ocean]

    print("\n--- OCEAN CURRENT DEBUG ---")

    if step is not None:
        print("Step:", step)

    print("Ocean cells:", s.size)
    print("Speed min:", float(np.min(s)), "m/s")
    print("Speed mean:", float(np.mean(s)), "m/s")
    print("Speed max:", float(np.max(s)), "m/s")
    print("Speed p95:", float(np.percentile(s, 95)), "m/s")

    print("u mean:", float(np.mean(ocean_u[ocean])), "m/s")
    print("v mean:", float(np.mean(ocean_v[ocean])), "m/s")

    print("Non-finite u:", int(np.sum(~np.isfinite(ocean_u[ocean]))))
    print("Non-finite v:", int(np.sum(~np.isfinite(ocean_v[ocean]))))

    land = ~ocean

    if np.any(land):
        land_speed_max = float(np.max(speed[land]))
        print("Max speed outside ocean:", land_speed_max, "m/s")

    if not np.all(np.isfinite(s)):
        print("WARNING: NaN or infinity detected!")

    if np.max(s) > 3.0:
        print("WARNING: Current speed exceeds expected limit!")
    ocean = (dem < 0.0) & (landmask <= 0.5)
    #ocean_u_surface=ocean_u
    speed = np.sqrt(
    ocean_u**2 + ocean_v**2
    )
    s = speed[ocean]
    print("Fraction at speed limit:",np.mean(s >= 2.99))
    print("Fraction above 2 m/s:",np.mean(s > 2.0))
    print("Fraction above 1 m/s:",np.mean(s > 1.0))        
        
        
        
@njit
def ocean_surface_current_step(
    ocean_u,
    ocean_v,

    u,
    v,

    dem,
    landmask,

    planet_radius_m,
    rotation_period_days,

    dt_seconds,

    # --------------------------------------------------
    # OCEAN PARAMETERS
    # --------------------------------------------------

    surface_layer_depth=50.0,
    water_density=1025.0,
    air_density=1.2,

    wind_drag_coefficient=1.3e-3,

    current_drag_timescale=86400.0,

    max_current_speed=3.0,
):
    """
    Simple wind-driven ocean surface current model.

    Inputs
    ------
    ocean_u, ocean_v:
        Previous ocean current velocities (m/s).

    u, v:
        Atmospheric wind velocities (m/s).

    dem:
        Elevation in meters.
        Ocean floor is negative; land is non-negative.

    landmask:
        1 = land, 0 = ocean.

    Returns
    -------
    ocean_u_new, ocean_v_new:
        Updated ocean current velocities (m/s).

    Method
    ------
    Exact local update for constant wind acceleration,
    Coriolis rotation, and linear drag during each step.

    Note
    ----
    This version does not yet include horizontal ocean
    momentum advection, pressure gradients, or deep currents.
    """

    ny, nx = dem.shape

    ocean_u_new = ocean_u.copy()
    ocean_v_new = ocean_v.copy()

    omega = (
        2.0 * np.pi
        / (rotation_period_days * 86400.0)
    )

    # Inverse drag timescale.
    if current_drag_timescale > 0.0:
        damping = 1.0 / current_drag_timescale
    else:
        damping = 0.0

    # Precompute the time-dependent damping factor.
    decay = np.exp(-damping * dt_seconds)

    for y in range(ny):

        latitude = (
            0.5 * np.pi
            - y * np.pi / (ny - 1)
        )

        f = 2.0 * omega * np.sin(latitude)

        angle = f * dt_seconds

        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Complex exponential for exact Coriolis rotation
        # combined with linear damping.
        rotation_decay = decay

        for x in range(nx):

            # ------------------------------------------
            # LAND MASK
            # ------------------------------------------

            if landmask[y, x] > 0.5 or dem[y, x] >= 0.0:

                ocean_u_new[y, x] = 0.0
                ocean_v_new[y, x] = 0.0

                continue

            # ------------------------------------------
            # LOCAL WATER DEPTH
            # ------------------------------------------

            depth = -dem[y, x]

            layer_depth = min(
                surface_layer_depth,
                depth
            )

            if layer_depth < 1.0:
                layer_depth = 1.0

            # ------------------------------------------
            # ATMOSPHERIC WIND
            # ------------------------------------------

            wind_u = u[y, x]
            wind_v = v[y, x]

            wind_speed = np.sqrt(
                wind_u * wind_u
                + wind_v * wind_v
            )

            # ------------------------------------------
            # WIND STRESS (N/m²)
            # ------------------------------------------

            tau_x = (
                air_density
                * wind_drag_coefficient
                * wind_speed
                * wind_u
            )

            tau_y = (
                air_density
                * wind_drag_coefficient
                * wind_speed
                * wind_v
            )

            # ------------------------------------------
            # WIND ACCELERATION (m/s²)
            # ------------------------------------------

            ax = tau_x / (
                water_density * layer_depth
            )

            ay = tau_y / (
                water_density * layer_depth
            )

            # ------------------------------------------
            # EXACT SOLUTION OF LOCAL MOMENTUM EQUATION
            #
            # du/dt = f*v + ax - damping*u
            # dv/dt = -f*u + ay - damping*v
            #
            # Constant ax, ay, f and damping during dt.
            # ------------------------------------------

            old_u = ocean_u[y, x]
            old_v = ocean_v[y, x]

            denominator = (
                damping * damping
                + f * f
            )

            if denominator > 1.0e-30:

                # Steady-state velocity for the local
                # wind forcing, Coriolis and linear drag.
                eq_u = (
                    damping * ax
                    + f * ay
                ) / denominator

                eq_v = (
                    damping * ay
                    - f * ax
                ) / denominator

                # Difference from steady-state velocity.
                du = old_u - eq_u
                dv = old_v - eq_v

                # Exact damped Coriolis rotation.
                new_u = eq_u + rotation_decay * (
                    cos_angle * du
                    + sin_angle * dv
                )

                new_v = eq_v + rotation_decay * (
                    cos_angle * dv
                    - sin_angle * du
                )

            else:

                # Degenerate case: no drag and no rotation.
                new_u = old_u + ax * dt_seconds
                new_v = old_v + ay * dt_seconds

            # ------------------------------------------
            # SAFETY LIMIT
            # ------------------------------------------

            speed = np.sqrt(
                new_u * new_u
                + new_v * new_v
            )

            if speed > max_current_speed:

                factor = max_current_speed / speed

                new_u *= factor
                new_v *= factor

            # ------------------------------------------
            # SAVE
            # ------------------------------------------

            ocean_u_new[y, x] = new_u
            ocean_v_new[y, x] = new_v

    return ocean_u_new, ocean_v_new


@njit
def ocean_heat_transport_step(
    temperature,
    ocean_u,
    ocean_v,
    material,
    dem,
    dt_seconds,
    planet_radius_m,

    # Ocean heat capacity per unit area.
    ocean_capacity=43000000.0,

    # Effective land heat capacity per unit area.
    land_capacity=4050000.0,

    # Air/land/ocean coast coupling [W/(m² K)].
    coast_exchange=15.0,

    # Safety limiter for numerical stability.
    max_temperature_change=2.0,
):
    """
    Ocean heat transport and conservative coast exchange.

    material:
        0 = ocean
        1 = land
        2 = sea ice
        3 = snow / land ice

    ocean_u, ocean_v:
        Surface ocean currents in m/s.

    temperature:
        Surface temperature in kelvin.

    Returns:
        Updated surface temperature.

    This is a simplified grid model. Coast exchange is
    represented between neighboring land and ocean cells.
    """

    ny, nx = temperature.shape

    T_old = temperature
    T_new = T_old.copy()

    dlat = np.pi / (ny - 1)
    dlon = 2.0 * np.pi / (nx - 1)

    # --------------------------------------------------
    # 1. Ocean temperature advection
    # --------------------------------------------------

    for y in range(1, ny - 1):

        latitude = -0.5 * np.pi + y * dlat
        cos_lat = abs(np.cos(latitude))

        dx = planet_radius_m * max(cos_lat, 1.0e-6) * dlon
        dy = planet_radius_m * dlat

        for x in range(nx - 1):

            if material[y, x] != 0:
                continue

            xm = x - 1
            xp = x + 1

            if xm < 0:
                xm = nx - 2

            if xp >= nx - 1:
                xp = 0

            uo = ocean_u[y, x]
            vo = ocean_v[y, x]

            T = T_old[y, x]

            # Upwind gradients. Do not advect heat
            # through land or ice-covered cells.
            dTdx = 0.0
            dTdy = 0.0

            if uo > 0.0 and material[y, xm] == 0:
                dTdx = (T - T_old[y, xm]) / dx
            elif uo < 0.0 and material[y, xp] == 0:
                dTdx = (T_old[y, xp] - T) / dx

            if vo > 0.0 and material[y - 1, x] == 0:
                dTdy = (T - T_old[y - 1, x]) / dy
            elif vo < 0.0 and material[y + 1, x] == 0:
                dTdy = (T_old[y + 1, x] - T) / dy

            tendency = -uo * dTdx - vo * dTdy

            dT = tendency * dt_seconds

            if dT > max_temperature_change:
                dT = max_temperature_change
            elif dT < -max_temperature_change:
                dT = -max_temperature_change

            T_new[y, x] += dT

    # --------------------------------------------------
    # 2. Conservative coast heat exchange
    # --------------------------------------------------

    # Flux is computed from the old temperature field.
    # Equal and opposite energy is transferred.
    for y in range(1, ny - 1):

        latitude = -0.5 * np.pi + y * dlat
        cos_lat = abs(np.cos(latitude))

        dx = planet_radius_m * max(cos_lat, 1.0e-6) * dlon
        dy = planet_radius_m * dlat

        # Approximate grid-cell area.
        area = dx * dy

        for x in range(nx - 1):

            xm = x - 1
            xp = x + 1

            if xm < 0:
                xm = nx - 2

            if xp >= nx - 1:
                xp = 0

            # Check east and north edges only,
            # avoiding duplicate neighbor pairs.
            for direction in range(2):

                if direction == 0:
                    yy = y
                    xx = xp
                else:
                    yy = y + 1
                    xx = x

                m1 = material[y, x]
                m2 = material[yy, xx]

                if not (
                    (m1 == 0 and m2 == 1)
                    or (m1 == 1 and m2 == 0)
                ):
                    continue

                T1 = T_old[y, x]
                T2 = T_old[yy, xx]

                # Use actual material capacities where
                # available, otherwise defaults.
                if m1 == 0:
                    C1 = ocean_capacity
                    C2 = land_capacity
                else:
                    C1 = land_capacity
                    C2 = ocean_capacity

                # Positive flux transfers energy from
                # cell 1 to cell 2.
                flux = coast_exchange * (T1 - T2)

                energy = flux * dt_seconds * area

                dT1 = -energy / (C1 * area)
                dT2 =  energy / (C2 * area)

                T_new[y, x] += dT1
                T_new[yy, xx] += dT2

    return T_new




import numpy as np
from numba import njit


@njit
def ocean_temperature_step_wanha(
    ocean_temperature,
    temperature,
    ocean_u,
    ocean_v,
    dem,
    landmask,
    planet_radius_m,
    dt_seconds,

    surface_layer_depth=50.0,
    water_density=1025.0,
    water_heat_capacity=3990.0,

    # Heat exchange timescale, seconds
    air_sea_exchange_timescale=5.0 * 86400.0,
):
    """
    Simple ocean surface temperature model.

    - Horizontal temperature advection by ocean currents.
    - Relaxation toward atmospheric temperature.
    - Land cells are excluded.

    Temperatures are in kelvin.
    Velocities are in m/s.
    """

    ny, nx = ocean_temperature.shape

    dlat = np.pi / (ny - 1)
    dlon = 2.0 * np.pi / (nx - 1)

    # Separate input and output arrays.
    T_old = ocean_temperature
    T_new = ocean_temperature.copy()

    # Atmospheric coupling factor.
    if air_sea_exchange_timescale > 0.0:
        exchange_factor = (
            dt_seconds / air_sea_exchange_timescale
        )
        exchange_factor = min(1.0, exchange_factor)
    else:
        exchange_factor = 1.0

    for y in range(1, ny - 1):

        latitude = 0.5 * np.pi - y * dlat
        cos_lat = abs(np.cos(latitude))

        dx = (
            planet_radius_m
            * max(cos_lat, 1.0e-6)
            * dlon
        )

        dy = planet_radius_m * dlat

        for x in range(nx):

            if landmask[y, x] > 0.5 or dem[y, x] >= 0.0:
                T_new[y, x] = T_old[y, x]
                continue

            # Periodic longitude indexing.
            xm = x - 1
            xp = x + 1

            if xm < 0:
                xm = nx - 2

            if xp >= nx - 1:
                xp = 1

            T = T_old[y, x]

            # ------------------------------------------
            # UPWIND TEMPERATURE ADVECTION
            # ------------------------------------------

            uo = ocean_u[y, x]
            vo = ocean_v[y, x]

            if uo >= 0.0:
                dTdx = (T - T_old[y, xm]) / dx
            else:
                dTdx = (T_old[y, xp] - T) / dx

            if vo >= 0.0:
                dTdy = (T - T_old[y + 1, x]) / dy
            else:
                dTdy = (T_old[y - 1, x] - T) / dy

            advective_tendency = (
                -uo * dTdx
                -vo * dTdy
            )

            # ------------------------------------------
            # AIR-SEA HEAT EXCHANGE
            # ------------------------------------------

            T_air = temperature[y, x]

            exchange_tendency = (
                (T_air - T)
                / air_sea_exchange_timescale
            )

            # ------------------------------------------
            # UPDATE
            # ------------------------------------------

            T_new[y, x] = T + dt_seconds * (
                advective_tendency
                + exchange_tendency
            )


    for x in range(nx - 1):
        if dem[0, x] < 0.0 and landmask[0, x] <= 0.5:
            T_new[0, x] = T_new[1, x]
        if dem[ny - 1, x] < 0.0 and landmask[ny - 1, x] <= 0.5:
            T_new[ny - 1, x] = T_new[ny - 2, x]

    # Close periodic longitude seam.
    T_new[0, nx - 1] = T_new[0, 0]
    T_new[ny - 1, nx - 1] = T_new[ny - 1, 0]
    return T_new
# ------------------------------------------------------------
# SOLAR DECLINATION
# ------------------------------------------------------------

@njit
def solar_declination(tilt, orbital_longitude):

    tilt_r = tilt * math.pi / 180.0
    lam_r = orbital_longitude * math.pi / 180.0

    return math.asin(
        math.sin(lam_r) * math.sin(tilt_r)
    )


# ------------------------------------------------------------
# SOLAR RADIATION AT ONE GRID CELL
#
# longitude : degrees
# latitude  : degrees
# hour      : 0...24
# ------------------------------------------------------------

@njit
def solar_radiation(
    solar_constant,
    eccentricity,
    perihelion,
    tilt,
    latitude,
    longitude,
    orbital_longitude,
    rotation_angle
):

    deg = math.pi / 180.0

    lat = latitude * deg
    lon = longitude * deg
    lam = orbital_longitude * deg
    peri = perihelion * deg
    tilt_r = tilt * deg

    # --------------------------------------------------------
    # Solar declination
    # --------------------------------------------------------

    decl = math.asin(
        math.sin(lam) * math.sin(tilt_r)
    )

    # --------------------------------------------------------
    # Orbital distance correction
    # --------------------------------------------------------

    rfac = (
        1.0 + eccentricity * math.cos(lam - peri)
    ) / (
        1.0 - eccentricity * eccentricity
    )

    # --------------------------------------------------------
    # LOCAL SOLAR HOUR ANGLE
    #
    # rotation_angle:
    # planetary rotation angle, degrees
    #
    # orbital_longitude:
    # orbital position, degrees
    #
    # Their difference determines where the Sun is
    # relative to the planetary surface.
    # --------------------------------------------------------

    H = (
        lon
        + rotation_angle * deg
        - lam
    )

    # wrap to [-pi, +pi]

    H = (H + math.pi) % (2.0 * math.pi) - math.pi

    # --------------------------------------------------------
    # Solar zenith angle
    # --------------------------------------------------------

    cosz = (
        math.sin(lat) * math.sin(decl)
        + math.cos(lat) * math.cos(decl) * math.cos(H)
    )

    if cosz <= 0.0:
        return 0.0

    return solar_constant * rfac * cosz





# ------------------------------------------------------------
# INITIAL CAPACITY + ALBEDO ARRAYS
# ------------------------------------------------------------

@njit
def initialize_properties(material):

    ny, nx = material.shape

    capacity = np.empty((ny, nx))
    albedo = np.empty((ny, nx))

    for y in range(ny):
        for x in range(nx):

            C, A = material_properties(material[y, x])

            capacity[y, x] = C
            albedo[y, x] = A

    return capacity, albedo


# ------------------------------------------------------------
# ONE ENERGY-BALANCE STEP
#
# This is the main Numba kernel.
# ------------------------------------------------------------

@njit
def energy_step(
    temperature, humidity,
    material,
    capacity,
    base_albedo,
    dt_seconds,

    solar_constant,
    eccentricity,
    perihelion,
    tilt,

    orbital_longitude,
    rotation_angle,
    pressure_Pa,
    co2_ppm,
    clouds_present,
    cloud_strength,
    cloud_raster,
    use_ice,
    ice_temperature
):

    ny, nx = temperature.shape

    new_temperature = np.empty_like(temperature)
    new_humidity = np.empty_like(humidity)
    for y in range(ny):

        # latitude from -90 to +90
        if ny == 1:
            lat = 0.0
        else:
            lat = -90.0 + 180.0 * y / (ny - 1)



        for x in range(nx):
            cloud = cloud_raster[y, x]
            ca = cloud_albedo(cloud)
            longitude = -180.0 + 360.0 * x / nx

            T = temperature[y, x]

            # ------------------------------------------------
            # solar radiation
            # ------------------------------------------------

            S = solar_radiation(
                solar_constant,
                eccentricity,
                perihelion,
                tilt,
                lat,
                longitude,
                orbital_longitude,
                rotation_angle
            )

            # ------------------------------------------------
            # surface albedo
            # ------------------------------------------------

            A0 = base_albedo[y, x]

            if (clouds_present>0):

                # cloud albedo mixed with surface albedo
                A = (
                    A0 * (1.0 - cloud * cloud_strength)
                    + ca * cloud * cloud_strength
                )

            else:
                A = A0

            if A < 0.0:
                A = 0.0

            if A > 1.0:
                A = 1.0

            # ------------------------------------------------
            # absorbed shortwave
            # ------------------------------------------------

            ASR = (1.0 - A) * S

            # ------------------------------------------------
            # outgoing longwave radiation
            # ------------------------------------------------

            #OLR = emissivity * SIGMA * T**4
            # ------------------------------------------------
            # outgoing longwave radiation
            # ------------------------------------------------
            # CO2:n osapaine
            #pCO2 = pressure_Pa * co2_ppm / 1000000.0
            # Referenssi-CO2:n osapaine
            #pCO2_ref = 101325.0 * 420.0 / 1000000.0
            # CO2:n aiheuttama säteilypakote
            #CO2_forcing = 5.35 * math.log(pCO2 / pCO2_ref)
            # Muutetaan forcing efektiivisen emissiivisyyden muutokseksi
            #emissivity_CO2 = emissivity - CO2_forcing / (SIGMA * T**4)
            #print(emissivity_CO2)
            # OLR
            #OLR = emissivity_CO2 * SIGMA * T**4
            # ------------------------------------------------
            # CO2:n vaikutus OLR:ään
            # ------------------------------------------------
            # CO2:n tilavuusosuus
            co2_fraction = co2_ppm / 1_000_000.0
            # CO2:n osapaine [Pa]
            pCO2 = pressure_Pa * co2_fraction
            # Referenssitila:
            # 1 atm, 420 ppm ja emissiivisyys 0.62
            P_REF = 101325.0
            CO2_REF = 420.0
            EPSILON_REF = 0.62
            pCO2_ref = P_REF * CO2_REF / 1_000_000.0
            # Referenssin optinen paksuus.
            # Harmaan ilmakehän likiarvossa:
            # epsilon = 1 - exp(-tau)
            #tau_ref = -math.log(1.0 - EPSILON_REF)
            tau_ref = -math.log(1.0 - EPSILON_REF)
            # CO2-kolonnin kasvu.
            #
            # Eksponentti < 1 ottaa karkeasti huomioon sen,
            # että absorptio ei kasva lineaarisesti CO2-määrän mukana.
            #
            # 0.5 = sqrt-skaalaus
            co2_ratio = max( pCO2_ref/pCO2 , 1e-30)
            tau_CO2 = tau_ref * math.sqrt(co2_ratio)
            # Efektiivinen CO2-emissiivisyys
            #epsilon_CO2 = 1.0 - math.exp(-tau_CO2)
            epsilon_CO2 = 1.0 - math.exp(-tau_CO2)
            # OLR
            OLR = epsilon_CO2 * SIGMA * T**4

            # ------------------------------------------------
            # energy balance
            # ------------------------------------------------

            dT = dt_seconds / capacity[y, x] * (
                ASR - OLR
            )

            Tnew = T + dT

            # ------------------------------------------------
            # ice feedback
            #
            # This changes material for the NEXT timestep.
            # ------------------------------------------------

            if use_ice:

                if material[y, x] == 0:
                    if Tnew < ice_temperature:
                        material[y, x] = 2

                elif material[y, x] == 2:
                    if Tnew >= ice_temperature:
                        material[y, x] = 0

                elif material[y, x] == 1:
                    if Tnew < ice_temperature:
                        material[y, x] = 3

                elif material[y, x] == 3:
                    if Tnew >= ice_temperature:
                        material[y, x] = 1

            new_temperature[y, x] = Tnew

    return new_temperature


# ------------------------------------------------------------
# OPTIONAL SIMPLE DIFFUSION
#
# This is deliberately separate from the energy balance.
# diffusion = 0 means no horizontal heat transport.
# ------------------------------------------------------------

@njit
def diffuse_temperature_origo_nosphere(T, diffusion):

    ny, nx = T.shape

    out = T.copy()

    if diffusion <= 0.0:
        return out

    for y in range(1, ny - 1):

        for x in range(nx):

            xm = x - 1
            xp = x + 1

            # periodic longitude
            if xm < 0:
                xm = nx - 1

            if xp >= nx:
                xp = 0

            laplace = (
                T[y, xm]
                + T[y, xp]
                + T[y - 1, x]
                + T[y + 1, x]
                - 4.0 * T[y, x]
            )

            out[y, x] = T[y, x] + diffusion * laplace

    return out


@njit
def diffuse_temperature(
    T,
    material,
    K0,
    dt,
    k_ocean=1.0,
    k_land=0.15,
    k_sea_ice=0.10,
    k_snow_ice=0.05,
):
    ny, nx = T.shape
    R = 6_371_000.0

    if K0 <= 0.0 or dt <= 0.0:
        return T.copy()

    dlat = np.pi / ny
    dlon = 2.0 * np.pi / nx

    # Material heat capacity per unit surface area [J/m²/K]
    Cmat = np.array([
        1030.0 * 4184.0 * 10.0,  # ocean
        2700.0 * 750.0 * 2.0,    # land
        917.0 * 2100.0 * 3.0,    # sea ice
        400.0 * 880.0 * 3.0,     # snow / land ice
    ])

    kmat = np.array([
        k_ocean,
        k_land,
        k_sea_ice,
        k_snow_ice,
    ])

    # Effective energy-transport coefficient [J/(s K)]
    # per unit geometry factor
    Gmat = Cmat * K0 * kmat

    phi_edge = np.empty(ny + 1)
    area = np.empty(ny)

    for y in range(ny + 1):
        phi_edge[y] = -0.5 * np.pi + y * dlat

    for y in range(ny):
        area[y] = (
            R * R * dlon
            * (
                np.sin(phi_edge[y + 1])
                - np.sin(phi_edge[y])
            )
        )

    H = np.empty((ny, nx))
    G = np.empty((ny, nx))

    for y in range(ny):
        for x in range(nx):
            m = material[y, x]
            if m < 0 or m > 3:
                m = 1

            H[y, x] = Cmat[m] * area[y]
            G[y, x] = Gmat[m]

    # Face conductances: W/K
    # Store each face once so the energy exchange is equal/opposite.
    Ge = np.zeros((ny, nx))
    Gn = np.zeros((ny - 1, nx))

    for y in range(ny):
        phi = -0.5 * np.pi + (y + 0.5) * dlat
        distance = R * np.cos(phi) * dlon
        face_area = R * dlat

        for x in range(nx):
            xp = (x + 1) % nx
            g1 = G[y, x]
            g2 = G[y, xp]

            if g1 > 0.0 and g2 > 0.0:
                gf = 2.0 * g1 * g2 / (g1 + g2)
                Ge[y, x] = gf * face_area / distance

    distance_ns = R * dlat

    for y in range(ny - 1):
        phi = phi_edge[y + 1]
        face_area = R * np.cos(phi) * dlon

        for x in range(nx):
            g1 = G[y, x]
            g2 = G[y + 1, x]

            if g1 > 0.0 and g2 > 0.0:
                gf = 2.0 * g1 * g2 / (g1 + g2)
                Gn[y, x] = gf * face_area / distance_ns

    # Estimate the maximum explicit update rate [1/s].
    max_rate = 0.0

    for y in range(ny):
        for x in range(nx):
            xp = (x + 1) % nx
            xm = (x - 1) % nx

            rate = (
                Ge[y, x] + Ge[y, xm]
            ) / H[y, x]

            if y > 0:
                rate += Gn[y - 1, x] / H[y, x]

            if y < ny - 1:
                rate += Gn[y, x] / H[y, x]

            if rate > max_rate:
                max_rate = rate

    # Safety factor for explicit diffusion.
    nsub = max(1, int(math.ceil(dt * max_rate / 0.45)))
    sub_dt = dt / nsub

    out = T.copy()

    for step in range(nsub):
        new = out.copy()

        for y in range(ny):
            for x in range(nx):
                xp = (x + 1) % nx
                xm = (x - 1) % nx

                # East face: cell -> east neighbour
                q = (
                    sub_dt * Ge[y, x]
                    * (out[y, x] - out[y, xp])
                )
                new[y, x] -= q / H[y, x]
                new[y, xp] += q / H[y, xp]

                # North face: cell -> north neighbour
                if y < ny - 1:
                    q = (
                        sub_dt * Gn[y, x]
                        * (out[y, x] - out[y + 1, x])
                    )
                    new[y, x] -= q / H[y, x]
                    new[y + 1, x] += q / H[y + 1, x]

        out = new

    return out

@njit
def diffuse_temperature_material_01(
    T,
    material,
    K0,
    dt,
    k_ocean=1.0,
    k_land=0.15,
    k_sea_ice=0.10,
    k_snow_ice=0.05,
):
    ny, nx = T.shape

    R = 6_371_000.0
    dlat = np.pi / ny
    dlon = 2.0 * np.pi / nx

    out = T.copy()

    if K0 <= 0.0 or dt <= 0.0:
        return out

    # -----------------------------------------
    # Material properties
    # C: areal heat capacity [J / m² / K]
    # -----------------------------------------

    C_material = np.empty(4)
    k_material = np.empty(4)

    C_material[0] = 1030.0 * 4184.0 * 10.0
    C_material[1] = 2700.0 * 750.0 * 2.0
    C_material[2] = 917.0 * 2100.0 * 3.0
    C_material[3] = 400.0 * 880.0 * 3.0

    k_material[0] = k_ocean
    k_material[1] = k_land
    k_material[2] = k_sea_ice
    k_material[3] = k_snow_ice

    # -----------------------------------------
    # Latitude edges and cell areas
    # -----------------------------------------

    phi_edge = np.empty(ny + 1)
    area = np.empty(ny)

    for y in range(ny + 1):
        phi_edge[y] = -0.5 * np.pi + y * dlat

    for y in range(ny):
        area[y] = (
            R * R * dlon
            * (
                np.sin(phi_edge[y + 1])
                - np.sin(phi_edge[y])
            )
        )

    # -----------------------------------------
    # Cell heat capacities [J / K]
    # -----------------------------------------

    H = np.empty((ny, nx))

    for y in range(ny):
        for x in range(nx):
            m = material[y, x]

            if m < 0 or m > 3:
                m = 1

            H[y, x] = C_material[m] * area[y]

    # -----------------------------------------
    # EAST / WEST
    # -----------------------------------------

    for y in range(ny):

        phi = -0.5 * np.pi + (y + 0.5) * dlat

        cos_phi = np.cos(phi)

        if cos_phi < 1e-12:
            continue

        distance = R * cos_phi * dlon
        face_area = R * dlat

        for x in range(nx):

            xp = (x + 1) % nx

            m1 = material[y, x]
            m2 = material[y, xp]

            if m1 < 0 or m1 > 3:
                m1 = 1
            if m2 < 0 or m2 > 3:
                m2 = 1

            k1 = k_material[m1]
            k2 = k_material[m2]

            if k1 <= 0.0 or k2 <= 0.0:
                continue

            # Harmonic mean at material interface
            k_face = 2.0 * k1 * k2 / (k1 + k2)

            # Effective energy transferred between cells [J]
            Q = (
                K0 * k_face * dt
                * (T[y, x] - T[y, xp])
                / distance
                * face_area
            )

            out[y, x] -= Q / H[y, x]
            out[y, xp] += Q / H[y, xp]

    # -----------------------------------------
    # NORTH / SOUTH
    # -----------------------------------------

    distance = R * dlat

    for y in range(ny - 1):

        phi = phi_edge[y + 1]
        face_area = R * np.cos(phi) * dlon

        for x in range(nx):

            m1 = material[y, x]
            m2 = material[y + 1, x]

            if m1 < 0 or m1 > 3:
                m1 = 1
            if m2 < 0 or m2 > 3:
                m2 = 1

            k1 = k_material[m1]
            k2 = k_material[m2]

            if k1 <= 0.0 or k2 <= 0.0:
                continue

            k_face = 2.0 * k1 * k2 / (k1 + k2)

            Q = (
                K0 * k_face * dt
                * (T[y, x] - T[y + 1, x])
                / distance
                * face_area
            )

            out[y, x] -= Q / H[y, x]
            out[y + 1, x] += Q / H[y + 1, x]

    return out


@njit
def diffuse_temperature_material_00(T, material, K0, dt):

    ny, nx = T.shape

    R = 6_371_000.0

    dlat = np.pi / ny
    dlon = 2.0 * np.pi / nx

    out = T.copy()

    if K0 <= 0.0 or dt <= 0.0:
        return out

    # --------------------------------------------------
    # Relative material diffusivity multipliers
    # --------------------------------------------------

    k_material = np.empty(4)

    k_material[0] = 1.00  # Ocean
    k_material[1] = 0.15  # Land
    k_material[2] = 0.10  # Sea ice
    k_material[3] = 0.05  # Snow / land ice

    # --------------------------------------------------
    # Latitude edges and cell areas
    # --------------------------------------------------

    phi_edge = np.empty(ny + 1)
    area = np.empty(ny)

    for y in range(ny + 1):
        phi_edge[y] = -0.5 * np.pi + y * dlat

    for y in range(ny):
        area[y] = (
            R * R * dlon
            * (
                np.sin(phi_edge[y + 1])
                - np.sin(phi_edge[y])
            )
        )

    factor = K0 * dt

    # --------------------------------------------------
    # EAST / WEST
    # --------------------------------------------------

    for y in range(ny):

        phi = -0.5 * np.pi + (y + 0.5) * dlat

        distance = R * np.cos(phi) * dlon
        face_area = R * dlat

        for x in range(nx):

            xp = (x + 1) % nx

            m1 = material[y, x]
            m2 = material[y, xp]

            k1 = k_material[m1]
            k2 = k_material[m2]

            # Harmonic mean across material boundary
            if k1 + k2 > 0.0:
                k_face = 2.0 * k1 * k2 / (k1 + k2)
            else:
                k_face = 0.0

            gradient = (
                T[y, x] - T[y, xp]
            ) / distance

            flux = (
                factor * k_face
                * gradient * face_area
            )

            out[y, x] -= flux / area[y]
            out[y, xp] += flux / area[y]

    # --------------------------------------------------
    # NORTH / SOUTH
    # --------------------------------------------------

    distance = R * dlat

    for y in range(ny - 1):

        phi = phi_edge[y + 1]
        face_area = R * np.cos(phi) * dlon

        for x in range(nx):

            m1 = material[y, x]
            m2 = material[y + 1, x]

            k1 = k_material[m1]
            k2 = k_material[m2]

            if k1 + k2 > 0.0:
                k_face = 2.0 * k1 * k2 / (k1 + k2)
            else:
                k_face = 0.0

            gradient = (
                T[y, x] - T[y + 1, x]
            ) / distance

            flux = (
                factor * k_face
                * gradient * face_area
            )

            out[y, x] -= flux / area[y]
            out[y + 1, x] += flux / area[y + 1]

    return out

   
@njit
def diffuse_temperature_uniformal(T, K, dt):

    ny, nx = T.shape

    R = 6_371_000.0

    dlat = np.pi / ny
    dlon = 2.0 * np.pi / nx

    out = T.copy()

    if K <= 0.0 or dt <= 0.0:
        return out

    # ----------------------------------------------------------
    # Latitude boundaries
    # ----------------------------------------------------------

    phi_edge = np.empty(ny + 1)

    for y in range(ny + 1):
        phi_edge[y] = -0.5 * np.pi + y * dlat

    # ----------------------------------------------------------
    # Cell areas
    # ----------------------------------------------------------

    area = np.empty(ny)

    for y in range(ny):

        area[y] = (
            R * R
            * dlon
            * (
                np.sin(phi_edge[y + 1])
                - np.sin(phi_edge[y])
            )
        )

    factor = K * dt

    # ==========================================================
    # EAST / WEST
    # ==========================================================

    for y in range(ny):

        phi = (
            -0.5 * np.pi
            + (y + 0.5) * dlat
        )

        cos_phi = np.cos(phi)

        distance = (
            R
            * cos_phi
            * dlon
        )

        face_area = R * dlat

        for x in range(nx):

            xp = x + 1

            if xp >= nx:
                xp = 0

            # Temperature gradient A -> B
            gradient = (
                T[y, x] - T[y, xp]
            ) / distance

            flux = (
                factor
                * gradient
                * face_area
            )

            # A loses heat
            out[y, x] -= flux / area[y]

            # B gains heat
            out[y, xp] += flux / area[y]

    # ==========================================================
    # NORTH / SOUTH
    # ==========================================================

    for y in range(ny - 1):

        phi = phi_edge[y + 1]

        face_area = (
            R
            * np.cos(phi)
            * dlon
        )

        distance = R * dlat

        for x in range(nx):

            gradient = (
                T[y, x] - T[y + 1, x]
            ) / distance

            flux = (
                factor
                * gradient
                * face_area
            )

            # South cell loses
            out[y, x] -= flux / area[y]

            # North cell gains
            out[y + 1, x] += (
                flux / area[y + 1]
            )

    return out







import numpy as np
import matplotlib.pyplot as plt
from numba import njit


# ============================================================
# CONSTANTS
# ============================================================

G_CONST = 6.67430e-11       # m^3 kg^-1 s^-2
R_AIR = 287.05              # J kg^-1 K^-1
P_REF = 101325.0            # Pa

#Yksi tärkeä huomio ennen ajoa

#Tässä versiossa upper_wind_strength=1.0 ja vertical_circulation_strength=10000.0 ovat nyt kaksi tärkeintä säätöparametria.

#Jos ylävirta on liian heikkoa:

#upper_wind_strength=2.0

#Jos pystysuuntainen kierto on liian heikkoa:

#vertical_circulation_strength=20000.0




import numpy as np
from numba import njit


@njit
def evaporation_step(
    humidity,
    temperature,
    landmask,
    soil_moisture,
    dt_seconds,

    # Haihtumisen säätöparametrit
    ocean_evaporation_rate=2.0e-5,
    land_evaporation_rate=1.0e-5,
    humidity_reference=0.020,
    reference_temperature=288.0,
    temperature_sensitivity=0.04,
):
    """
    Simple surface evaporation model.

    humidity:
        Specific humidity, kg/kg.

    soil_moisture:
        Relative soil water availability, 0...1.

    evaporation rates:
        Approximate humidity tendency in kg/kg/s.

    Returns:
        New humidity field.
    """

    ny, nx = humidity.shape

    humidity_new = humidity.copy()

    for y in range(ny):
        for x in range(nx):

            T = temperature[y, x]
            q = humidity[y, x]

            if T < 1.0:
                T = 1.0

            # Temperature dependence.
            temp_factor = np.exp(
                temperature_sensitivity
                * (T - reference_temperature)
            )

            # Limit extreme temperature scaling.
            if temp_factor > 3.0:
                temp_factor = 3.0

            if temp_factor < 0.1:
                temp_factor = 0.1

            # Surface water availability.
            if landmask[y, x] > 0.5:
                water_factor = soil_moisture[y, x]

                if water_factor < 0.0:
                    water_factor = 0.0
                elif water_factor > 1.0:
                    water_factor = 1.0

                base_rate = land_evaporation_rate
            else:
                water_factor = 1.0
                base_rate = ocean_evaporation_rate

            # Evaporation weakens as air approaches
            # the reference humidity.
            humidity_deficit = (
                humidity_reference - q
            )

            if humidity_deficit < 0.0:
                humidity_deficit = 0.0

            evaporation = (
                base_rate
                * temp_factor
                * water_factor
                * humidity_deficit
                / humidity_reference
            )

            humidity_new[y, x] += (
                evaporation * dt_seconds
            )

    return humidity_new


@njit
def advect_temperature_humidity(
    temperature,
    humidity,
    u,
    v,
    dem,
    planet_radius_m,
    dt_seconds,
):
    ny, nx = temperature.shape

    dlat = np.pi / (ny - 1)
    dlon = 2.0 * np.pi / (nx - 1)

    temperature_new = temperature.copy()
    humidity_new = humidity.copy()

    for y in range(1, ny - 1):

        latitude = 0.5 * np.pi - y * dlat
        cos_lat = abs(np.cos(latitude))

        for x in range(nx - 1):

            xm = x - 1
            xp = x + 1

            if xm < 0:
                xm = nx - 2

            if xp >= nx - 1:
                xp = 0

            r = planet_radius_m + dem[y, x]

            if r <= 0.0:
                continue

            dx = max(1.0, r * cos_lat * dlon)
            dy = max(1.0, r * dlat)

            # Temperature: upwind gradients.
            if u[y, x] >= 0.0:
                dTdx = (
                    temperature[y, x]
                    - temperature[y, xm]
                ) / dx
            else:
                dTdx = (
                    temperature[y, xp]
                    - temperature[y, x]
                ) / dx

            if v[y, x] >= 0.0:
                dTdy = (
                    temperature[y, x]
                    - temperature[y + 1, x]
                ) / dy
            else:
                dTdy = (
                    temperature[y - 1, x]
                    - temperature[y, x]
                ) / dy

            # Humidity: upwind gradients.
            if u[y, x] >= 0.0:
                dqdx = (
                    humidity[y, x]
                    - humidity[y, xm]
                ) / dx
            else:
                dqdx = (
                    humidity[y, xp]
                    - humidity[y, x]
                ) / dx

            if v[y, x] >= 0.0:
                dqdy = (
                    humidity[y, x]
                    - humidity[y + 1, x]
                ) / dy
            else:
                dqdy = (
                    humidity[y - 1, x]
                    - humidity[y, x]
                ) / dy

            temperature_new[y, x] = (
                temperature[y, x]
                - dt_seconds * (
                    u[y, x] * dTdx
                    + v[y, x] * dTdy
                )
            )

            humidity_new[y, x] = (
                humidity[y, x]
                - dt_seconds * (
                    u[y, x] * dqdx
                    + v[y, x] * dqdy
                )
            )

            if humidity_new[y, x] < 0.0:
                humidity_new[y, x] = 0.0

    # Periodic longitude seam.
    for y in range(ny):
        temperature_new[y, nx - 1] = temperature_new[y, 0]
        humidity_new[y, nx - 1] = humidity_new[y, 0]
    
    return temperature_new, humidity_new


@njit
def atmosphere_step_v7(
    temperature, humidity,
    pressure_raster,

    u,
    v,

    u_upper,
    v_upper,

    vertical_velocity,

    dem,
    landmask,

    planet_radius_m,
    planet_mass_kg,
    rotation_period_days,

    dt_seconds,

    # ========================================================
    # UPPER ATMOSPHERE
    # ========================================================

    upper_drag_timescale=172800.0,

    #upper_wind_strength=1.0,

    # ========================================================
    # UPPER -> VERTICAL CIRCULATION
    # ========================================================

    #vertical_circulation_strength=10000.0,
    upper_wind_strength=0.0003,

    # ========================================================
    # UPPER -> VERTICAL CIRCULATION
    # ========================================================

    vertical_circulation_strength=100.0,
    # ========================================================
    # SURFACE DRAG
    # ========================================================

    surface_drag_timescale_land=21600.0,
    surface_drag_timescale_ocean=86400.0,

    # ========================================================
    # MASS -> SURFACE PRESSURE
    # ========================================================

    pressure_mass_coupling=2.0e-5,

    # ========================================================
    # CORIOLIS
    # ========================================================

    equatorial_fmin=2.0e-6,

    # ========================================================
    # SAFETY
    # ========================================================

    max_wind_speed=100.0,
    min_pressure=100.0,
):
    """
    V7 two-layer atmospheric model.

    Surface:
        pressure_raster
        u
        v

    Upper atmosphere:
        u_upper
        v_upper

    Vertical circulation:
        vertical_velocity

    Main coupling:

        temperature
            ↓
        upper thermal wind
            ↓
        upper divergence
            ↓
        vertical motion
            ↓
        surface pressure
            ↓
        surface wind

    Temperature does NOT directly modify surface pressure.
    """

    ny, nx = temperature.shape
    speed=u
    # ========================================================
    # ROTATION
    # ========================================================

    omega = (
        2.0
        * np.pi
        / (
            rotation_period_days
            * 86400.0
        )
    )

    # ========================================================
    # GRID
    # ========================================================

    dlat = np.pi / (ny - 1)
    dlon = 2.0 * np.pi / (nx - 1)

    # ========================================================
    # RESET VERTICAL VELOCITY
    #
    # It will now be determined ONLY from upper-level
    # divergence.
    # ========================================================

    for y in range(ny):

        for x in range(nx):

            vertical_velocity[y, x] = 0.0

    # ========================================================
    # STEP 1
    #
    # UPPER ATMOSPHERE
    #
    # Temperature gradients drive upper-level flow.
    # ========================================================

    for y in range(1, ny - 1):

        latitude = (
            0.5 * np.pi
            - y * dlat
        )

        sin_lat = np.sin(
            latitude
        )

        cos_lat = np.cos(
            latitude
        )

        # ----------------------------------------------------
        # Coriolis
        # ----------------------------------------------------

        f = (
            2.0
            * omega
            * sin_lat
        )

        if np.abs(f) < equatorial_fmin:

            if f >= 0.0:
                f_eff = equatorial_fmin
            else:
                f_eff = -equatorial_fmin

        else:

            f_eff = f

        for x in range(nx):

            # ------------------------------------------------
            # PERIODIC LONGITUDE
            # ------------------------------------------------

            xm = x - 1
            xp = x + 1

            if xm < 0:
                xm = nx - 2

            if xp >= nx - 1:
                xp = 1

            # ------------------------------------------------
            # LOCAL RADIUS
            # ------------------------------------------------

            r = (
                planet_radius_m
                + dem[y, x]
            )

            if r <= 0.0:
                continue

            # ------------------------------------------------
            # GRID DISTANCES
            # ------------------------------------------------

            dx = (
                r
                * np.abs(cos_lat)
                * dlon
            )

            dy = (
                r
                * dlat
            )

            if dx < 1.0:
                dx = 1.0

            if dy < 1.0:
                dy = 1.0

            # =================================================
            # TEMPERATURE GRADIENT
            # =================================================

            dTdx = (
                temperature[y, xp]
                - temperature[y, xm]
            ) / (
                2.0 * dx
            )

            dTdy = (
                temperature[y - 1, x]
                - temperature[y + 1, x]
            ) / (
                2.0 * dy
            )

            # =================================================
            # THERMAL ACCELERATION
            # =================================================

            thermal_ax = (
                -R_AIR
                * dTdx
                * upper_wind_strength
            )

            thermal_ay = (
                -R_AIR
                * dTdy
                * upper_wind_strength
            )

            # =================================================
            # UPPER MOMENTUM BALANCE
            #
            # 0 = F_u + f*v - u/tau
            # 0 = F_v - f*u - v/tau
            # =================================================

            ft = (
                f_eff
                * upper_drag_timescale
            )

            denominator = (
                1.0
                + ft * ft
            )

            uu = (
                upper_drag_timescale
                * (
                    thermal_ax
                    + ft * thermal_ay
                )
                / denominator
            )

            vv = (
                upper_drag_timescale
                * (
                    thermal_ay
                    - ft * thermal_ax
                )
                / denominator
            )

            # =================================================
            # WIND LIMIT
            # =================================================

            s = np.sqrt(
                uu * uu
                + vv * vv
            )

            if s > max_wind_speed:

                factor = (
                    max_wind_speed
                    / s
                )

                uu *= factor
                vv *= factor

            u_upper[y, x] = uu
            v_upper[y, x] = vv

    # ========================================================
    # STEP 2
    #
    # UPPER-LEVEL DIVERGENCE
    #
    # div(U) > 0
    #     upper divergence
    #     -> rising air
    #
    # div(U) < 0
    #     upper convergence
    #     -> sinking air
    # ========================================================

    for y in range(1, ny - 1):

        latitude = (
            0.5 * np.pi
            - y * dlat
        )

        cos_lat = np.cos(
            latitude
        )

        for x in range(nx):

            # ------------------------------------------------
            # PERIODIC LONGITUDE
            # ------------------------------------------------

            xm = x - 1
            xp = x + 1

            if xm < 0:
                xm = nx - 2

            if xp >= nx - 1:
                xp = 1

            # ------------------------------------------------
            # LOCAL RADIUS
            # ------------------------------------------------

            r = (
                planet_radius_m
                + dem[y, x]
            )

            if r <= 0.0:
                continue

            dx = (
                r
                * np.abs(cos_lat)
                * dlon
            )

            dy = (
                r
                * dlat
            )

            if dx < 1.0:
                dx = 1.0

            if dy < 1.0:
                dy = 1.0

            # =================================================
            # UPPER HORIZONTAL DIVERGENCE
            # =================================================

            du_dx = (
                u_upper[y, xp]
                - u_upper[y, xm]
            ) / (
                2.0 * dx
            )

            dv_dy = (
                v_upper[y - 1, x]
                - v_upper[y + 1, x]
            ) / (
                2.0 * dy
            )

            divergence = (
                du_dx
                + dv_dy
            )

            # =================================================
            # VERTICAL CIRCULATION
            #
            # Upper divergence -> rising
            # Upper convergence -> sinking
            # =================================================

            vertical_velocity[y, x] = (
                -vertical_circulation_strength
                * divergence
            )

    # ========================================================
    # STEP 3
    #
    # SURFACE PRESSURE TENDENCY
    #
    # Rising:
    #     pressure decreases
    #
    # Sinking:
    #     pressure increases
    # ========================================================

    pressure_tendency = np.zeros(
        (ny, nx),
        dtype=np.float64
    )

    for y in range(1, ny - 1):

        for x in range(nx):

            w = (
                vertical_velocity[y, x]
            )

            pressure_tendency[y, x] = (
                -pressure_mass_coupling
                * pressure_raster[y, x]
                * w
            )

    # ========================================================
    # STEP 4
    #
    # REMOVE GLOBAL PRESSURE DRIFT
    #
    # Keeps the global mean pressure approximately constant.
    # ========================================================

    mean_tendency = 0.0
    tendency_count = 0

    for y in range(1, ny - 1):

        for x in range(nx):

            mean_tendency += (
                pressure_tendency[y, x]
            )

            tendency_count += 1

    if tendency_count > 0:

        mean_tendency /= (
            tendency_count
        )

    for y in range(1, ny - 1):

        for x in range(nx):

            pressure_tendency[y, x] -= (
                mean_tendency
            )

    # ========================================================
    # STEP 5
    #
    # UPDATE SURFACE PRESSURE
    # ========================================================

    for y in range(1, ny - 1):

        for x in range(nx):

            pressure_raster[y, x] += (
                pressure_tendency[y, x]
                * dt_seconds
            )

            if pressure_raster[y, x] < min_pressure:

                pressure_raster[y, x] = (
                    min_pressure
                )

    # ========================================================
    # STEP 6
    #
    # SURFACE WIND FROM PRESSURE GRADIENT
    # ========================================================

    for y in range(1, ny - 1):

        latitude = (
            0.5 * np.pi
            - y * dlat
        )

        sin_lat = np.sin(
            latitude
        )

        cos_lat = np.cos(
            latitude
        )

        f = (
            2.0
            * omega
            * sin_lat
        )

        if np.abs(f) < equatorial_fmin:

            if f >= 0.0:
                f_eff = equatorial_fmin
            else:
                f_eff = -equatorial_fmin

        else:

            f_eff = f

        for x in range(nx):

            # ------------------------------------------------
            # PERIODIC LONGITUDE
            # ------------------------------------------------

            xm = x - 1
            xp = x + 1

            if xm < 0:
                xm = nx - 2

            if xp >= nx - 1:
                xp = 1

            # ------------------------------------------------
            # LOCAL RADIUS
            # ------------------------------------------------

            r = (
                planet_radius_m
                + dem[y, x]
            )

            if r <= 0.0:
                continue

            dx = (
                r
                * np.abs(cos_lat)
                * dlon
            )

            dy = (
                r
                * dlat
            )

            if dx < 1.0:
                dx = 1.0

            if dy < 1.0:
                dy = 1.0

            # =================================================
            # PRESSURE GRADIENT
            # =================================================

            dpdx = (
                pressure_raster[y, xp]
                - pressure_raster[y, xm]
            ) / (
                2.0 * dx
            )

            dpdy = (
                pressure_raster[y - 1, x]
                - pressure_raster[y + 1, x]
            ) / (
                2.0 * dy
            )

            # =================================================
            # AIR DENSITY
            # =================================================

            T = temperature[y, x]

            if T < 1.0:
                T = 1.0

            rho = (
                pressure_raster[y, x]
                / (
                    R_AIR
                    * T
                )
            )

            if rho < 1.0e-8:
                continue

            # =================================================
            # PRESSURE ACCELERATION
            # =================================================

            pressure_ax = (
                -dpdx
                / rho
            )

            pressure_ay = (
                -dpdy
                / rho
            )

            # =================================================
            # SURFACE DRAG
            # =================================================

            if landmask[y, x] > 0.5:

                tau = (
                    surface_drag_timescale_land
                )

            else:

                tau = (
                    surface_drag_timescale_ocean
                )

            # =================================================
            # STEADY MOMENTUM BALANCE
            # =================================================

            ft = (
                f_eff
                * tau
            )

            denominator = (
                1.0
                + ft * ft
            )

            uu = (
                tau
                * (
                    pressure_ax
                    + ft * pressure_ay
                )
                / denominator
            )

            vv = (
                tau
                * (
                    pressure_ay
                    - ft * pressure_ax
                )
                / denominator
            )

            # =================================================
            # WIND LIMIT
            # =================================================

            s = np.sqrt(
                uu * uu
                + vv * vv
            )

            if s > max_wind_speed:

                factor = (
                    max_wind_speed
                    / s
                )

                uu *= factor
                vv *= factor

            u[y, x] = uu
            v[y, x] = vv

    # ========================================================
    # STEP 7
    #
    # SURFACE WIND SPEED
    # ========================================================

    for y in range(ny):

        for x in range(nx):

            speed[y, x] = np.sqrt(
                u[y, x] * u[y, x]
                + v[y, x] * v[y, x]
            )


    

    # ========================================================
    # RETURN
    # ========================================================

    return (
        pressure_raster, humidity, temperature,

        u,
        v,
        speed,

        u_upper,
        v_upper,

        vertical_velocity
    )






# ============================================================
# WIND VISUALIZATION
# ============================================================


def visualize_wind_v5(
    u,
    v,
    speed,
    quiver_step=8,
    stream_density=2.0,
):

    ny, nx = u.shape

    lon = np.linspace(
        -180.0,
        180.0,
        nx
    )

    lat = np.linspace(
        90.0,
        -90.0,
        ny
    )

    lon2, lat2 = np.meshgrid(
        lon,
        lat
    )

    fig, ax = plt.subplots(
        figsize=(16, 8)
    )

    # ========================================================
    # SPEED
    # ========================================================

    im = ax.imshow(
        speed,
        extent=(
            -180,
            180,
            -90,
            90
        ),
        origin="upper",
        cmap="turbo",
        aspect="auto"
    )

    cb = fig.colorbar(
        im,
        ax=ax
    )

    cb.set_label(
        "Surface wind speed [m/s]"
    )

    # ========================================================
    # QUIVER
    # ========================================================

    sl = (
        slice(
            None,
            None,
            quiver_step
        ),
        slice(
            None,
            None,
            quiver_step
        )
    )

    q = ax.quiver(
        lon2[sl],
        lat2[sl],
        u[sl],
        v[sl],
        color="white",
        width=0.0015,
        alpha=0.9
    )

    ax.quiverkey(
        q,
        X=0.90,
        Y=1.03,
        U=10.0,
        label="10 m/s",
        labelpos="E"
    )

    # ========================================================
    # STREAMPLOT
    # ========================================================

    lat_sp = lat[::-1]

    u_sp = np.nan_to_num(
        u[::-1, :],
        nan=0.0
    )

    v_sp = np.nan_to_num(
        v[::-1, :],
        nan=0.0
    )

    ax.streamplot(
        lon,
        lat_sp,
        u_sp,
        v_sp,
        density=stream_density,
        color="black",
        linewidth=0.6,
        arrowsize=0.8
    )

    ax.set_xlim(
        -180,
        180
    )

    ax.set_ylim(
        -90,
        90
    )

    ax.set_xlabel(
        "Longitude [deg]"
    )

    ax.set_ylabel(
        "Latitude [deg]"
    )

    ax.set_title(
        "Planetary surface winds — V5"
    )

    ax.grid(
        alpha=0.2
    )

    plt.show()





def wind_latitude_debug(
    u,
    v,
    speed,
    temperature_K,
    dem,
    radius,
    mass,
    rotation_days,
    landmask=None,
):
    """
    Leveysastekohtainen tuulidiagnostiikka.

    Rasteri:
        (ny, nx)

    Latitude:
        +90 -> -90

    Palauttaa dictionaryn, jossa jokainen rivi vastaa
    yhtä rasterin leveysastetta.
    """

    ny, nx = u.shape

    # --------------------------------------------------------
    # Planeetan pyörimisnopeus
    # --------------------------------------------------------

    omega = (
        2.0 * np.pi
        / (rotation_days * 86400.0)
    )

    # --------------------------------------------------------
    # Latitudit
    # --------------------------------------------------------

    latitudes = np.linspace(
        90.0,
        -90.0,
        ny
    )

    # --------------------------------------------------------
    # Tulokset
    # --------------------------------------------------------

    result = {
        "latitude": [],
        "mean_u": [],
        "mean_v": [],
        "mean_speed": [],
        "median_speed": [],
        "rms_speed": [],
        "max_speed": [],

        "north_fraction": [],
        "south_fraction": [],

        "mean_temperature_K": [],
        "mean_temperature_C": [],

        "mean_dem_m": [],

        "coriolis_f": [],
    }

    # ========================================================
    # LOOP LATITUDES
    # ========================================================

    for y, lat in enumerate(latitudes):

        u_row = u[y]
        v_row = v[y]
        speed_row = speed[y]
        T_row = temperature_K[y]
        dem_row = dem[y]

        # ----------------------------------------------------
        # Valid values
        # ----------------------------------------------------

        valid = np.isfinite(
            speed_row
        )

        if not np.any(valid):
            continue

        u_valid = u_row[valid]
        v_valid = v_row[valid]
        speed_valid = speed_row[valid]

        T_valid = T_row[valid]
        dem_valid = dem_row[valid]

        # ----------------------------------------------------
        # Basic statistics
        # ----------------------------------------------------

        mean_u = np.mean(u_valid)
        mean_v = np.mean(v_valid)

        mean_speed = np.mean(
            speed_valid
        )

        median_speed = np.median(
            speed_valid
        )

        rms_speed = np.sqrt(
            np.mean(
                speed_valid
                * speed_valid
            )
        )

        max_speed = np.max(
            speed_valid
        )

        # ----------------------------------------------------
        # North / south flow
        # ----------------------------------------------------

        moving = (
            speed_valid > 1.0e-6
        )

        if np.any(moving):

            north_fraction = np.mean(
                v_valid[moving] > 0.0
            )

            south_fraction = np.mean(
                v_valid[moving] < 0.0
            )

        else:

            north_fraction = 0.0
            south_fraction = 0.0

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        mean_T = np.mean(
            T_valid
        )

        # ----------------------------------------------------
        # Terrain
        # ----------------------------------------------------

        mean_z = np.mean(
            dem_valid
        )

        # ----------------------------------------------------
        # Coriolis
        # ----------------------------------------------------

        phi = (
            np.radians(lat)
        )

        f = (
            2.0
            * omega
            * np.sin(phi)
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        result["latitude"].append(lat)

        result["mean_u"].append(mean_u)
        result["mean_v"].append(mean_v)

        result["mean_speed"].append(
            mean_speed
        )

        result["median_speed"].append(
            median_speed
        )

        result["rms_speed"].append(
            rms_speed
        )

        result["max_speed"].append(
            max_speed
        )

        result["north_fraction"].append(
            north_fraction
        )

        result["south_fraction"].append(
            south_fraction
        )

        result["mean_temperature_K"].append(
            mean_T
        )

        result["mean_temperature_C"].append(
            mean_T - 273.15
        )

        result["mean_dem_m"].append(
            mean_z
        )

        result["coriolis_f"].append(
            f
        )

    # --------------------------------------------------------
    # Convert to numpy arrays
    # --------------------------------------------------------

    for key in result:

        result[key] = np.asarray(
            result[key]
        )

    return result


    
# ============================================================
# SIMULATION
# ============================================================

def run_model(dem, landmask, 
    material,
    initial_temperature=288.15,
    initial_surface_pressure=101325.0,
    planet_radius_m=R_e,
    planet_mass_kg=M_e,
    solar_constant=1365.2,
    eccentricity=0.0167,
    perihelion=283.0,
    tilt=23.44,
    orbital_period_years=1.0,
    rotation_period_days=1.0,
    co2_ppm=278,
    years=1,
    dt_hours=1.0,
    clouds_present=1.0,
    cloud_strength=1.0,

    use_ice=True,

    ice_temperature=263.0,

    diffusion=0.0,

    save_every_hours=24
):

    # --------------------------------------------------------
    # arrays
    # --------------------------------------------------------

    ocean_depth = np.maximum(
    0.0,
    -dem
    ).astype(np.float64)

    material = np.asarray(
        material,
        dtype=np.int8
    ).copy()
    ocean_depth[landmask > 0.5] = 0.0
    ny, nx = material.shape

    temperature = np.full(
        (ny, nx),
        initial_temperature,
        dtype=np.float64
    )

    # Ominaiskosteus kg/kg
    humidity = np.full(
    temperature.shape,
    0.005,
    dtype=np.float64,
    )

    soil_moisture = np.ones_like(
    temperature,
    dtype=np.float64, 
    )

    # Kuivempi alkutila mantereille.
    for y in range(temperature.shape[0]):
        for x in range(temperature.shape[1]):
            if landmask[y, x] > 0.5:
                soil_moisture[y, x] = 0.5

    pressure_raster = np.full(
    (ny, nx),
    initial_surface_pressure,
    dtype=np.float64
    )
    u = np.zeros(
    (ny, nx),
    dtype=np.float64
    )

    v = np.zeros(
    (ny, nx),
    dtype=np.float64
    )
    u_upper = np.zeros(
    (ny, nx),
    dtype=np.float64
    )
    v_upper = np.zeros(
    (ny, nx),
    dtype=np.float64
    )
    vertical_velocity = np.zeros(
    (ny, nx),
    dtype=np.float64
    )
    capacity, base_albedo = initialize_properties(
        material
    )

    ocean_u_surface = np.zeros_like(dem, dtype=np.float64)
    ocean_v_surface = np.zeros_like(dem, dtype=np.float64)

    ocean_u_deep = np.zeros_like(dem, dtype=np.float64)
    ocean_v_deep = np.zeros_like(dem, dtype=np.float64)

    ocean_temperature_surface = np.full_like(
    dem, 288.15, dtype=np.float64
    )
    ocean_temperature_surface[landmask > 0.5] = 0.0
    ocean_temperature_deep = np.full_like(
    dem, 277.15, dtype=np.float64
    )

    # --------------------------------------------------------
    # timestep
    # --------------------------------------------------------

    dt_seconds = dt_hours * 3600.0

    total_hours = int(
        years * 365.25 * 24.0 / dt_hours
    )

    #print(orbital_period_years)
    #orbital_hours = rotation_period_days*orbital_period_years*365.25 * 24.0
    orbital_hours = (
    orbital_period_years
    * 365.25
    * 24.0
    )

    # saved output
    saved_temperature = []

    # --------------------------------------------------------
    # MAIN TIME LOOP
    # --------------------------------------------------------
    nn=-1
    for step in range(total_hours):
        nn=nn+1
        time_hours = step * dt_hours

        orbital_longitude = (
        360.0
        * time_hours
        / orbital_hours
        ) % 360.0
        rotation_angle = (
        360.0
        * time_hours
        / (rotation_period_days * 24.0)
        ) % 360.0
        # ----------------------------------------------------
        # energy balance
        # ----------------------------------------------------
        cloud_raster = create_cloud_raster(
        simu_ny,
        simu_nx,
        clouds_present,
        tilt,
        orbital_longitude)
        temperature= energy_step(
            temperature, humidity,
            material,
            capacity,
            base_albedo,

            dt_seconds,

            solar_constant,
            eccentricity,
            perihelion,
            tilt,

            orbital_longitude,
            rotation_angle,
            initial_surface_pressure,
            co2_ppm,
            clouds_present,
            cloud_strength,
            cloud_raster,
            use_ice,
            ice_temperature
        )



        # ----------------------------------------------------
        # optional horizontal heat transport
        # ----------------------------------------------------

        if diffusion > 0.0:
           temperature = diffuse_temperature(
           temperature,
           material,
           K0=diffusion,
           dt=dt_seconds,
           k_ocean=1.0,
           k_land=0.15,
           k_sea_ice=0.10,
           k_snow_ice=0.05,
           ) 

        use_humidity=True 
        use_winds=True
        use_humidity=True
        use_ocean_currents=True

        #use_humidity=False
        #use_winds=False
        #use_humidity=False
        #use_ocean_currents=False

        if (use_winds==True):
            (
            pressure_raster, humidity, temperature,
            u,
            v,
            wind_speed,
            u_upper,
            v_upper,
            vertical_velocity
            )=atmosphere_step_v7(
            temperature, humidity,
            pressure_raster,
            u,
            v,
            u_upper,
            v_upper,
            vertical_velocity,
            dem,
            landmask,
            planet_radius_m,
            planet_mass_kg,
            rotation_period_days,
            dt_seconds
            )
        #print(
        #"INIT temperature:",
        #np.min(temperature),
        #np.max(temperature),
        #"dt:",
        #dt_seconds,
        #)

        #print("INIT:", humidity.min(), humidity.max())
        if(use_humidity==True):
            temperature, humidity = advect_temperature_humidity(
            temperature,
            humidity,
            u,
            v,
            dem,
            planet_radius_m,
            dt_seconds,
            )
            #print("AFTER ADVECT:", humidity.min(), humidity.max())
            humidity = evaporation_step(
            humidity,
            temperature,
            landmask,
            soil_moisture,
            dt_seconds,
            )
            #print("AFTER EVAP:", humidity.min(), humidity.max())

        if(use_ocean_currents==True):
            ocean_u_surface, ocean_v_surface = (
            ocean_surface_current_step(
            ocean_u_surface,
            ocean_v_surface,
            u,
            v,
            dem,
            landmask,
            planet_radius_m,
            rotation_period_days,
            dt_seconds,
            )
            )

        #ocean_temperature_surface = ocean_temperature_step(
        #ocean_temperature_surface,
        #temperature,
        #ocean_u_surface,
        #ocean_v_surface,
        #dem,
        #landmask,
        #planet_radius_m,
        #dt_seconds,
        #)
        temperature = ocean_heat_transport_step(
        temperature,
        ocean_u_surface,
        ocean_v_surface,
        material,
        dem,
        dt_seconds,
        planet_radius_m,
        )

        #debug_ocean_currents(
        #ocean_u_surface,
        #ocean_v_surface,
        #dem,
        #landmask,
        #step=dt_seconds,
        #)
        if step % (24*30) == 0:
            speed = np.sqrt(
            ocean_u_surface**2 + ocean_v_surface**2
            )
            ocean = (dem < 0.0) & (landmask <= 0.5)
            #print(
            #"Day:", step // 24, 
            #"Ocean mean:", speed[ocean].mean(),
            #"Ocean max:", speed[ocean].max(),
            #)
        # ----------------------------------------------------
        # save
        # ----------------------------------------------------

        if step % int(save_every_hours / dt_hours) == 0:
            saved_temperature.append(
                temperature.copy()
            )

        #debug
        #print(
        #"temperature:",
        #np.min(temperature),
        #np.max(temperature),
        #"dt:",
        #dt_seconds,
        #)
        #print(
        #"humidity:",
        #np.min(humidity),
        #np.max(humidity),
        #np.mean(humidity),
        #)
        #print("Pressure:",pressure_raster.min(),pressure_raster.max(),pressure_raster.mean())
        #print("Wind:",wind_speed.min(),wind_speed.max(),wind_speed.mean())

    #visualize_wind(u,v,speed=None,temperature=None,quiver_step=10,stream_density=2.0,title="Planetary surface wind")
    #plt.imshow(pressure_raster)
    #plt.show()
    #plt.imshow(vertical_velocity)
    #plt.show()
    #plt.imshow(ocean_temperature_surface, cmap="coolwarm")
    #plt.imshow(humidity)
    #plt.show()
    return (
        temperature, humidity,
        material,
        np.array(saved_temperature)
    )





def visualize_wind(
    u,
    v,
    speed=None,
    temperature=None,
    quiver_step=10,
    stream_density=2.0,
    title="Planetary surface wind",
):
    """
    Visualisoi globaalin pintatuulikentän:

        1. Quiver
        2. Streamplot

    Rasterin muoto:
        (ny, nx)

    Maantieteellinen alue:
        longitude = -180 ... +180
        latitude  = +90 ... -90
    """

    ny, nx = u.shape

    # --------------------------------------------------------
    # Longitude / latitude
    # --------------------------------------------------------

    lon = np.linspace(-180.0, 180.0, nx)
    lat = np.linspace(90.0, -90.0, ny)

    # --------------------------------------------------------
    # Nopeus
    # --------------------------------------------------------

    if speed is None:
        speed = np.sqrt(u * u + v * v)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(16, 8),
        constrained_layout=True
    )

    # --------------------------------------------------------
    # Taustaväri = tuulen nopeus
    # --------------------------------------------------------

    im = ax.imshow(
        speed,
        extent=(-180, 180, -90, 90),
        origin="upper",
        cmap="turbo",
        aspect="auto",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Wind speed [m/s]")

    # --------------------------------------------------------
    # QUÍVER
    # --------------------------------------------------------

    yy, xx = np.meshgrid(
        lat,
        lon,
        indexing="ij"
    )

    # Harvennetaan nuolet
    sl = (
        slice(None, None, quiver_step),
        slice(None, None, quiver_step)
    )

    q = ax.quiver(
        xx[sl],
        yy[sl],
        u[sl],
        v[sl],
        color="white",
        scale=None,
        width=0.0015,
        alpha=0.85,
    )

    ax.quiverkey(
        q,
        X=0.90,
        Y=1.03,
        U=10,
        label="10 m/s",
        labelpos="E",
    )

    # --------------------------------------------------------
    # STREAMPLOT
    # --------------------------------------------------------
    #
    # streamplot tarvitsee x:n kasvamaan vasemmalta oikealle
    # ja y:n alhaalta ylös.
    #
    # Alkuperäinen rasteri on:
    # y = +90 -> -90
    #
    # Käännetään siksi rasterit streamplotia varten.
    # --------------------------------------------------------

    lon_stream = lon
    lat_stream = lat[::-1]

    u_stream = u[::-1, :]
    v_stream = v[::-1, :]

    # Streamplot ei pidä NaN-arvoista kovin hyvin kaikissa
    # matplotlib-versioissa, joten korvataan ne nollalla.
    u_stream = np.nan_to_num(
        u_stream,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    v_stream = np.nan_to_num(
        v_stream,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    ax.streamplot(
        lon_stream,
        lat_stream,
        u_stream,
        v_stream,
        density=stream_density,
        color="black",
        linewidth=0.6,
        arrowsize=0.8,
        minlength=0.2,
        maxlength=4.0,
        integration_direction="both",
    )

    # --------------------------------------------------------
    # Jos lämpötila annettiin, voidaan tehdä toinen kuva
    # --------------------------------------------------------

    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    ax.set_title(title)

    ax.grid(
        True,
        color="white",
        alpha=0.20,
        linewidth=0.5
    )

    plt.show()



def naiive_downscale_temperature(
    simu_temperature,
    simu_dem,
    accurate_dem,
    lapse_rate=-6.5 / 1000.0,
):
    """
    Downscaleaa simulaation lämpötilan accurate_dem:n resoluutioon
    käyttäen naiivia lapse-rate-korjausta.

    Korkeuserosta huomioidaan vain tapaukset, joissa accurate_dem
    on korkeammalla kuin simu_dem.

    Parametrit
    ----------
    simu_temperature : ndarray
        Simulaation lämpötila, muoto (simu_ny, simu_nx).

    simu_dem : ndarray
        Simulaation DEM metreinä, sama resoluutio kuin simu_temperature.

    accurate_dem : ndarray
        Tarkka DEM metreinä.

    lapse_rate : float
        Lämpötilan muutos korkeuden mukana [°C/m].
        Oletus: -6.5 °C/km.

    Palauttaa
    ----------
    accurate_temperature : ndarray
        Downscalattu lämpötila accurate_dem:n resoluutiossa.
    """

    # ------------------------------------------------------------
    # 1. Tarkistetaan, että simulaation DEM ja lämpötila täsmäävät
    # ------------------------------------------------------------

    if simu_temperature.shape != simu_dem.shape:
        raise ValueError(
            "simu_temperature ja simu_dem pitää olla samassa resoluutiossa."
        )

    # ------------------------------------------------------------
    # 2. Skaalauskerroin simulaatioresoluutiosta accurate-resoluutioon
    # ------------------------------------------------------------

    zoom_y = accurate_dem.shape[0] / simu_temperature.shape[0]
    zoom_x = accurate_dem.shape[1] / simu_temperature.shape[1]

    zoom_factor = (zoom_y, zoom_x)

    # ------------------------------------------------------------
    # 3. Interpoloidaan simulaation lämpötila accurate-resoluutioon
    # ------------------------------------------------------------

    temperature_base = zoom(
        simu_temperature,
        zoom_factor,
        order=1,
    )

    # ------------------------------------------------------------
    # 4. Interpoloidaan simu-DEM samaan resoluutioon
    # ------------------------------------------------------------

    simu_dem_accurate = zoom(
        simu_dem,
        zoom_factor,
        order=1,
    )

    # ------------------------------------------------------------
    # 5. Korkeusero
    # ------------------------------------------------------------
    #
    # Vain jos accurate DEM on korkeammalla kuin simu DEM,
    # korkeusero vaikuttaa lämpötilaan.
    #

    height_difference = np.maximum(
        accurate_dem - simu_dem_accurate,
        0.0,
    )

    # ------------------------------------------------------------
    # 6. Lapse-rate-korjaus
    # ------------------------------------------------------------

    temperature_correction = (
        lapse_rate * height_difference
    )

    # ------------------------------------------------------------
    # 7. Lopullinen tarkka lämpötila
    # ------------------------------------------------------------

    accurate_temperature = (
        temperature_base
        + temperature_correction
    )

    return accurate_temperature

def create_dem_and_landmask(seed1,
    height,
    width,
    simu_ny,
    simu_nx,
    height_delta,
    sea_level,
    octaves=6,
    persistence=0.5,
    lacunarity=2.0,
    radius=1.0,
):
    """
    Luo saumattoman globaalin DEM:n metreinä käyttäen 3D Perlin-kohinaa.

    DEM:n korkeudet määritellään seuraavasti:

        min_height = -sea_level
        max_height = height_delta - sea_level

    Esimerkiksi:

        height_delta = 8000
        sea_level    = 4000

    antaa:

        abyss = -4000 m
        sea   =     0 m
        peak  = +4000 m

    Parametrit
    ----------
    height : int
        Tarkemman DEM:n rivimäärä (latitude).
    width : int
        Tarkemman DEM:n sarakemäärä (longitude).

    simu_ny : int
        Simulaatioresoluution rivimäärä.
    simu_nx : int
        Simulaatioresoluution sarakemäärä.

    height_delta : float
        Koko korkeusväli abyssista korkeimpaan huippuun metreinä.

    sea_level : float
        Merenpinnan korkeus mitattuna abyssista metreinä.

        Pitää täyttää:
            0 <= sea_level <= height_delta

    octaves : int
        Perlin-kohinan oktaavien määrä.

    persistence : float
        Oktaavien amplitudien vaimeneminen.

    lacunarity : float
        Oktaavien taajuuden kasvukerroin.

    radius : float
        Pallon säde Perlin-kohinan koordinaatistossa.

    Palauttaa
    ----------
    accurate_dem : ndarray
        DEM metreinä, shape (height, width).

    accurate_landmask : ndarray
        Maa = 1, meri = 0.

    simu_dem : ndarray
        Simulaatio-DEM metreinä.

    simu_landmask : ndarray
        Simulaation maamaski.

    accurate_stats : dict
        Tarkemman DEM:n tilastot.

    simu_stats : dict
        Simulaatio-DEM:n tilastot.
    """

    # ------------------------------------------------------------
    # Parametrien tarkistus
    # ------------------------------------------------------------

    if height_delta <= 0:
        raise ValueError("height_delta pitää olla > 0.")

    if not 0 <= sea_level <= height_delta:
        raise ValueError(
            "sea_level pitää olla välillä 0 ... height_delta."
        )

    # ------------------------------------------------------------
    # DEM:n generointi
    # ------------------------------------------------------------

    def generate_spherical_resolution(ny, nx):

        # --------------------------------------------------------
        # 1. Leveys- ja pituusasteet
        # --------------------------------------------------------

        lats = np.linspace(
            -np.pi / 2,
            np.pi / 2,
            ny,
            endpoint=True,
            dtype=np.float64,
        )

        lons = np.linspace(
            -np.pi,
            np.pi,
            nx,
            endpoint=False,
            dtype=np.float64,
        )

        lat_grid, lon_grid = np.meshgrid(
            lats,
            lons,
            indexing="ij",
        )

        # --------------------------------------------------------
        # 2. Pallokoordinaatit
        # --------------------------------------------------------

        X = radius * np.cos(lat_grid) * np.cos(lon_grid)
        Y = radius * np.cos(lat_grid) * np.sin(lon_grid)
        Z = radius * np.sin(lat_grid)

        # --------------------------------------------------------
        # 3. Raaka Perlin-DEM
        # --------------------------------------------------------

        noise_dem = np.empty(
            (ny, nx),
            dtype=np.float32,
        )

        for r in range(ny):
            for c in range(nx):
                noise_dem[r, c] = pnoise3(
                    float(X[r, c]),
                    float(Y[r, c]),
                    float(Z[r, c]), base=seed1,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                )

        # --------------------------------------------------------
        # 4. Normalisointi 0...1
        # --------------------------------------------------------
        #
        # Raaka Perlin-kohina ei välttämättä käytä koko [-1, +1]
        # väliä. Siksi käytetään todellista min/max-arvoa.
        #

        noise_min = np.min(noise_dem)
        noise_max = np.max(noise_dem)

        if noise_max == noise_min:
            raise ValueError(
                "Perlin-kohinan minimi ja maksimi ovat samat."
            )

        normalized_dem = (
            noise_dem - noise_min
        ) / (
            noise_max - noise_min
        )

        # Nyt:
        #
        # min(normalized_dem) = 0
        # max(normalized_dem) = 1

        # --------------------------------------------------------
        # 5. Skaalataan fyysiseksi korkeudeksi metreinä
        # --------------------------------------------------------
        #
        # 0 ... 1
        #   ↓
        # 0 ... height_delta
        #

        dem = normalized_dem * height_delta

        # --------------------------------------------------------
        # 6. Siirretään merenpinta korkeudelle 0
        # --------------------------------------------------------
        #
        # Esimerkiksi:
        #
        # height_delta = 8000
        # sea_level    = 4000
        #
        # 0       -> -4000 m
        # 4000    ->     0 m
        # 8000    -> +4000 m
        #

        dem -= sea_level

        # --------------------------------------------------------
        # 7. Maamaski
        # --------------------------------------------------------
        #
        # Meri:
        #     dem < 0
        #
        # Maa:
        #     dem >= 0
        #

        landmask = (dem >= 0).astype(np.uint8)

        # --------------------------------------------------------
        # 8. Cos(latitude)-painotus
        # --------------------------------------------------------

        weights = np.cos(lat_grid)

        total_weight = np.sum(weights)

        land_indices = landmask == 1
        sea_indices = landmask == 0

        land_weight_sum = np.sum(
            weights[land_indices]
        )

        sea_weight_sum = np.sum(
            weights[sea_indices]
        )

        # --------------------------------------------------------
        # 9. Maa-/meriosuudet
        # --------------------------------------------------------

        if total_weight > 0:
            land_share = (
                land_weight_sum / total_weight
            )

            sea_share = (
                sea_weight_sum / total_weight
            )
        else:
            land_share = 0.0
            sea_share = 0.0

        # --------------------------------------------------------
        # 10. Keskimääräinen maan korkeus
        # --------------------------------------------------------

        if land_weight_sum > 0:
            mean_land_height = (
                np.sum(
                    dem[land_indices]
                    * weights[land_indices]
                )
                / land_weight_sum
            )
        else:
            mean_land_height = 0.0

        # --------------------------------------------------------
        # 11. Keskimääräinen meren syvyys
        # --------------------------------------------------------
        #
        # DEM on meren alla negatiivinen.
        #
        # Esim:
        #     -1200 m -> syvyys 1200 m
        #

        if sea_weight_sum > 0:

            mean_sea_elevation = (
                np.sum(
                    dem[sea_indices]
                    * weights[sea_indices]
                )
                / sea_weight_sum
            )

            mean_sea_depth = -mean_sea_elevation

        else:
            mean_sea_elevation = 0.0
            mean_sea_depth = 0.0

        # --------------------------------------------------------
        # 12. Tilastot
        # --------------------------------------------------------

        stats = {
            "land_share": float(land_share),
            "sea_share": float(sea_share),

            "mean_land_height": float(
                mean_land_height
            ),

            "mean_sea_elevation": float(
                mean_sea_elevation
            ),

            "mean_sea_depth": float(
                mean_sea_depth
            ),

            "min_height": float(np.min(dem)),
            "max_height": float(np.max(dem)),
            "sea_level": 0.0,
            "height_delta": float(height_delta),
        }

        return dem, landmask, stats

    # ------------------------------------------------------------
    # Tarkka resoluutio
    # ------------------------------------------------------------

    accurate_dem, accurate_landmask, accurate_stats = (
        generate_spherical_resolution(
            height,
            width,
        )
    )

    # ------------------------------------------------------------
    # Simulaatioresoluutio
    # ------------------------------------------------------------

    simu_dem, simu_landmask, simu_stats = (
        generate_spherical_resolution(
            simu_ny,
            simu_nx,
        )
    )

    return (
        accurate_dem,
        accurate_landmask,
        simu_dem,
        simu_landmask,
        accurate_stats,
        simu_stats,
    )




        
# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":

    # 64 x 32 planet
    nx = simu_nx
    ny = simu_ny
    T = np.full((32, 64), initial_temperature-273.15)
    #co2_ppm=1000000
    #surface_pressure_atm=100
    #emissivity=laske_emissiivisyys(ppm=co2_ppm, pressure_atm=surface_pressure_atm)
    #print("emissivity " ,emissivity)
    #quit(0)
    #T2 = diffuse_temperature(
    #T,
    #K=1e6,
    #dt=3600.0
    #)
    #print(np.max(np.abs(T2 - T)))
    #lat=np.linspace(-90,90,181)
    #solar_declination=-25
    
    #clouds=cloud_fraction_np(lat, solar_declination)
    #plt.plot(lat,clouds)
    #plt.show()
    #quit(-1)  
    #dem, landmask, simudem, simulandmask, accurate_stats, simu_stats=create_dem_and_landmask(height, width, simu_ny, simu_nx, octaves=6, persistence=0.5, lacunarity=2.0, radius=1.5, sea_level=0.0)
    accurate_dem, accurate_landmask, simu_dem, simu_landmask, \
    accurate_stats, simu_stats = create_dem_and_landmask(seed1=seed1,
    height=height,
    width=width,
    simu_ny=simu_ny,
    simu_nx=simu_nx,
    height_delta=height_delta,
    sea_level=sea_level,
    )
    
    
    
    print(accurate_stats)
    print(simu_stats)
    #plt.imshow(simu_dem)
    #plt.imshow(accurate_dem*accurate_landmask)
    #plt.show()
    #quit(-1)    
    planet=simu_landmask
   
  
    # --------------------------------------------------------
    # material map
    #
    # 0 = ocean
    # 1 = land
    # --------------------------------------------------------

    #planet = np.zeros(
    #    (ny, nx),
    #    dtype=np.int8
    #)

    # simple continent
    #planet[
    #    ny // 4 : 3 * ny // 4,
    #    nx // 4 : 3 * nx // 4
    #] = 1

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    temperature, humidity, material, history = run_model(simu_dem, simu_landmask, 
    material=planet,
    initial_temperature=initial_temperature, ### 288.15
    initial_surface_pressure=surface_pressure,
    planet_radius_m=planet_radius_m,
    planet_mass_kg=planet_mass_kg,
    solar_constant=solar_constant,
    eccentricity=eccentricity,
    perihelion=perihelion,
    tilt=tilt,
    orbital_period_years=orbital_period_years,
    rotation_period_days=rotation_period_days,
    co2_ppm=co2_ppm,
    years=simu_run_years,
    dt_hours=simu_dt_hours,
    clouds_present=clouds_present,
    cloud_strength=cloud_strength,
    use_ice=use_ice,
    ice_temperature=ice_temperature,
    diffusion=diffusion,
    save_every_hours=simu_save_every_hours
    )
   
    print()
    print("RESULT")
    print("------")
    print(
        "Mean T:",
        np.mean(temperature) - 273.15,
        "C"
    )

    print(
        "Min T:",
        np.min(temperature) - 273.15,
        "C"
    )

    print(
        "Max T:",
        np.max(temperature) - 273.15,
        "C"
    )

    print(
        "Range:",
        np.max(temperature)
        - np.min(temperature),
        "K"
    )



# ============================================================
# YEARLY MEAN TEMPERATURE RASTER
# ============================================================

# history:
#   time, latitude, longitude
#
# Lasketaan vuoden keskilämpötila jokaiselle grid-solulle.
simu_mean_temperature_K = np.mean(history, axis=0)

# Celsius
simu_mean_temperature_C = simu_mean_temperature_K - 273.15

#simu_landmask=np.copy(simu_dem)
#simu_landmask=np.where(simu_landmask>0,1,0)


#rotation_days=rotation_period_days
#dem=simu_dem
#landmask=simu_landmask
#temperature_K=simu_mean_temperature_K






downscaled_temperature_C = naiive_downscale_temperature(
    simu_mean_temperature_C,
    np.where(simu_dem>0,simu_dem,0),
    np.where(accurate_dem>0,accurate_dem, 0)
)

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(12, 5))


# Merimaski: DEM <= 0
sea_mask = accurate_dem <= 0
data_min = np.nanmin(downscaled_temperature_C)
data_max = np.nanmax(downscaled_temperature_C)

max_abs = max(abs(data_min), abs(data_max))

limit = np.ceil(max_abs / 10) * 10

vmin = -limit
vmax = limit
# Lämpötilakuva
im = plt.imshow(
    downscaled_temperature_C,
    interpolation="bilinear",
    origin="lower", vmin=-limit, vmax=limit,
    extent=[-180, 180, -90, 90],
    aspect="auto",
    cmap="RdYlBu_r"
)

# Piirretään meri mustaksi
plt.imshow(
    np.ma.masked_where(~sea_mask, sea_mask),
    origin="lower",
    extent=[-180, 180, -90, 90],
    aspect="auto",
    cmap="gray",
    vmin=0,
    vmax=1,
    alpha=1.0
)
plt.contourf(
    accurate_dem,
    levels=[accurate_dem.min(), 0],
    colors=["#001f5f"],
    alpha=1.0,
    origin="lower",
    extent=[-180, 180, -90, 90]
)
# Rantaviiva
plt.contour(
    accurate_dem,
    levels=[0],
    linewidths=1.5,
    colors="white",
    alpha=1.0,
    origin="lower",
    extent=[-180, 180, -90, 90]
)

# Lämpötilakäyrät
cont1 = plt.contour(
    downscaled_temperature_C,
    levels=[-150, -120, -100, -80, -50,-40, -30, -25, -20,-15, -10,-5,
            0,5, 10,15, 20,25, 30, 40, 50, 80, 100, 120, 150, 250],
    linewidths=1,
    colors="black",
    alpha=0.5,
    origin="lower",
    extent=[-180, 180, -90, 90]
)


plt.clabel(cont1, inline=1, fmt='%3.1f',fontsize=10, colors=["#3f0000"])



plt.colorbar(
    im,
    label="Mean temperature (°C)"
)

plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")

plt.title(
    "Mean temperature of planet, °C"
)

plt.tight_layout()
plt.show()

