
##################################################################
#
## Simple terrestrial planet temperature energy balance model 2D
#
## Python3, Numba JIT
#
# 07.10.2026 00.04.00
#
#################################################################



import numpy as np
import math

import matplotlib.pyplot as plt
from noise import pnoise3

from numba import njit

import noise

# ============================================================
# SIMPLE PLANETARY ENERGY BALANCE MODEL
# Numba version
#
# dt_hours = 1.0 by default
#
# T       : K
# S       : W/m2
# C       : J/(m2 K)
# OLR     : W/m2
# ============================================================


# ============================================================
#  MAIN CONTROL PARAMETERS
# ============================================================

## simulation dimensions

simu_nx = 64
simu_ny = 32
        # simulation
simu_run_years=50
        # save daily

# <-- THIS IS THE MAIN TIMESTEP
simu_dt_hours=1.0

simu_save_every_hours=24

## accurate map dimensions
height=180
width=360
height_delta=10000
sea_level=6000

initial_temperature=288.15

# Earth

solar_constant=1365.2

orbital_period_years=1.0
rotation_period_days=1.0*365.25
eccentricity=0.0167
perihelion=283.0
tilt=23.44

# KOI 4878.01

#solar_constant=1365.2*0.98 #0.92-1.04

#orbital_period_years=449.015/365.25
#rotation_period_days=1.0
#eccentricity=0.0167
#perihelion=283.0
#tilt=23.44

# greenhouse parameter
#emissivity=0.58
#emissivity=0.61

emissivity=0.62
# clouds
#use_clouds=True
use_clouds=False

cloud_strength=1.0

# ice
#use_ice=True
use_ice=False

ice_temperature=263.0

# horizontal heat transport
# start with 0 while testing energy balance
diffusion=0.0




# ============================================================
#  CONSTANTS
# ============================================================

SIGMA = 5.670374419e-8


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

    else:                   # snow / land ice
        rho = 400.0
        cp = 880.0
        depth = 3.0
        albedo = 0.75

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

@njit
def cloud_fraction(lat, solar_declination):

    # six broad cloud belts
    x = (lat - solar_declination * 0.5) * math.pi / 180.0

    c = 0.5 + 0.5 * math.sin(6.0 * x)

    if c < 0.0:
        c = 0.0
    if c > 1.0:
        c = 1.0

    return c


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
    temperature,
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

    emissivity,

    use_clouds,
    cloud_strength,

    use_ice,
    ice_temperature
):

    ny, nx = temperature.shape

    new_temperature = np.empty_like(temperature)

    for y in range(ny):

        # latitude from -90 to +90
        if ny == 1:
            lat = 0.0
        else:
            lat = -90.0 + 180.0 * y / (ny - 1)

        # cloud field is zonal
        if use_clouds:
            cloud = cloud_fraction(
                lat,
                solar_declination(
                    tilt,
                    orbital_longitude
                )
            )

            ca = cloud_albedo(cloud)

        else:
            cloud = 0.0
            ca = 0.0

        for x in range(nx):

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

            if use_clouds:

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

            OLR = emissivity * SIGMA * T**4

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
def diffuse_temperature(T, diffusion):

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


# ============================================================
# SIMULATION
# ============================================================

def run_model(
    material,
    initial_temperature=288.15,

    solar_constant=1365.2,
    eccentricity=0.0167,
    perihelion=283.0,
    tilt=23.44,
    orbital_period_years=1.0,
    rotation_period_days=1.0,
    emissivity=0.58,

    years=1,
    dt_hours=1.0,

    cloud_strength=1.0,
    use_clouds=True,
    use_ice=True,

    ice_temperature=263.0,

    diffusion=0.0,

    save_every_hours=24
):

    # --------------------------------------------------------
    # arrays
    # --------------------------------------------------------

    material = np.asarray(
        material,
        dtype=np.int8
    ).copy()

    ny, nx = material.shape

    temperature = np.full(
        (ny, nx),
        initial_temperature,
        dtype=np.float64
    )

    capacity, base_albedo = initialize_properties(
        material
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

    for step in range(total_hours):

        time_hours = step * dt_hours

        #rotation_angle = (
        #360.0
        #* time_hours
        #/ (rotation_period_days * 24.0)
        #) % 360.0

        # planetary rotation
        #hour = time_hours % 24.0
        #hour = time_hours % (24.0*rotation_period_days)
        # orbital longitude
        #orbital_longitude = (
        #    360.0
        #    * (time_hours / orbital_hours)
        #) % 360.0
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

        temperature = energy_step(
            temperature,
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

            emissivity,

            use_clouds,
            cloud_strength,

            use_ice,
            ice_temperature
        )

        # ----------------------------------------------------
        # optional horizontal heat transport
        # ----------------------------------------------------

        if diffusion > 0.0:
            temperature = diffuse_temperature(
                temperature,
                diffusion
            )

        # ----------------------------------------------------
        # save
        # ----------------------------------------------------

        if step % int(save_every_hours / dt_hours) == 0:
            saved_temperature.append(
                temperature.copy()
            )

    return (
        temperature,
        material,
        np.array(saved_temperature)
    )



import numpy as np
from noise import pnoise3
import numpy as np
from scipy.ndimage import zoom


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

def create_dem_and_landmask(
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
                    float(Z[r, c]),
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

    #dem, landmask, simudem, simulandmask, accurate_stats, simu_stats=create_dem_and_landmask(height, width, simu_ny, simu_nx, octaves=6, persistence=0.5, lacunarity=2.0, radius=1.5, sea_level=0.0)
    accurate_dem, accurate_landmask, simu_dem, simu_landmask, \
    accurate_stats, simu_stats = create_dem_and_landmask(
    height=height,
    width=width,
    simu_ny=simu_ny,
    simu_nx=simu_nx,
    height_delta=height_delta,
    sea_level=sea_level,
    )
    print(accurate_stats)
    print(simu_stats)
    #plt.imshow(simulandmask)
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

    temperature, material, history = run_model(

        planet,

        initial_temperature=initial_temperature,

        # Earth
        solar_constant=solar_constant,
        eccentricity=eccentricity,
        perihelion=perihelion,
        tilt=tilt,
        orbital_period_years=orbital_period_years,
        rotation_period_days=rotation_period_days,
        # greenhouse parameter
        emissivity=emissivity,

        # simulation
        years=simu_run_years,

        # <-- THIS IS THE MAIN TIMESTEP
        dt_hours=simu_dt_hours,

        # clouds
        use_clouds=use_clouds,
        cloud_strength=cloud_strength,

        # ice
        use_ice=use_ice,
        ice_temperature=ice_temperature,

        # horizontal heat transport
        # start with 0 while testing energy balance
        diffusion=diffusion,

        # save daily
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
downscaled_temperature_C = naiive_downscale_temperature(
    simu_mean_temperature_C,
    np.where(simu_dem>0,simu_dem,0),
    np.where(accurate_dem>0,accurate_dem, 0)
)

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(12, 5))

#im = plt.imshow(
#    downscaled_temperature_C,
#    origin="lower",
#    extent=[-180, 180, -90, 90],
#    aspect="auto",
#    cmap="RdYlBu_r"
#)
im = plt.imshow(
    simu_mean_temperature_C,
    origin="lower",
    extent=[-180, 180, -90, 90],
    aspect="auto",
    cmap="RdYlBu_r"
)

plt.contour(accurate_dem, levels=[0,1], lw=2, colors=["black"], alpha=1.0,origin="lower",
    extent=[-180, 180, -90, 90])

#plt.contour(downscaled_temperature_C)


plt.colorbar(
    im,
    label="Mean tamperature (°C)"
)

plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")

plt.title(
    "Mean temperature of planet, °C"
)

plt.tight_layout()
plt.show()

