
##################################################################
#
## Simple terrestrial planet temperature energy balance model 2D
#
## Python3, Numba JIT
#
# 17.10.2026 00.15.00
#
#################################################################


import math

import numpy as np
from scipy.ndimage import zoom

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


from numba import njit

import cartopy.crs as ccrs

import noise
from noise import pnoise3
from scipy.ndimage import distance_transform_edt

import rebound


#from scipy import constants
#from astropy.constants import M_earth, R_earth

# ============================================================
#  MAIN CONTROL PARAMETERS
# ============================================================



# ============================================================
# YLEINEN TÄHTIJÄRJESTELMÄN SIMULAATIO
#
# Yksikot:
#   Aika       = vuotta
#   Etaisyys   = AU
#   Massa      = Auringon massat
#
# Planeetan koordinaatisto:
#   Pituusaste = [-180, 180)
#   Leveysaste = [-90, 90]
#
# Tulokset:
#   orbital_longitude
#   rotation_angle
#   star_directions
# ============================================================

DAYS_PER_YEAR = 365.256


# ============================================================
# 1. TAHTIJARJESTELMA
#
# parent=None: jarjestelman ensimmainen kappale
# parent      : kappaleen kiertama kappale
#
# Rataelementit:
#   a      = isoakselin puolikas AU
#   e      = eksentrisyys
#   inc    = inklinaatio asteina
#   Omega  = nousevan solmun pituus asteina
#   omega  = periapsiksen argumentti asteina
#   M      = keskianomalia asteina
#
# Pyoriminen:
#   rotation_period_days:
#       positiivinen = positiivinen pyorimissuunta
#       negatiivinen = vastakkainen pyorimissuunta
#       None         = ei pyorimismallia
#
# Axial tilt:
#   Kallistus globaalista +Z-akselista asteina.
#
# Axial azimuth:
#   Akselin atsimuutti globaalissa XY-tasossa asteina.
# ============================================================

bodies = [
    {
        "name": "Freyr",
        "kind": "star",
        "parent": None,
        "mass": 14.8,
        "luminosity": 60000.0,
    },
    {
        "name": "Batalix",
        "kind": "star",
        "parent": "Freyr",
        "mass": 0.96,
        "luminosity": 0.8,
        "orbit": {
            "a": 473.0,
            "e": 0.50,
            "inc": 0.0,
            "Omega": 0.0,
            "omega": 0.0,
            "M": 0.0,
        },
    },
    {
        "name": "Helliconia",
        "kind": "planet",
        "parent": "Batalix",
        "mass": 1.28 * 3.0035e-6,
        "luminosity": 0.0,
        "orbit": {
            "a": 1.26,
            "e": 0.01,
            "inc": 0.0,
            "Omega": 0.0,
            "omega": 0.0,
            "M": 0.0,
        },
        "rotation_period_days": 25.0 / 24.0,
        "initial_rotation_angle": 0.0,
        "rotation_epoch": 0.0,
        "axial_tilt": 35.0,
        "axial_azimuth": 0.0,
    },
]




## simulation dimensions

simu_nx = 32
simu_ny = 16
        # simulation
simu_run_years=3
        # save daily

# <-- THIS IS THE MAIN TIMESTEP
simu_dt_hours=1

simu_save_every_hours=24

## accurate map dimensions
height=360*2
width=720*2
height_delta=10000
sea_level=7250
#sea_level=3000
#sea_level=4000

seed1=0 ## dem map seed
#seed1=42

duration_days = 30.0
step_days = 1.0
albedo = 0.30
# Vaihtoehtoisesti astrofysiikan astropy-kirjastosta:
# 


## earth-like
planet_mass_me=1
planet_radius_re=1   
orbital_period_years=1.0
solar_constant_suns=1
rotation_period_days=1
eccentricity=0.0167
perihelion=283.0
tilt=23.44
surface_pressure_atm=1
co2_ppm=280
fake_deserts=1

## tilt 0
#planet_mass_me=1
#planet_radius_re=1 
#rotation_period_days=365.25*1.0
#orbital_period_years=1.0
#solar_constant_suns=1
#rotation_period_days=1
#eccentricity=0.00
#perihelion=0
#tilt=0.0
#surface_pressure_atm=1
#co2_ppm=280
#fake_deserts=0

##locked 
#planet_mass_me=1
#planet_radius_re=1 
#orbital_period_years=1.0
#rotation_period_days=365.25*1.0
#solar_constant_suns=1
#eccentricity=0.00
#perihelion=0
#tilt=0.0
#surface_pressure_atm=1
#co2_ppm=280


# KOI 4878.01
#planet_mass_me=1
#planet_radius_re=1 
#solar_constant_suns=1*0.98 #0.92-1.04
#solar_constant_suns=1*0.92 #0.92-1.04
#orbital_period_years=449.015/365.25
#rotation_period_days=1.0
#eccentricity=0.0167
#perihelion=283.0
#tilt=23.44
#surface_pressure_atm=1
#co2_ppm=280
#fake_deserts=0

# clouds
#use_clouds=True
clouds_present=0 # 0 none 1 zonal 2 locked hurricane

cloud_strength=3

fake_rain_system=1 ## use Earth-like preset rain system
diffusion_scaled=0.0
# ice
#use_ice=True
use_simple_ice=True # default simple ice yes/np
use_dynamic_ice=False ## attempt to develop dynamic ice

use_atmosphere=False ## NOTE Only experimental atmosphere code

budyko_heat_transport_coeff=0

##scalet or unscaled diffusion, D or K

diffusion_scaled=0.0 ## use scaled D diffuin, scaled
diffusion=0.0 ## use unscaled K diffusion

# horizontal heat transport
# start with 0 while testing energy balance

#diffusion=2e6*pow(surface_pressure_atm, -0.5) ## slowrot 1e7 ... 1e8
diffusion=2e5*pow(surface_pressure_atm, -0.5) ##  fastrot

#diffusion=6e5


diffusion_scaled=0
#diffusion=1e-6

#diffusion=1 ## new scaled system

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
initial_temperature=288.15*math.sqrt(solar_constant_suns)
planet_radius_m=R_e*planet_radius_re
planet_mass_kg=M_e*planet_mass_me
surface_pressure=101325*surface_pressure_atm
solar_constant=solar_constant_suns*1365.2




# ============================================================
# 2. SIMULAATION RAKENTAMINEN
# ============================================================

sim = rebound.Simulation()
sim.units = ("yr", "AU", "Msun")
sim.integrator = "ias15"

indices = {}
body_data = {}

for body in bodies:
    name = body["name"]
    parent_name = body.get("parent")

    if name in indices:
        raise ValueError(f"Duplikaattinimi: {name}")

    if parent_name is None:
        sim.add(m=body["mass"])
    else:
        if parent_name not in indices:
            raise ValueError(
                f"Vanhempi {parent_name} on lisattava "
                f"ennen kappaletta {name}."
            )

        orbit = body["orbit"]

        sim.add(
            primary=sim.particles[indices[parent_name]],
            m=body["mass"],
            a=orbit["a"],
            e=orbit.get("e", 0.0),
            inc=np.deg2rad(orbit.get("inc", 0.0)),
            Omega=np.deg2rad(orbit.get("Omega", 0.0)),
            omega=np.deg2rad(orbit.get("omega", 0.0)),
            M=np.deg2rad(orbit.get("M", 0.0)),
        )

    indices[name] = len(indices)
    body_data[name] = body

sim.move_to_com()


# ============================================================
# 3. PLANEETAN TILAMUUTTUJAT
# ============================================================

def planet_state(planet_name, time_years):
    """
    Palauttaa planeetan ratakulman ja pyorahdyskulman.

    orbital_longitude:
        Planeetan paikkavektorin XY-projektion kulma
        suhteessa sen vanhempaan kappaleeseen.
        Valilla [0, 360).

    rotation_angle:
        Pyorimisvaihe asteina valilla [0, 360).

    Huomaa:
        orbital_longitude ei ole yleisesti sama kuin
        ratatasossa mitattu todellinen pituusaste.
    """

    body = body_data[planet_name]

    if body["kind"] != "planet":
        raise ValueError(f"{planet_name} ei ole planeetta.")

    parent_name = body.get("parent")

    if parent_name is None:
        raise ValueError("Planeetalla pitaa olla parent.")

    planet = sim.particles[indices[planet_name]]
    parent = sim.particles[indices[parent_name]]

    dx = planet.x - parent.x
    dy = planet.y - parent.y
    dz = planet.z - parent.z

    orbital_longitude = float(
        np.degrees(np.arctan2(dy, dx)) % 360.0
    )

    period = body.get("rotation_period_days")

    if period is None:
        rotation_angle = None
    else:
        elapsed_days = (
            time_years - body.get("rotation_epoch", 0.0)
        ) * DAYS_PER_YEAR

        rotation_angle = float(
            (
                body.get("initial_rotation_angle", 0.0)
                + 360.0 * elapsed_days / period
            ) % 360.0
        )

    return {
        "orbital_longitude": orbital_longitude,
        "rotation_angle": rotation_angle,
        "axial_tilt": body.get("axial_tilt", 0.0),
        "axial_azimuth": body.get("axial_azimuth", 0.0),
        "relative_position": (dx, dy, dz),
        "distance_to_parent": float(
            np.sqrt(dx**2 + dy**2 + dz**2)
        ),
    }


# ============================================================
# 4. TAHTIEN SUUNNAT PLANEETAN KOORDINAATISTOSSA
# ============================================================

def star_directions(planet_name, time_years):
    """
    Laskee kaikkien tahtien suunnat planeetan
    kallistetussa ja pyorivassa koordinaatistossa.

    Palauttaa jokaisesta tahdesta:
        star
        longitude   [-180, 180)
        latitude    [-90, 90]
        distance_au

    Pituusasteen nollameridiaani maaraytyy planeetan
    alkukoordinaatiston ja initial_rotation_angle-arvon
    perusteella.
    """

    body = body_data[planet_name]

    if body["kind"] != "planet":
        raise ValueError(f"{planet_name} ei ole planeetta.")

    planet = sim.particles[indices[planet_name]]

    tilt = np.deg2rad(body.get("axial_tilt", 0.0))
    azimuth = np.deg2rad(body.get("axial_azimuth", 0.0))

    # Planeetan akselin maarittama ortonormaali kanta.
    ex = np.array([
        np.cos(tilt) * np.cos(azimuth),
        np.cos(tilt) * np.sin(azimuth),
        -np.sin(tilt),
    ])

    ey = np.array([
        -np.sin(azimuth),
        np.cos(azimuth),
        0.0,
    ])

    ez = np.array([
        np.sin(tilt) * np.cos(azimuth),
        np.sin(tilt) * np.sin(azimuth),
        np.cos(tilt),
    ])

    period = body.get("rotation_period_days")

    if period is None:
        phase = 0.0
    else:
        elapsed_days = (
            time_years - body.get("rotation_epoch", 0.0)
        ) * DAYS_PER_YEAR

        phase = np.deg2rad(
            body.get("initial_rotation_angle", 0.0)
            + 360.0 * elapsed_days / period
        )

    results = []

    for star in bodies:
        if star["kind"] != "star":
            continue

        star_particle = sim.particles[indices[star["name"]]]

        # Tahti planeetan suhteen.
        r = np.array([
            star_particle.x - planet.x,
            star_particle.y - planet.y,
            star_particle.z - planet.z,
        ])

        distance = float(np.linalg.norm(r))

        if distance == 0.0:
            continue

        # Muunnos planeetan kallistettuun akselistoon.
        x = np.dot(r, ex)
        y = np.dot(r, ey)
        z = np.dot(r, ez)

        # Muunnos pyorivaan koordinaatistoon.
        xr = x * np.cos(phase) + y * np.sin(phase)
        yr = -x * np.sin(phase) + y * np.cos(phase)

        longitude = float(
            (np.degrees(np.arctan2(yr, xr)) + 180.0)
            % 360.0 - 180.0
        )

        latitude = float(
            np.degrees(
                np.arctan2(z, np.hypot(xr, yr))
            )
        )

        results.append({
            "star": star["name"],
            "longitude": longitude,
            "latitude": latitude,
            "distance_au": distance,
        })

    return results


# ============================================================
# 5. TAHTIEN SATEILY JA TASAPAINOLAMPOTILA
# ============================================================

def radiation_and_temperature(planet_name, albedo=0.30):
    """
    Laskee planeetan vastaanottaman kokonais-sateilyn
    suhteessa Maan nykyiseen auringonsateilyyn.

    Yksinkertaistettu tasapainolampotila ei huomioi
    ilmakehaa, lampokapasiteettia tai kasvihuoneilmion
    vaikutuksia.
    """

    planet = sim.particles[indices[planet_name]]
    total_flux = 0.0
    contributions = []

    for star in bodies:
        if star["kind"] != "star":
            continue

        particle = sim.particles[indices[star["name"]]]

        dx = particle.x - planet.x
        dy = particle.y - planet.y
        dz = particle.z - planet.z

        distance = np.sqrt(dx**2 + dy**2 + dz**2)

        if distance <= 0.0:
            continue

        luminosity = star.get("luminosity", 0.0)
        flux = luminosity / distance**2

        total_flux += flux

        contributions.append({
            "star": star["name"],
            "flux": float(flux),
            "distance_au": float(distance),
        })

    if total_flux > 0.0:
        equilibrium_temperature = (
            278.5 * (total_flux * (1.0 - albedo))**0.25
        )
    else:
        equilibrium_temperature = 0.0

    return {
        "total_flux": float(total_flux),
        "equilibrium_temperature_K": float(
            equilibrium_temperature
        ),
        "contributions": contributions,
    }








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
    elif material == 3:    # snow / glacier
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




def generate_humidity_raster(temperature_k, landmask):
    """
    Laskee absoluuttisen kosteusrasterin (g/m3) lämpötilan (K) ja landmaskin (1=maa, 0=meri) perusteella.
    """
    # 1. Muunnos Celsiukseksi
    T_c = temperature_k - 273.15
    
    # 2. Kyllästyspaine (Magnus-kaava, tulos Pascalina)
    e_sat = 611.2 * np.exp((17.67 * T_c) / (T_c + 243.5))
    
    # 3. Maksimikosteus g/m3 (kaasuvakio vesihöyrylle R_v = 461.5 J/(kg*K))
    max_humidity = (e_sat / (461.5 * temperature_k)) * 1000
    
    # 4. Suhteellisen kosteuden kerroin (meri = 85%, maa = 50%)
    # Voit korvata tämän pehmeämmällä siirtymällä (esim. etäisyys rannikosta)
    relative_humidity = np.where(landmask == 1, 0.50, 0.85)
    
    # 5. Lopullinen absoluuttinen kosteus
    humidity_raster = max_humidity * relative_humidity
    
    return humidity_raster
    


import numpy as np
from numba import njit

@njit
def calculate_fake_rain(T, material, sun_decl, dt_seconds, itcz_coeff=1.0):
    ny, nx = T.shape
    rain_mm_h = np.zeros((ny, nx), dtype=np.float64)
    
    dlat = np.pi / ny
    
    # 1. Lasketaan ITCZ:n (pasaatituulien kohtaamisvyöhykkeen) tämänhetkinen leveysaste radiaaneina
    # sun_decl on radiaaneina välillä [-23.45°, 23.45°] eli noin [-0.4, 0.4]
    itcz_lat = itcz_coeff * sun_decl
    
    for y in range(ny):
        # Solun keskikohdan leveysaste radiaaneina (-pi/2 ... pi/2)
        phi = -0.5 * np.pi + (y + 0.5) * dlat
        
        # --- FEIKATTU ILMASTOVYÖHYKE-GEOMETRIA ---
        # Lasketaan etäisyys dynaamisesti liikkuvaan ITCZ-päiväntasaajaan
        dist_to_itcz = phi - itcz_lat
        
        # ITCZ: Voimakas nousuvirtaus ja sateet kapealla vyöhykkeellä ITCZ:n ympärillä
        itcz_force = np.exp(- (dist_to_itcz / 0.12)**2) # kapea Gaussin käyrä
        
        # Subtrooppinen korkeapaine (Aavikkovyöhykkeet noin +-30 astetta päiväntasaajalta)
        # Täällä ilma laskeutuu, mikä estää sateet (vaimennuskerroin)
        subtropics_force = np.sin(2.0 * (phi - itcz_lat))**2
        
        # Polaaririntama (Lauhkean vyöhykkeen sateet noin +-60 asteessa, Maassa n. 1.0 rad)
        polar_front_force = np.exp(- (np.abs(phi) - 1.0)**2 / 0.15)
        
        # Yhdistetään pystysuuntaisen ilmanliikkeen (nousuvirtauksen) feikki-indeksi
        # Positiivinen arvo = nouseva ilma (sateita), negatiivinen = laskeva kuiva ilma
        updraft_index = 1.2 * itcz_force + 0.6 * polar_front_force - 0.7 * subtropics_force
        
        for x in range(nx):
            t_celsius = T[y, x] - 273.15
            
            # --- BASE HUMIDITY (Clausius-Clapeyron -approksimaatio) ---
            # Lasketaan ilman suurin mahdollinen kosteuskapasiteetti q_sat (g/m³) lämpötilan mukaan
            # Kaava pohjautuu siihen, että kosteuskapasiteetti tuplautuu noin joka 10. aste
            if t_celsius > -20.0:
                q_sat = 4.85 * np.exp(0.065 * t_celsius)
            else:
                q_sat = 1.0 # Hyvin kylmässä ilmassa ei ole lähes lainkaan kosteutta
            
            # Oletetaan peruskosteus (RH=100% pohjalta). 
            # Meren päällä ilman suhteellinen kosteus on korkeampi kuin mantereella.
            m = material[y, x]
            if m == 0:
                rh = 0.85  # Meri on kostea
            else:
                rh = 0.55  # Manner on kuivempi
                
            base_humidity = q_sat * rh
            
            # --- SADEMÄÄRÄN LASKENTA ---
            # Sadetta syntyy vain jos on nousuvirtausta (updraft_index > 0) ja ilmakehässä on kosteutta
            if updraft_index > 0.0 and base_humidity > 2.0:
                # Sadekynnyksen ylitys ja sateen tehokkuus
                # Mitä lämpimämpää ja kosteampaa ilma on, sitä rajumpi kaatosade syntyy (kuten tropiikissa)
                rain_intensity = updraft_index * (base_humidity - 2.0) * 0.12
                
                # Manner vähentää hieman sateen määrää verrattuna mereen (vähemmän dynaamista haihtumista suoraan alta)
                if m != 0:
                    rain_intensity *= 0.8
                    
                # Varmistetaan nollataso
                if rain_intensity < 0.0:
                    rain_intensity = 0.0
                    
                rain_mm_h[y, x] = rain_intensity
            else:
                rain_mm_h[y, x] = 0.0
                
    return rain_mm_h

import numpy as np
from numba import njit


@njit
def locked_planet_weather_old(
    T, q, pressure, material, sun_decl, dt_seconds,
    itcz_coeff=1.0,
    h_eff=10.0,
    u=None,
    v=None,
    substellar_lat=0.0,
    substellar_lon=0.0,
    rotation_period=20.0 * 86400.0,
    T_day=310.0,
    T_night=180.0,
    tau_rad=20.0 * 86400.0,
    tau_pressure=3.0 * 86400.0,
    pressure_sensitivity=35.0,
    drag=1.0 / 86400.0,
    c_eff=25.0
):
    """
    Yksinkertaistettu vuorovesilukitun planeetan säämalli.

    Syötteet
    --------
    T           : pintalämpötila [K], (ny, nx)
    q           : vesihöyry [g/m^3], (ny, nx)
    pressure    : paine [Pa], (ny, nx)
    material    : materiaalikartta 0...3
    sun_decl    : säilytetään yhteensopivuuden vuoksi
    dt_seconds  : aika-askel sekunteina

    Parametrit
    ----------
    substellar_lat : tähden alla olevan pisteen leveysaste [rad]
    substellar_lon : tähden alla olevan pisteen pituusaste [rad]
    rotation_period: pyörähdysaika [s], lukitussa tilassa
                     sama kuin ratakierroksen pituus
    T_day          : päiväpuolen säteilytasapainon tavoite [K]
    T_night        : yöpuolen säteilytasapainon tavoite [K]
    tau_rad        : lämpötilan palautumisaika [s]
    tau_pressure   : paineen palautumisaika [s]
    pressure_sensitivity: paine-eron voimakkuus [Pa]
    drag           : tuulen vaimennus [1/s]
    c_eff          : painevasteen tehollinen nopeus [m/s]

    Palauttaa
    ---------
    T_new, q_new, p_new, rain_mm_h, rh_matrix,
    cloudiness, u_new, v_new
    """

    ny, nx = T.shape

    R = 6_371_000.0
    rho_air = 1.2
    g = 9.81

    dlat = np.pi / ny
    dlon = 2.0 * np.pi / nx
    dy = R * dlat

    dt = max(dt_seconds, 0.0)
    tau_rad = max(tau_rad, 1.0)
    tau_pressure = max(tau_pressure, 1.0)
    h_eff = max(h_eff, 0.1)
    rotation_period = max(rotation_period, 1.0)

    omega = 2.0 * np.pi / rotation_period

    if u is None:
        u = np.zeros((ny, nx), dtype=np.float64)
    if v is None:
        v = np.zeros((ny, nx), dtype=np.float64)

    T_new = T.copy()
    q_new = q.copy()
    p_new = pressure.copy()

    u_new = u.copy()
    v_new = v.copy()

    rain_mm_h = np.zeros((ny, nx), dtype=np.float64)
    rh_matrix = np.zeros((ny, nx), dtype=np.float64)
    cloudiness = np.zeros((ny, nx), dtype=np.float64)

    T_eq = np.empty((ny, nx), dtype=np.float64)
    illumination = np.zeros((ny, nx), dtype=np.float64)
    lift_field = np.zeros((ny, nx), dtype=np.float64)

    # Pintamateriaalien teholliset lämpökapasiteetit [J/m²/K]
    Cmat = np.array([
        1030.0 * 4184.0 * 10.0,  # vesi
        2700.0 * 750.0 * 2.0,    # kivi
        917.0 * 2100.0 * 3.0,    # jää
        400.0 * 880.0 * 3.0      # kuiva maa
    ])

    # --------------------------------------------------
    # 1. PYSYVÄ PÄIVÄ- JA YÖPUOLI
    # --------------------------------------------------

    p_mean = 0.0
    T_mean = 0.0

    for y in range(ny):
        lat = -0.5 * np.pi + (y + 0.5) * dlat

        for x in range(nx):
            lon = -np.pi + (x + 0.5) * dlon

            cos_zenith = (
                np.sin(lat) * np.sin(substellar_lat)
                + np.cos(lat) * np.cos(substellar_lat)
                * np.cos(lon - substellar_lon)
            )

            mu = max(0.0, cos_zenith)
            illumination[y, x] = mu

            # Päiväpuolen ja yöpuolen tavoitelämpötila.
            T_eq[y, x] = (
                T_night
                + (T_day - T_night) * np.sqrt(mu)
            )

            p_mean += pressure[y, x]
            T_mean += T[y, x]

    p_mean /= ny * nx
    T_mean /= ny * nx

    # --------------------------------------------------
    # 2. PAINE-EROT JA TUULET
    # --------------------------------------------------
    # Lämmin päiväpuoli -> matalamman paineen suunta.
    # Kylmä yöpuoli -> korkeamman paineen suunta.
    #
    # Paineen lämpötilapakote on parametrisoitu
    # yksinkertaistus, ei hydrostaattinen ratkaisu.

    for y in range(ny):
        lat = -0.5 * np.pi + (y + 0.5) * dlat
        cosphi = max(0.05, np.cos(lat))
        dx = R * dlon * cosphi

        yn = min(y + 1, ny - 1)
        ys = max(y - 1, 0)

        f = 2.0 * omega * np.sin(lat)

        for x in range(nx):
            xe = (x + 1) % nx
            xw = (x - 1 + nx) % nx

            dpdx = (
                pressure[y, xe] - pressure[y, xw]
            ) / (2.0 * dx)

            dpdy = (
                pressure[yn, x] - pressure[ys, x]
            ) / (2.0 * dy)

            # Painegradientti kiihdyttää tuulta.
            # Coriolis-voima riippuu hitaasta pyörimisestä.
            ua = -dpdx / rho_air + f * v[y, x]
            va = -dpdy / rho_air - f * u[y, x]

            un = (
                u[y, x] + dt * ua
            ) / (1.0 + drag * dt)

            vn = (
                v[y, x] + dt * va
            ) / (1.0 + drag * dt)

            speed = np.sqrt(un * un + vn * vn)

            # Numeerinen turvaraja, ei tuulen fysikaalinen laki.
            if speed > 100.0:
                scale = 100.0 / speed
                un *= scale
                vn *= scale

            u_new[y, x] = un
            v_new[y, x] = vn

    # --------------------------------------------------
    # 3. LÄMMÖN JA KOSTEUDEN ADVEKTIO
    # --------------------------------------------------

    for y in range(ny):
        lat = -0.5 * np.pi + (y + 0.5) * dlat
        dx = R * dlon * max(0.05, np.cos(lat))

        yn = min(y + 1, ny - 1)
        ys = max(y - 1, 0)

        for x in range(nx):
            xe = (x + 1) % nx
            xw = (x - 1 + nx) % nx

            ux = u_new[y, x]
            vy = v_new[y, x]

            if ux >= 0.0:
                dTdx = (T[y, x] - T[y, xw]) / dx
                dqdx = (q[y, x] - q[y, xw]) / dx
            else:
                dTdx = (T[y, xe] - T[y, x]) / dx
                dqdx = (q[y, xe] - q[y, x]) / dx

            if vy >= 0.0:
                dTdy = (T[y, x] - T[ys, x]) / dy
                dqdy = (q[y, x] - q[ys, x]) / dy
            else:
                dTdy = (T[yn, x] - T[y, x]) / dy
                dqdy = (q[yn, x] - q[y, x]) / dy

            T_adv = T[y, x] - dt * (
                ux * dTdx + vy * dTdy
            )

            q_adv = max(
                0.0,
                q[y, x] - dt * (
                    ux * dqdx + vy * dqdy
                )
            )

            # --------------------------------------------------
            # 4. SÄTEILY JA PINTAMATERIAALI
            # --------------------------------------------------

            m = material[y, x]
            if m < 0 or m > 3:
                m = 1

            # Säteilyllinen palautuminen tavoitelämpötilaan.
            # Materiaalin lämpökapasiteetti hidastaa muutosta.
            #
            # Referenssikapasiteetti vastaa kalliopintaa.
            thermal_factor = Cmat[1] / Cmat[m]

            alpha = min(
                1.0,
                dt / tau_rad * thermal_factor
            )

            T_rad = T_adv + alpha * (
                T_eq[y, x] - T_adv
            )

            T_new[y, x] = T_rad
            q_new[y, x] = q_adv

            # Päiväpuolen lämpö synnyttää nousuliikettä.
            mu = illumination[y, x]

            day_lift = max(
                0.0,
                (T_eq[y, x] - T_night)
                / max(T_day - T_night, 1.0)
            )

            # Lievä lisänousu päiväntasaajan alueella.
            equatorial = max(0.0, np.cos(lat))

            lift_field[y, x] = min(
                1.0,
                day_lift * (0.75 + 0.25 * equatorial)
            )

    # --------------------------------------------------
    # 5. PAINEEN PÄIVITYS
    # --------------------------------------------------
    # Paine reagoi sekä lämpötilarakenteeseen että
    # vaakasuuntaisen tuulen divergenssiin.

    for y in range(ny):
        lat = -0.5 * np.pi + (y + 0.5) * dlat
        dx = R * dlon * max(0.05, np.cos(lat))

        yn = min(y + 1, ny - 1)
        ys = max(y - 1, 0)

        for x in range(nx):
            xe = (x + 1) % nx
            xw = (x - 1 + nx) % nx

            dudx = (
                u_new[y, xe] - u_new[y, xw]
            ) / (2.0 * dx)

            dvdy = (
                v_new[yn, x] - v_new[ys, x]
            ) / (2.0 * dy)

            divergence = dudx + dvdy

            # Lämmin alue saa matalamman tavoitepaineen.
            p_target = p_mean - pressure_sensitivity * (
                T_eq[y, x] - T_mean
            ) / max(T_day - T_night, 1.0)

            dp_relax = (
                (p_target - pressure[y, x])
                / tau_pressure
            )

            dp_wave = (
                -rho_air * c_eff**2 * divergence
            )

            dp = dt * (dp_relax + dp_wave)

            # Yhden aika-askeleen muutosrajoitus.
            dp = min(100.0, max(-100.0, dp))

            p_new[y, x] = max(
                1000.0,
                pressure[y, x] + dp
            )

    # --------------------------------------------------
    # 6. PILVET, SADE JA KOSTEUDEN TIIVISTYMINEN
    # --------------------------------------------------

    for y in range(ny):
        for x in range(nx):
            m = material[y, x]
            if m < 0 or m > 3:
                m = 1

            tc = T_new[y, x] - 273.15

            # Kyllästyskosteuden karkea likiarvo [g/m^3].
            q_sat = max(
                1.0,
                4.85 * np.exp(
                    0.065 * max(tc, -20.0)
                )
            )

            rh = min(
                1.0,
                max(0.0, q_new[y, x] / q_sat)
            )

            rh_matrix[y, x] = rh

            # Pilvisyys kasvaa kosteuden mukana.
            z_cloud = min(
                1.0,
                max(0.0, (rh - 0.35) / 0.50)
            )

            cloud_base = (
                z_cloud * z_cloud * (3.0 - 2.0 * z_cloud)
            )

            lift = lift_field[y, x]

            cloud_val = cloud_base * (
                0.65 + 0.35 * lift
            )

            cloudiness[y, x] = min(
                1.0, max(0.0, cloud_val)
            )

            # Sade edellyttää korkeaa kosteutta ja nousuliikettä.
            z_rain = min(
                1.0,
                max(0.0, (rh - 0.65) / 0.25)
            )

            rain_gate = (
                z_rain * z_rain * (3.0 - 2.0 * z_rain)
            )

            rain_intensity = (
                5.0 * lift * rain_gate * cloud_val
            )

            if m != 0:
                rain_intensity *= 0.8

            # Käytettävissä olevan vesihöyryn yläraja.
            available_mm = (
                max(0.0, q_new[y, x]) * h_eff / 1000.0
            )

            max_rain_mm_h = (
                available_mm * 3600.0 / max(dt, 1.0)
            )

            rain_intensity = min(
                rain_intensity,
                max_rain_mm_h
            )

            rain_mm_h[y, x] = rain_intensity

            if rain_intensity > 0.0:
                rain_g_m3_s = (
                    rain_intensity * 1000.0
                    / (3600.0 * h_eff)
                )

                q_new[y, x] = max(
                    0.0,
                    q_new[y, x] - rain_g_m3_s * dt
                )

                # Sateen yksinkertaistettu haihtumis-/jäähdytysvaikutus.
                cooling_power = rain_intensity * 20.0

                T_new[y, x] -= (
                    cooling_power * dt / Cmat[m]
                )

    return (
        T_new, q_new, p_new,
        rain_mm_h, rh_matrix, cloudiness,
        u_new, v_new
    )



@njit
def locked_planet_weather(
    T, q, pressure, material, sun_decl, dt_seconds,
    itcz_coeff=1.0,
    h_eff=10.0,
    u=None,
    v=None,
    substellar_lat=0.0,
    substellar_lon=0.0,
    rotation_period=20.0 * 86400.0,
    T_day=310.0,
    T_night=180.0,
    tau_rad=20.0 * 86400.0,
    tau_pressure=3.0 * 86400.0,
    pressure_sensitivity=35.0,
    drag=1.0 / 86400.0,
    c_eff=25.0,
    central_hurricane=False,
    r_rainy=np.pi / 6.0,
    r_cloudy=np.pi / 3.0,
    hurricane_rain_max=5.0,
    hurricane_cloud_max=0.8,
    hurricane_wind_max=30.0
):
    """
    Yksinkertaistettu vuorovesilukitun planeetan säämalli.

    Keskushurrikaani:
      central_hurricane : kytkee myrskyn päälle/pois
      r_rainy           : sadealueen säde [rad], oletus 30 astetta
      r_cloudy          : pilvialueen säde [rad], oletus 60 astetta
      hurricane_rain_max: myrskyn enimmäissade [mm/h]
      hurricane_cloud_max: myrskyn enimmäispilvisyys [0...1]
      hurricane_wind_max: pyörivän tuulen enimmäisnopeus [m/s]

    Palauttaa:
      T_new, q_new, p_new, rain_mm_h, rh_matrix,
      cloudiness, u_new, v_new
    """

    ny, nx = T.shape

    R = 6_371_000.0
    rho_air = 1.2

    dlat = np.pi / ny
    dlon = 2.0 * np.pi / nx
    dy = R * dlat

    dt = max(dt_seconds, 0.0)
    tau_rad = max(tau_rad, 1.0)
    tau_pressure = max(tau_pressure, 1.0)
    h_eff = max(h_eff, 0.1)
    rotation_period = max(rotation_period, 1.0)

    r_rainy = max(r_rainy, 1e-6)
    r_cloudy = max(r_cloudy, r_rainy)

    omega = 2.0 * np.pi / rotation_period

    if u is None:
        u = np.zeros((ny, nx), dtype=np.float64)
    if v is None:
        v = np.zeros((ny, nx), dtype=np.float64)

    T_new = T.copy()
    q_new = q.copy()
    p_new = pressure.copy()

    u_new = u.copy()
    v_new = v.copy()

    rain_mm_h = np.zeros((ny, nx), dtype=np.float64)
    rh_matrix = np.zeros((ny, nx), dtype=np.float64)
    cloudiness = np.zeros((ny, nx), dtype=np.float64)

    T_eq = np.empty((ny, nx), dtype=np.float64)
    illumination = np.zeros((ny, nx), dtype=np.float64)
    lift_field = np.zeros((ny, nx), dtype=np.float64)

    Cmat = np.array([
        1030.0 * 4184.0 * 10.0,  # vesi
        2700.0 * 750.0 * 2.0,    # kivi
        917.0 * 2100.0 * 3.0,    # jää
        400.0 * 880.0 * 3.0      # kuiva maa
    ])

    # --------------------------------------------------
    # 1. PYSYVÄ PÄIVÄ- JA YÖPUOLI
    # --------------------------------------------------

    p_mean = 0.0
    T_mean = 0.0

    for y in range(ny):
        lat = -0.5 * np.pi + (y + 0.5) * dlat

        for x in range(nx):
            lon = -np.pi + (x + 0.5) * dlon

            cos_zenith = (
                np.sin(lat) * np.sin(substellar_lat)
                + np.cos(lat) * np.cos(substellar_lat)
                * np.cos(lon - substellar_lon)
            )

            mu = max(0.0, cos_zenith)
            illumination[y, x] = mu

            T_eq[y, x] = (
                T_night + (T_day - T_night) * np.sqrt(mu)
            )

            p_mean += pressure[y, x]
            T_mean += T[y, x]

    p_mean /= ny * nx
    T_mean /= ny * nx

    # --------------------------------------------------
    # 2. PAINE-EROT, TUULET JA KESKUSMYRSKYN PYÖRRE
    # --------------------------------------------------

    for y in range(ny):
        lat = -0.5 * np.pi + (y + 0.5) * dlat
        cosphi = max(0.05, np.cos(lat))
        dx = R * dlon * cosphi

        yn = min(y + 1, ny - 1)
        ys = max(y - 1, 0)

        f = 2.0 * omega * np.sin(lat)

        for x in range(nx):
            xe = (x + 1) % nx
            xw = (x - 1 + nx) % nx

            dpdx = (
                pressure[y, xe] - pressure[y, xw]
            ) / (2.0 * dx)

            dpdy = (
                pressure[yn, x] - pressure[ys, x]
            ) / (2.0 * dy)

            ua = -dpdx / rho_air + f * v[y, x]
            va = -dpdy / rho_air - f * u[y, x]

            un = (
                u[y, x] + dt * ua
            ) / (1.0 + drag * dt)

            vn = (
                v[y, x] + dt * va
            ) / (1.0 + drag * dt)

            if central_hurricane:
                lon = -np.pi + (x + 0.5) * dlon

                dlon_s = lon - substellar_lon

                # Pallopinnan paikallinen etäisyysvektori
                # tähdenalaisesta pisteestä.
                cos_dist = (
                    np.sin(substellar_lat) * np.sin(lat)
                    + np.cos(substellar_lat) * np.cos(lat)
                    * np.cos(dlon_s)
                )
                cos_dist = min(1.0, max(-1.0, cos_dist))
                dist = np.arccos(cos_dist)

                if dist > 1e-8 and dist < r_cloudy:
                    # Suunta itään/pohjoiseen tähdenalaisesta pisteestä.
                    east = np.cos(lat) * np.sin(dlon_s)
                    north = (
                        np.cos(substellar_lat) * np.sin(lat)
                        - np.sin(substellar_lat)
                        * np.cos(lat) * np.cos(dlon_s)
                    )

                    norm = np.sqrt(east * east + north * north)

                    if norm > 1e-10:
                        east /= norm
                        north /= norm

                        # Tangentin suuntainen tuuli.
                        # Vahvin sadealueen sisällä, heikkenee ulospäin.
                        z = max(0.0, 1.0 - dist / r_cloudy)
                        wind = hurricane_wind_max * z

                        # Pyörre kiertää vastapäivään paikallisessa
                        # itä-pohjois-koordinaatistossa.
                        un += -north * wind
                        vn += east * wind

            speed = np.sqrt(un * un + vn * vn)

            if speed > 100.0:
                scale = 100.0 / speed
                un *= scale
                vn *= scale

            u_new[y, x] = un
            v_new[y, x] = vn

    # --------------------------------------------------
    # 3. LÄMMÖN JA KOSTEUDEN ADVEKTIO
    # --------------------------------------------------

    for y in range(ny):
        lat = -0.5 * np.pi + (y + 0.5) * dlat
        dx = R * dlon * max(0.05, np.cos(lat))

        yn = min(y + 1, ny - 1)
        ys = max(y - 1, 0)

        for x in range(nx):
            xe = (x + 1) % nx
            xw = (x - 1 + nx) % nx

            ux = u_new[y, x]
            vy = v_new[y, x]

            if ux >= 0.0:
                dTdx = (T[y, x] - T[y, xw]) / dx
                dqdx = (q[y, x] - q[y, xw]) / dx
            else:
                dTdx = (T[y, xe] - T[y, x]) / dx
                dqdx = (q[y, xe] - q[y, x]) / dx

            if vy >= 0.0:
                dTdy = (T[y, x] - T[ys, x]) / dy
                dqdy = (q[y, x] - q[ys, x]) / dy
            else:
                dTdy = (T[yn, x] - T[y, x]) / dy
                dqdy = (q[yn, x] - q[y, x]) / dy

            T_adv = T[y, x] - dt * (
                ux * dTdx + vy * dTdy
            )

            q_adv = max(
                0.0,
                q[y, x] - dt * (
                    ux * dqdx + vy * dqdy
                )
            )

            # --------------------------------------------------
            # 4. SÄTEILY JA PINTAMATERIAALI
            # --------------------------------------------------

            m = material[y, x]
            if m < 0 or m > 3:
                m = 1

            thermal_factor = Cmat[1] / Cmat[m]

            alpha = min(
                1.0,
                dt / tau_rad * thermal_factor
            )

            T_rad = T_adv + alpha * (
                T_eq[y, x] - T_adv
            )

            T_new[y, x] = T_rad
            q_new[y, x] = q_adv

            day_lift = max(
                0.0,
                (T_eq[y, x] - T_night)
                / max(T_day - T_night, 1.0)
            )

            equatorial = max(0.0, np.cos(lat))

            lift_field[y, x] = min(
                1.0,
                day_lift * (0.75 + 0.25 * equatorial)
            )

    # --------------------------------------------------
    # 5. PAINEEN PÄIVITYS
    # --------------------------------------------------

    for y in range(ny):
        lat = -0.5 * np.pi + (y + 0.5) * dlat
        dx = R * dlon * max(0.05, np.cos(lat))

        yn = min(y + 1, ny - 1)
        ys = max(y - 1, 0)

        for x in range(nx):
            xe = (x + 1) % nx
            xw = (x - 1 + nx) % nx

            dudx = (
                u_new[y, xe] - u_new[y, xw]
            ) / (2.0 * dx)

            dvdy = (
                v_new[yn, x] - v_new[ys, x]
            ) / (2.0 * dy)

            divergence = dudx + dvdy

            p_target = p_mean - pressure_sensitivity * (
                T_eq[y, x] - T_mean
            ) / max(T_day - T_night, 1.0)

            dp_relax = (
                (p_target - pressure[y, x])
                / tau_pressure
            )

            dp_wave = -rho_air * c_eff**2 * divergence

            dp = dt * (dp_relax + dp_wave)
            dp = min(100.0, max(-100.0, dp))

            p_new[y, x] = max(
                1000.0,
                pressure[y, x] + dp
            )

    # --------------------------------------------------
    # 6. PILVET, SADE JA KOSTEUDEN TIIVISTYMINEN
    # --------------------------------------------------

    for y in range(ny):
        lat = -0.5 * np.pi + (y + 0.5) * dlat

        for x in range(nx):
            m = material[y, x]
            if m < 0 or m > 3:
                m = 1

            tc = T_new[y, x] - 273.15

            q_sat = max(
                1.0,
                4.85 * np.exp(0.065 * max(tc, -20.0))
            )

            rh = min(
                1.0,
                max(0.0, q_new[y, x] / q_sat)
            )

            rh_matrix[y, x] = rh

            z_cloud = min(
                1.0,
                max(0.0, (rh - 0.35) / 0.50)
            )

            cloud_base = (
                z_cloud * z_cloud * (3.0 - 2.0 * z_cloud)
            )

            lift = lift_field[y, x]

            cloud_val = cloud_base * (
                0.65 + 0.35 * lift
            )

            # Keskusmyrskyn pilvipeite.
            hurricane_rain = 0.0
            hurricane_cloud = 0.0

            if central_hurricane:
                lon = -np.pi + (x + 0.5) * dlon
                dlon_s = lon - substellar_lon

                cos_dist = (
                    np.sin(substellar_lat) * np.sin(lat)
                    + np.cos(substellar_lat) * np.cos(lat)
                    * np.cos(dlon_s)
                )

                cos_dist = min(1.0, max(-1.0, cos_dist))
                dist = np.arccos(cos_dist)

                if dist < r_rainy:
                    z = 1.0 - dist / r_rainy
                    hurricane_rain = z * z * (3.0 - 2.0 * z)

                if dist < r_cloudy:
                    zc = 1.0 - dist / r_cloudy
                    hurricane_cloud = (
                        zc * zc * (3.0 - 2.0 * zc)
                    )

                cloud_val = max(
                    cloud_val,
                    hurricane_cloud_max * hurricane_cloud
                )

            cloudiness[y, x] = min(
                1.0, max(0.0, cloud_val)
            )

            z_rain = min(
                1.0,
                max(0.0, (rh - 0.65) / 0.25)
            )

            rain_gate = (
                z_rain * z_rain * (3.0 - 2.0 * z_rain)
            )

            rain_intensity = (
                5.0 * lift * rain_gate * cloud_val
            )

            if m != 0:
                rain_intensity *= 0.8

            # Keskushurrikaani voi tuottaa sadetta myös
            # silloin, kun RH-pohjainen sadeparametrisointi
            # ei yksin tuottaisi sadetta.
            if central_hurricane:
                rain_intensity = max(
                    rain_intensity,
                    hurricane_rain_max * hurricane_rain
                )

            available_mm = (
                max(0.0, q_new[y, x]) * h_eff / 1000.0
            )

            max_rain_mm_h = (
                available_mm * 3600.0 / max(dt, 1.0)
            )

            rain_intensity = min(
                rain_intensity,
                max_rain_mm_h
            )

            rain_mm_h[y, x] = rain_intensity

            if rain_intensity > 0.0:
                rain_g_m3_s = (
                    rain_intensity * 1000.0
                    / (3600.0 * h_eff)
                )

                q_new[y, x] = max(
                    0.0,
                    q_new[y, x] - rain_g_m3_s * dt
                )

                cooling_power = rain_intensity * 20.0

                T_new[y, x] -= (
                    cooling_power * dt / Cmat[m]
                )

    return (
        T_new, q_new, p_new,
        rain_mm_h, rh_matrix, cloudiness,
        u_new, v_new
    )





@njit
def earth_like_weather(
    T, q, pressure, material, sun_decl, dt_seconds,
    itcz_coeff=1.0,
    h_eff=10.0,
    u=None,
    v=None
):
    ny, nx = T.shape

    # --------------------------------------------------
    # 1. VAKIOT JA HILAN MITAT
    # --------------------------------------------------
    R = 6_371_000.0
    omega = 7.2921e-5
    g = 9.81
    rho_air = 1.2
    c_eff = 25.0          # Tehollinen paineaaltojen nopeus
    drag = 1.0 / 86400.0  # Tuulen vaimenemisaika
    dx_lon = 2.0 * np.pi * R / nx
    dlat = np.pi / ny
    dy = R * dlat

    # dt kannattaa pitää riittävän pienenä.
    dt = dt_seconds

    if u is None:
        u = np.zeros((ny, nx), dtype=np.float64)
    if v is None:
        v = np.zeros((ny, nx), dtype=np.float64)

    T_new = T.copy()
    q_new = q.copy()
    p_new = pressure.copy()

    u_new = u.copy()
    v_new = v.copy()

    rain_mm_h = np.zeros((ny, nx), dtype=np.float64)
    rh_matrix = np.zeros((ny, nx), dtype=np.float64)
    cloudiness = np.zeros((ny, nx), dtype=np.float64)

    # Materiaalien lämpökapasiteetit [J/m²/K]
    Cmat = np.array([
        1030.0 * 4184.0 * 10.0,
        2700.0 * 750.0 * 2.0,
        917.0 * 2100.0 * 3.0,
        400.0 * 880.0 * 3.0
    ])

    # --------------------------------------------------
    # 2. PEHMEÄT ILMASTOVYÖHYKKEET
    # --------------------------------------------------
    itcz_lat = itcz_coeff * sun_decl
    updraft_profiles = np.empty(ny, dtype=np.float64)

    for y in range(ny):
        phi = -0.5 * np.pi + (y + 0.5) * dlat

        # ITCZ ja polaaririntamat
        itcz = np.exp(-((phi - itcz_lat) / 0.18)**2)
        polar = np.exp(-((abs(phi) - 1.0) / 0.35)**2)
        subsidence = np.sin(phi - itcz_lat)**2

        updraft_profiles[y] = (
            1.0 * itcz
            + 0.35 * polar
            - 0.35 * subsidence
        )

    # --------------------------------------------------
    # 3. PAINEGRADIENTTI JA TUULI
    # --------------------------------------------------
    for y in range(ny):
        phi = -0.5 * np.pi + (y + 0.5) * dlat

        # Pituuspiirisuunnan etäisyys pienenee navoilla.
        cosphi = max(0.05, np.cos(phi))
        dx = dx_lon * cosphi

        yn = min(y + 1, ny - 1)
        ys = max(y - 1, 0)

        for x in range(nx):
            xe = (x + 1) % nx
            xw = (x - 1 + nx) % nx

            # Painegradientit [Pa/m]
            dpdx = (
                pressure[y, xe] - pressure[y, xw]
            ) / (2.0 * dx)

            dpdy = (
                pressure[yn, x] - pressure[ys, x]
            ) / (2.0 * dy)

            # Coriolis-parametri
            f = 2.0 * omega * np.sin(phi)

            # Liikemäärän yhtälöt
            u_acc = -dpdx / rho_air + f * v[y, x]
            v_acc = -dpdy / rho_air - f * u[y, x]

            u_new[y, x] = (
                u[y, x] + dt * u_acc
            ) / (1.0 + drag * dt)

            v_new[y, x] = (
                v[y, x] + dt * v_acc
            ) / (1.0 + drag * dt)

            # Suoja liian suurilta nopeuksilta
            speed = np.sqrt(
                u_new[y, x]**2 + v_new[y, x]**2
            )
            if speed > 60.0:
                factor = 60.0 / speed
                u_new[y, x] *= factor
                v_new[y, x] *= factor

    # --------------------------------------------------
    # 4. LÄMPÖTILAN JA KOSTEUDEN ADVEKTIO
    # --------------------------------------------------
    # Upwind-differentiaatio vähentää numeerisia heilahteluja.
    # Advektio lasketaan vanhoista T- ja q-kentistä.

    for y in range(ny):
        phi = -0.5 * np.pi + (y + 0.5) * dlat
        cosphi = max(0.05, np.cos(phi))
        dx = dx_lon * cosphi

        yn = min(y + 1, ny - 1)
        ys = max(y - 1, 0)

        for x in range(nx):
            xe = (x + 1) % nx
            xw = (x - 1 + nx) % nx

            ux = u_new[y, x]
            vy = v_new[y, x]

            # Upwind-gradientit lämpötilalle
            if ux >= 0.0:
                dTdx = (T[y, x] - T[y, xw]) / dx
                dqdx = (q[y, x] - q[y, xw]) / dx
            else:
                dTdx = (T[y, xe] - T[y, x]) / dx
                dqdx = (q[y, xe] - q[y, x]) / dx

            if vy >= 0.0:
                dTdy = (T[y, x] - T[ys, x]) / dy
                dqdy = (q[y, x] - q[ys, x]) / dy
            else:
                dTdy = (T[yn, x] - T[y, x]) / dy
                dqdy = (q[yn, x] - q[y, x]) / dy

            T_new[y, x] = T[y, x] - dt * (
                ux * dTdx + vy * dTdy
            )

            q_new[y, x] = max(
                0.0,
                q[y, x] - dt * (
                    ux * dqdx + vy * dqdy
                )
            )

    # --------------------------------------------------
    # 5. PILVISYYS, SADE JA SATEEN JÄÄHDYTYS
    # --------------------------------------------------
    for y in range(ny):
        updraft = updraft_profiles[y]
        phi = -0.5 * np.pi + (y + 0.5) * dlat

        yn = min(y + 1, ny - 1)
        ys = max(y - 1, 0)

        for x in range(nx):
            m = material[y, x]
            if m < 0 or m > 3:
                m = 1

            tc = T_new[y, x] - 273.15

            # Kyllästyskosteuden likiarvo [g/m³]
            q_sat = max(
                1.0,
                4.85 * np.exp(0.065 * max(tc, -20.0))
            )

            rh = min(1.0, max(0.0, q_new[y, x] / q_sat))
            rh_matrix[y, x] = rh

            # Pehmeä kynnys: smoothstep
            z_cloud = min(1.0, max(0.0, (rh - 0.35) / 0.50))
            cloud_base = z_cloud * z_cloud * (3.0 - 2.0 * z_cloud)

            # Nousuliikkeen vaikutus pilvisyyteen
            cloud_val = cloud_base * (
                0.75 + 0.25 * min(1.0, max(0.0, updraft))
            )
            cloudiness[y, x] = min(1.0, max(0.0, cloud_val))

            # Sade voimistuu asteittain RH:n kasvaessa.
            z_rain = min(1.0, max(0.0, (rh - 0.65) / 0.25))
            rain_gate = z_rain * z_rain * (3.0 - 2.0 * z_rain)

            lift = max(0.0, updraft)
            rain_intensity = (
                5.0 * lift * rain_gate * cloud_val
            )

            if m != 0:
                rain_intensity *= 0.8

            # Käytettävissä olevan veden asettama yläraja.
            # q [g/m³] * h_eff [m] -> g/m².
            available_mm = max(0.0, q_new[y, x]) * h_eff / 1000.0
            max_rain_mm_h = available_mm * 3600.0 / max(dt, 1.0)

            rain_intensity = min(
                rain_intensity, max_rain_mm_h
            )
            rain_mm_h[y, x] = rain_intensity

            if rain_intensity > 0.0:
                rain_g_m3_s = (
                    rain_intensity * 1000.0 / 3600.0 / h_eff
                )

                q_new[y, x] = max(
                    0.0,
                    q_new[y, x] - rain_g_m3_s * dt
                )

                cooling_power = rain_intensity * 20.0

                T_new[y, x] -= (
                    cooling_power * dt / Cmat[m]
                )

    # --------------------------------------------------
    # 6. PAINEEN PÄIVITYS
    # --------------------------------------------------
    # Tehollinen matalataajuinen painevaste tuulen divergenssiin.
    # c_eff on tarkoituksella pieni numeerisen vakauden parantamiseksi.

    for y in range(ny):
        phi = -0.5 * np.pi + (y + 0.5) * dlat
        dx = dx_lon * max(0.05, np.cos(phi))
        yn = min(y + 1, ny - 1)
        ys = max(y - 1, 0)

        for x in range(nx):
            xe = (x + 1) % nx
            xw = (x - 1 + nx) % nx

            dudx = (
                u_new[y, xe] - u_new[y, xw]
            ) / (2.0 * dx)

            dvdy = (
                v_new[yn, x] - v_new[ys, x]
            ) / (2.0 * dy)

            divergence = dudx + dvdy

            dp = -rho_air * c_eff**2 * divergence * dt

            # Rajoitetaan yhden askeleen paineen muutos.
            dp = min(100.0, max(-100.0, dp))
            p_new[y, x] = max(1000.0, pressure[y, x] + dp)

    return (
        T_new, q_new, p_new,
        rain_mm_h, rh_matrix, cloudiness,
        u_new, v_new
    )




def generate_planet_base(width, height, landmask, max_blur_distance=20):
    """
    Luo lämpötila- (K) ja kosteusrasterit (g/m3) Maan etäisyydellä olevalle planeetalle.
    landmask: 2D-numpy-taulukko (1 = maa, 0 = meri)
    max_blur_distance: Kuinka monta pikseliä rannikon vaikutus ulottuu sisämaahan/merelle
    """
    # --- 1. LÄMPÖTILA LEVEYSASTEIDEN MUKAAN ---
    # Luodaan leveysasteet y-akselille (-90 etelänapa, 0 päiväntasaaja, 90 pohjoisnapa)
    latitudes = np.linspace(-90, 90, height)
    # Tehdään 2D-rasteri toistamalla leveysasteet leveyssuunnassa
    lat_grid = np.tile(latitudes[:, np.newaxis], (1, width))
    
    # Peruslämpötila kosinifunktiolla: Päiväntasaaja ~300K (+27°C), navat ~225K (-48°C)
    # (Huom: voit lisätä tähän kohinaa/noisea tekemään vyöhykkeistä elävämpiä)
    temp_equator = 300.15
    temp_pole = 225.15
    temperature_k = temp_pole + (temp_equator - temp_pole) * np.cos(np.radians(lat_grid))

    # --- 2. PEHMEÄ SUHTEELLINEN KOSTEUS (RH) ---
    # Lasketaan etäisyys maasta mereen ja merestä maahan
    dist_to_sea = distance_transform_edt(landmask == 1)   # Etäisyys merelle (kun ollaan maalla)
    dist_to_land = distance_transform_edt(landmask == 0)  # Etäisyys maalle (kun ollaan merellä)
    
    # Normalisoidaan etäisyydet välille 0.0 - 1.0 halutulla vaikutusalueella
    factor_land = np.clip(dist_to_sea / max_blur_distance, 0, 1)  # 0 rannikolla, 1 syvällä sisämaassa
    factor_sea = np.clip(dist_to_land / max_blur_distance, 0, 1)   # 0 rannikolla, 1 avomerellä
    
    # Interpoloidaan kosteusarvot rannikon (0.85) ja ääripäiden välillä
    #rh_land = 0.85 - (0.85 - 0.40) * factor_land  # Laskee 85% -> 40%
    #rh_sea = 0.85 + (0.95 - 0.85) * factor_sea   # Nousee 85% -> 95%
    # Interpoloidaan kosteusarvot rannikon (0.85) ja ääripäiden välillä
    rh_land = 0.1 - (0.1 - 0.0001) * factor_land  # Laskee 85% -> 40%
    rh_sea = 0.85 + (0.95 - 0.85) * factor_sea   # Nousee 85% -> 95%    
    # Yhdistetään landmaskin avulla
    relative_humidity = np.where(landmask == 1, rh_land, rh_sea)

    # --- 3. ABSOLUUTTINEN KOSTEUS (g/m3) ---
    T_c = temperature_k - 273.15
    e_sat = 611.2 * np.exp((17.67 * T_c) / (T_c + 243.5))
    max_humidity = (e_sat / (461.5 * temperature_k)) * 1000
    
    # Lopullinen absoluuttinen kosteus
    humidity_raster = max_humidity * relative_humidity

    return temperature_k, humidity_raster


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
  
@njit(cache=True)
def spread_ice(
    demup0,
    landmask,
    glacier, 
    dt_seconds,
    flow_rate=0.02,
    min_glacier_m=1.0,
    max_transfer_fraction=0.1, planet_radius_m=6371000
):
    h, w = glacier.shape

    # Uusi rasteri: päivitykset eivät vaikuta naapureihin
    delta = np.zeros((h, w), dtype=np.float64)

    # Naapurisolut: 4-suuntaisesti
    dy = np.array([-1, 1, 0, 0])
    dx = np.array([0, 0, -1, 1])

    dt_h = dt_seconds / 3600.0

    for y in range(h):
        for x in range(w):
            ice = glacier[y, x]

            if landmask[y, x] == 0 or ice < min_glacier_m:
                continue

            surface = demup0[y, x] + ice

            # Jäätikön paksuus ja pinnan kaltevuus ohjaavat virtausta
            for k in range(4):
                ny = y + dy[k]
                nx = x + dx[k]

                # Ei wrap-aroundia rasterin reunoilla
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue

                if landmask[ny, nx] == 0:
                    continue

                neighbor_surface = demup0[ny, nx] + glacier[ny, nx]
                slope = surface - neighbor_surface

                if slope <= 0.0:
                    continue

                # Yksinkertainen kaltevuuteen perustuva virtaus
                transfer = (
                    flow_rate
                    * slope
                    * dt_h
                    * 0.01
                )

                # Rajoita siirto, jotta solu ei tyhjene kerralla
                max_transfer = ice * max_transfer_fraction
                transfer = min(transfer, max_transfer)

                if transfer > 0.0:
                    delta[y, x] -= transfer
                    delta[ny, nx] += transfer

    # Sovella kaikki muutokset vasta laskennan jälkeen
    for y in range(h):
        for x in range(w):
            glacier[y, x] = max(
                0.0,
                glacier[y, x] + delta[y, x],
            )  
    return(glacier) 
        
@njit(cache=True)
def update_ice(
    demup0,
    landmask,
    temperature,
    rain_mm_h,
    snow,
    glacier,
    sea_ice,
    material,
    snow_mask,
    snow_depth_m,
    glacier_mask,
    glacier_depth_m,
    sea_ice_mask,
    sea_ice_depth_m,
    dt_seconds,
    planet_radius_m=6371000.0,
):
    h, w = demup0.shape
    dt_h = dt_seconds / 3600.0

    # Lämpötilarajat Celsius-asteina
    snow_temp_low = -1.0
    snow_temp_high = 2.0

    # Sulamisparametrit
    snow_melt_rate = 0.10
    glacier_melt_rate = 0.005

    # Merijään parametrit
    sea_freeze_temp = -1.8
    sea_freeze_rate = 0.02
    sea_melt_rate = 0.005

    glacier_formation_mm = 3000.0
    sea_ice_material_min = 0.1
    max_sea_ice = 5.0

    # Lumen tiheys kg/m³
    snow_density = 100.0

    for y in range(h):
        for x in range(w):
            t = temperature[y, x] - 273.15
            p = max(rain_mm_h[y, x], 0.0) * dt_h

            if landmask[y, x] != 0:

                # Sateen olomuoto
                if t <= snow_temp_low:
                    snow_fraction = 1.0
                elif t >= snow_temp_high:
                    snow_fraction = 0.0
                else:
                    snow_fraction = (
                        snow_temp_high - t
                    ) / (snow_temp_high - snow_temp_low)

                snow[y, x] += p * snow_fraction

                # Lumensulaminen
                melt = min(
                    snow[y, x],
                    snow_melt_rate * max(t, 0.0) * dt_h,
                )
                snow[y, x] -= melt

                # Lumesta jäätiköksi
                if snow[y, x] >= glacier_formation_mm:
                    excess_mm = snow[y, x] - glacier_formation_mm
                    glacier[y, x] += excess_mm / 1000.0 * 0.9
                    snow[y, x] = glacier_formation_mm

                # Jäätikön sulaminen
                glacier_melt = min(
                    glacier[y, x],
                    glacier_melt_rate * max(t, 0.0) * dt_h,
                )
                glacier[y, x] -= glacier_melt

                # Päivitä syvyydet
                snow_depth_m[y, x] = (
                    snow[y, x] / snow_density
                )
                glacier_depth_m[y, x] = max(glacier[y, x], 0.0)

                # Päivitä maskit
                snow_mask[y, x] = 1 if snow[y, x] > 0.1 else 0
                glacier_mask[y, x] = (
                    1 if glacier[y, x] > 0.001 else 0
                )

                # Ei merijäätä maalla
                sea_ice[y, x] = 0.0
                sea_ice_mask[y, x] = 0
                sea_ice_depth_m[y, x] = 0.0

                if snow_mask[y, x] == 1 or glacier_mask[y, x] == 1:
                    material[y, x] = 3
                else:
                    material[y, x] = 1

            else:
                # Merellä ei ole lunta tai mannerjäätikköä
                snow[y, x] = 0.0
                glacier[y, x] = 0.0

                snow_mask[y, x] = 0
                snow_depth_m[y, x] = 0.0
                glacier_mask[y, x] = 0
                glacier_depth_m[y, x] = 0.0

                # Merijään muodostuminen ja sulaminen
                if t < sea_freeze_temp:
                    sea_ice[y, x] += (
                        sea_freeze_rate
                        * (sea_freeze_temp - t)
                        * dt_h
                    )
                elif t > 0.0:
                    sea_ice[y, x] -= (
                        sea_melt_rate * t * dt_h
                    )

                sea_ice[y, x] = min(
                    max(sea_ice[y, x], 0.0),
                    max_sea_ice,
                )

                sea_ice_depth_m[y, x] = sea_ice[y, x]
                sea_ice_mask[y, x] = (
                    1 if sea_ice[y, x] >= sea_ice_material_min else 0
                )

                if sea_ice_mask[y, x] == 1:
                    material[y, x] = 2
                else:
                    material[y, x] = 0

    return (
        snow, glacier, sea_ice, material,
        snow_mask, snow_depth_m,
        glacier_mask, glacier_depth_m,
        sea_ice_mask, sea_ice_depth_m,
    )

       
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
    #print(cosz)
    if cosz <= 0.0:
        return (0.0, 0.0, 0.0, 0.0)

    radiation=solar_constant * rfac * cosz
    #if(radiation==None):
    #    radiation=0
    #if(decl==None):
    #    decl=0
    #if(H==None):
    #    H=0
    #if(cosz==None):
    #    cosz=0
    #print("--------")
    #print(decl, H, cosz)
    return (radiation, decl, H, cosz)


@njit
def snapshot_solar_radiation(
    solar_constant,
    star_longitude,
    star_latitude,
    nx,
    ny,
):
    deg = math.pi / 180.0

    star_lon = star_longitude * deg
    star_lat = star_latitude * deg

    # Muoto: (korkeus, leveys) = (nx, ny)
    radiation = np.zeros((nx, ny), dtype=np.float64)

    for j in range(nx):  # pystysuunta: leveysaste
        lat = (90.0 - 180.0 * j / (nx - 1)) * deg

        for i in range(ny):  # vaakasuunta: pituusaste
            lon = (-180.0 + 360.0 * i / ny) * deg

            cosz = (
                math.sin(lat) * math.sin(star_lat)
                + math.cos(lat) * math.cos(star_lat)
                * math.cos(lon - star_lon)
            )

            if cosz > 0.0:
                radiation[j, i] = solar_constant * cosz

    return radiation


def cosine_weighted_average(temperature_grid, dimy, dimx):
    """
    Laskee kosinipainotetun keskiarvon lämpötilahilalle.
    
    Parametrit:
    - temperature_grid: 2D numpy-taulukko muodossa (dimy, dimx)
    - dimy: Leveysasteiden (latitude) määrä (-90:stä 90:een)
    - dimx: Pituusasteiden (longitude) määrä (-180:stä 180:een)
    """
    # 1. Luodaan leveysastevektori välille -90 ja 90 astetta
    # Huom: varmista vastaako ensimmäinen rivi -90:tä vai 90:tä astetta (tarvittaessa flip)
    lats_deg = np.linspace(-90, 90, dimy)
    
    # 2. Muutetaan leveysasteet radiaaneiksi ja lasketaan kosinipainot
    lats_rad = np.radians(lats_deg)
    weights = np.cos(lats_rad)  # Pituus on dimy
    
    # 3. Laajennetaan painovektori 2D-muotoon (dimy, dimx), jotta se vastaa datan muotoa
    # Jokainen pituusaste saa saman leveysastekohtaisen painon
    weight_grid = np.repeat(weights[:, np.newaxis], dimx, axis=1)
    
    # Jos datassa on puuttuvia arvoja (NaN), käytetään np.nanaverage-tyylistä painotusta
    # Korvaa nan-arvot nollilla painomaskissa, jotta ne eivät vääristä tulosta
    nan_mask = np.isnan(temperature_grid)
    weight_grid[nan_mask] = 0
    temp_zeroed = np.nan_to_num(temperature_grid, nan=0.0)
    
    # 4. Lasketaan painotettu keskiarvo
    weighted_mean = np.sum(temp_zeroed * weight_grid) / np.sum(weight_grid)
    
    return weighted_mean



# ------------------------------------------------------------
# INITIAL CAPACITY + ALBEDO ARRAYS
# ------------------------------------------------------------

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


import numpy as np
from numba import njit

@njit
def update_albedo_scientific_with_humidity(T, q, material, base_albedo, use_ice=True):
    ny, nx = T.shape
    current_albedo = base_albedo.copy()
    
    if not use_ice:
        return current_albedo

    # --- FYSIIKAN VAKIOT JA PARAMETRIT ---
    ALBEDO_OPEN_OCEAN = 0.07    # Puhtaan avomeren matala albedo
    ALBEDO_MAX_SEA_ICE = 0.65   # Kuiva, paksu merijää
    ALBEDO_MAX_LAND_SNOW = 0.75  # Puhdas kuiva lumi avomaalla

    # Meriveden kriittiset lämpötilat
    T_ocean_freeze = 271.15     # -2 °C
    T_ocean_melt   = 274.15     # +1 °C

    # Maan kriittinen lämpötila ja jyrkkyyskerroin
    T_land_center = 273.15      # 0 °C
    gamma_land = 2.0            # S-käyrän leveys asteina

    # Kosteuden (g/m³) parametrit maaperän tummumiselle
    # Tyypillinen maanpinnan lähellä oleva absoluuttinen kosteus vaihtelee välillä 0 ... 20+ g/m³
    Q_DRY = 2.0                 # Tämän alla maa on täysin kuivaa (maksimialbedo)
    Q_WET = 15.0                # Tämän yllä maa on täysin saturoitunut/märkää (minimialbedo)
    SOIL_DARKENING_FACTOR = 0.65 # Märkä maa heijastaa vain 65% kuivan maan albedosta (esim. 0.30 -> ~0.20)

    for y in range(ny):
        for x in range(nx):
            m = material[y, x]
            t_val = T[y, x]
            q_val = q[y, x]

            if m == 0:  # --- VALTAMERI ---
                if t_val >= T_ocean_melt:
                    current_albedo[y, x] = ALBEDO_OPEN_OCEAN
                elif t_val <= T_ocean_freeze:
                    current_albedo[y, x] = ALBEDO_MAX_SEA_ICE
                else:
                    frac = (t_val - T_ocean_freeze) / (T_ocean_melt - T_ocean_freeze)
                    ice_factor = 1.0 - (frac * frac)
                    current_albedo[y, x] = (1.0 - ice_factor) * ALBEDO_OPEN_OCEAN + ice_factor * ALBEDO_MAX_SEA_ICE

            else:  # --- MANNER / MAATYYPIT (1, 2, 3) ---
                # 1. Lasketaan ensin kosteuden aiheuttama maaperän tummuminen (vain sulalle maalle)
                if q_val <= Q_DRY:
                    humidity_darkening = 1.0  # Ei tummumista, täysin kuivaa
                elif q_val >= Q_WET:
                    humidity_darkening = SOIL_DARKENING_FACTOR  # Täysi tummuminen
                else:
                    # Lineaarinen siirtymä kuivan ja märän välillä
                    q_frac = (q_val - Q_DRY) / (Q_WET - Q_DRY)
                    humidity_darkening = 1.0 - q_frac * (1.0 - SOIL_DARKENING_FACTOR)

                # Sovelletaan tummuminen pohja-albedoon ennen lumisateen tarkistusta
                darkened_base = base_albedo[y, x] * humidity_darkening

                # 2. Lasketaan lumen/jään osuus lämpötilan S-käyrällä (tanh)
                tanh_val = np.tanh((t_val - T_land_center) / gamma_land)
                snow_factor = 0.5 - 0.5 * tanh_val
                
                # Jos ilmankosteus on erittäin korkea, lumi muuttuu sohjoksi/märkälumeksi, mikä pudottaa lumialbedoa
                effective_max_snow = ALBEDO_MAX_LAND_SNOW
                if q_val > 10.0 and t_val > 271.15:
                    # Pudotetaan maksimilumialbedoa suhteessa kosteuteen (max 15% pudotus sohjolle)
                    wet_snow_penalty = 0.15 * (min(q_val, Q_WET) - 10.0) / (Q_WET - 10.0)
                    effective_max_snow *= (1.0 - wet_snow_penalty)

                # Varmistetaan, ettei lumialbedo ole matalampi kuin alla oleva maasto
                max_possible_albedo = np.maximum(darkened_base, effective_max_snow)
                
                # Lopullinen albedo sulan (kosteusmuokatun) ja lumisen pinnan välillä
                current_albedo[y, x] = (1.0 - snow_factor) * darkened_base + snow_factor * max_possible_albedo

    return current_albedo


import numpy as np
from numba import njit

@njit
def update_albedo_scientific(T, material, base_albedo, use_ice=True):
    ny, nx = T.shape
    current_albedo = base_albedo.copy()
    
    if not use_ice:
        return current_albedo

    # Fysikaaliset vakiot
    ALBEDO_OPEN_OCEAN = 0.07   # Puhtaan avomeren matala albedo
    ALBEDO_MAX_SEA_ICE = 0.65  # Kuiva, paksu merijää
    ALBEDO_MAX_LAND_SNOW = 0.75 # Puhdas lumi tundralla/avomaalla

    # Meriveden kriittiset lämpötilat (Kelvineinä)
    T_ocean_freeze = 271.15     # -2 °C: Suolainen vesi alkaa hyytyä kunnolla
    T_ocean_melt   = 274.15     # +1 °C: Pintajää ja lätäköt sulavat täysin avovedeksi

    # Maan kriittinen lämpötila ja jyrkkyyskerroin
    T_land_center = 273.15      # 0 °C: Makean veden ja lumen sulamispiste
    gamma_land = 2.0            # S-käyrän leveys asteina (kaikki muutos tapahtuu välillä -4C ... +4C)

    for y in range(ny):
        for x in range(nx):
            m = material[y, x]
            t_val = T[y, x]

            if m == 0:  # VALTAMERI
                if t_val >= T_ocean_melt:
                    # Täysin sula avomeri
                    current_albedo[y, x] = ALBEDO_OPEN_OCEAN
                elif t_val <= T_ocean_freeze:
                    # Täysin paksu ja kylmä merijää
                    current_albedo[y, x] = ALBEDO_MAX_SEA_ICE
                else:
                    # Parabolinen käyrä: albedo nousee nopeasti heti kun alitetaan sulamispiste
                    # Kuvastaa sohjoutumista ja sulamislammikoiden katoamista
                    frac = (t_val - T_ocean_freeze) / (T_ocean_melt - T_ocean_freeze)
                    ice_factor = 1.0 - (frac * frac)  # Kaareutuu ylöspäin kohti kylmää
                    
                    current_albedo[y, x] = (1.0 - ice_factor) * ALBEDO_OPEN_OCEAN + ice_factor * ALBEDO_MAX_SEA_ICE

            else:  # MANNER / MAATYYPIT (1, 2, 3)
                # Käytetään tanh-funktiota (S-käyrää), joka tekee jyrkän pudotuksen 0 °C kohdalla
                # Kun T << 273.15, tanh -> -1, jolloin arg -> 1.0 (täysi lumi)
                # Kun T >> 273.15, tanh ->  1, jolloin arg -> 0.0 (perusalbedo)
                tanh_val = np.tanh((t_val - T_land_center) / gamma_land)
                snow_factor = 0.5 - 0.5 * tanh_val
                
                # Huomioidaan maaston maksimilumialbedo: metsäiset alueet (korkea base_albedo) 
                # eivät nouse yhtä korkealle kuin paljas maa
                max_possible_albedo = np.maximum(base_albedo[y, x], ALBEDO_MAX_LAND_SNOW)
                
                current_albedo[y, x] = (1.0 - snow_factor) * base_albedo[y, x] + snow_factor * max_possible_albedo

    return current_albedo



	

import numpy as np
from numba import njit

@njit
def update_albedo_with_ice(T, material, base_albedo, use_ice=True):
    ny, nx = T.shape
    current_albedo = base_albedo.copy()
    
    if not use_ice:
        return current_albedo

    # Jää/lumialbedojen tyypilliset arvot
    ALBEDO_SEA_ICE = 0.60
    ALBEDO_LAND_ICE = 0.70  # Lumi/mannerjää heijastaa enemmän kuin merijää

    # Määritellään kynnykset: [keskipiste, poikkeama]
    # materiaali 0 = valtameri, muut (1, 2, 3) = manner / maatyypit
    ice_config = np.array([
        [270.0, 4.0],  # 0: Valtameri (sula > 274, jäässä < 266)
        [273.0, 3.0],  # 1: Manner    (sula > 276, jäässä < 270)
        [273.0, 3.0],  # 2: Merijää   (oletus/pohja, jos muuttuu)
        [273.0, 3.0],  # 3: Lumi/Mannerjää
    ], dtype=np.float64)

    for y in range(ny):
        for x in range(nx):
            m = material[y, x]
            if m < 0 or m > 3:
                m = 1
                
            t_center = ice_config[m, 0]
            t_dev = ice_config[m, 1]
            
            t_freeze = t_center - t_dev  # Alaraja (täysin jäässä)
            t_melt = t_center + t_dev    # Yläraja (täysin sula)
            
            t_val = T[y, x]
            
            # Lasketaan jään osuus (0.0 = ei jäätä, 1.0 = täysin jäässä)
            if t_val >= t_melt:
                ice_fraction = 0.0
            elif t_val <= t_freeze:
                ice_fraction = 1.0
            else:
                # Lineaarinen interpolointi sulan ja jäätymisrajan välillä
                ice_fraction = (t_melt - t_val) / (t_melt - t_freeze)
            
            # Valitaan kohde-albedo jään tyypin mukaan
            target_ice_albedo = ALBEDO_SEA_ICE if m == 0 else ALBEDO_LAND_ICE
            
            # Painotettu keskiarvo perusalbedon ja jään albedon välillä
            current_albedo[y, x] = (1.0 - ice_fraction) * base_albedo[y, x] + ice_fraction * target_ice_albedo
            
    return current_albedo

import math
import numpy as np
from numba import njit

# Varmistetaan että SIGMA on määritelty Numban globaalissa avaruudessa
SIGMA = 5.670374419e-8

#@njit
#def cloud_albedo(cloud):
#    # Esimerkki pilven albedofunktiosta (korvaa omallasi jos käytössä eri kaava)
#    return 0.5 + 0.2 * cloud

import math
import numpy as np
from numba import njit

SIGMA = 5.670374419e-8
EMISSIVITY = 0.96


@njit
def nbody_energy_balance_step(
    temperature,
    radiation,
    material,
    step_days,
):
    nx, ny = temperature.shape

    new_temperature = np.empty_like(temperature)
    dt = step_days * 86400.0

    for i in range(nx):
        for j in range(ny):

            C, albedo = material_properties(material[i, j])

            T = temperature[i, j]
            absorbed = (1.0 - albedo) * radiation[i, j]
            emitted = EMISSIVITY * SIGMA * T**4

            net_flux = absorbed - emitted

            new_temperature[i, j] = (
                T + net_flux * dt / C
            )

    return new_temperature

import math
import numpy as np
from numba import njit

# Varmistetaan globaali SIGMA
SIGMA = 5.670374419e-8

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
    
    #base_albedo=update_albedo_scientific(temperature, material, base_albedo, use_ice=True)
    
    # --- ILMAKEHÄN PARAMETRIT ---
    P_REF = 101325.0
    #TAU_SW_REF = 0.22 
    TAU_SW_REF = 0.22   
    CO2_REF = 280.0
    HUMIDITY_REF = 7.5  # Globaali referenssikosteus g/m3
    EPSILON_REF = 0.62
    
    tau_ref = -math.log(1.0 - EPSILON_REF)
    
    for y in range(ny):

        # latitude from -90 to +90
        if ny == 1:
            lat = 0.0
        else:
            lat = -90.0 + 180.0 * y / (ny - 1)

        for x in range(nx):
            cloud = cloud_raster[y, x]
            cloud=0
            ca = cloud_albedo(cloud)
            longitude = -180.0 + 360.0 * x / nx

            T = temperature[y, x]
            hum = humidity[y, x]  # Haetaan solun kosteusarvo (g/m3)
            hum=0
            # ------------------------------------------------
            # solar radiation
            # ------------------------------------------------
            S, decl, H, cosz = solar_radiation(
                solar_constant,
                eccentricity,
                perihelion,
                tilt,
                lat,
                longitude,
                orbital_longitude,
                rotation_angle
            )
            
            if S is None:
                S = 0.0
				
            # ------------------------------------------------
            # ilmakehän vaimennus lyhytaaltoiselle säteilylle (PÄIVÄ)
            # ------------------------------------------------
            if S > 0.0:
                cos_zenith = S / (solar_constant * (1.0 + eccentricity)**2)
                if cos_zenith > 1.0: cos_zenith = 1.0
                
                if cos_zenith > 0.001:
                    airmass = 1.0 / cos_zenith
                    tau_sw = TAU_SW_REF * (pressure_Pa / P_REF)
                    S = S * math.exp(-tau_sw * airmass)
                else:
                    S = 0.0

            # ------------------------------------------------
            # surface albedo
            # ------------------------------------------------
            A0 = base_albedo[y, x]

            if (clouds_present > 0):
                A = (
                    A0 * (1.0 - cloud * cloud_strength)
                    + ca * cloud * cloud_strength
                )
            else:
                A = A0

            if A < 0.0: A = 0.0
            if A > 1.0: A = 1.0

            # ------------------------------------------------
            # absorbed shortwave
            # ------------------------------------------------
            ASR = (1.0 - A) * S

            # ------------------------------------------------
            # Kasvihuoneilmiö: CO2 + Kosteus (H2O) -> OLR
            # ------------------------------------------------
            # CO2 osuus optisesta paksuudesta
            #co2_ratio = max(CO2_REF/co2_ppm , 1e-30) ## 
            co2_ratio = max(co2_ppm/CO2_REF , 1e-30)
            tau_CO2 = tau_ref * math.sqrt(co2_ratio)
            
            # Vesihöyryn osuus optisesta paksuudesta
            humidity_ratio = max(hum / HUMIDITY_REF, 1e-30)
            tau_H2O = tau_ref * math.sqrt(humidity_ratio)
            
            # Yhdistetään painotetusti (70% vesihöyry, 30% CO2)
            #tau_clear = 0.3 * tau_CO2 + 0.7 * tau_H2O
            tau_clear = 1.0 * tau_CO2            
            # Selkeän taivaan imukyky
            epsilon_clear = 1.0 - math.exp(-tau_clear)

            # ------------------------------------------------
            # YÖ/PÄIVÄVAIKUTUS: Pilvien pitkäaaltoinen peitovaikutus (OLR)
            # ------------------------------------------------
            if (clouds_present > 0):
                epsilon_cloud_effect = 0.18 * cloud * cloud_strength
                epsilon_effective = epsilon_clear - epsilon_cloud_effect
                
                if epsilon_effective < 0.05:
                    epsilon_effective = 0.05
            else:
                epsilon_effective = epsilon_clear

            # Muutetaan transmissiivisuudeksi (mitä karkaa läpi avaruuteen)
            transmissivity = 1.0 - epsilon_effective
            if transmissivity < 0.1: transmissivity = 0.1
            
            OLR = transmissivity * SIGMA * T**4

            # ------------------------------------------------
            # energy balance
            # ------------------------------------------------
            dT = dt_seconds / capacity[y, x] * (
                ASR - OLR
            )

            Tnew = T + dT

            # ------------------------------------------------
            # ice feedback
            # ------------------------------------------------
            if use_ice:
                if material[y, x] == 0 and Tnew < ice_temperature:
                    material[y, x] = 2
                elif material[y, x] == 2 and Tnew >= ice_temperature:
                    material[y, x] = 0
                elif material[y, x] == 1 and Tnew < ice_temperature:
                    material[y, x] = 3
                elif material[y, x] == 3 and Tnew >= ice_temperature:
                    material[y, x] = 1

            new_temperature[y, x] = Tnew
            
            # Jos kosteus muuttuu ajan myötä, voit laskea uuden arvon tähän.
            # Toistaiseksi pidetään kosteus muuttumattomana tässä askeleessa:
            new_humidity[y, x] = hum

    return new_temperature, ASR, decl, H, cosz




# ------------------------------------------------------------
# OPTIONAL SIMPLE DIFFUSION
#
# This is deliberately separate from the energy balance.
# diffusion = 0 means no horizontal heat transport.
# ------------------------------------------------------------

@njit
def diffuse_temperature_origo_nosphere(T, diffusion):
    ny, nx = T.shape
    Tout = T.copy()

    if diffusion <= 0.0:
        return Tout

    for y in range(1, ny - 1):
        for x in range(nx):
            xm = (x - 1) % nx
            xp = (x + 1) % nx

            laplace = (
                T[y, xm]
                + T[y, xp]
                + T[y - 1, x]
                + T[y + 1, x]
                - 4.0 * T[y, x]
            )

            Tout[y, x] = T[y, x] + diffusion * laplace

    return Tout





















@njit
def diffuse_temperature_and_humidity_nosphere(
    T,
    humidity,
    diffusion_T=1,
    diffusion_H=1,
):
    ny, nx = T.shape

    Tout = T.copy()
    Hout = humidity.copy()

    # ==========================================
    # 1. LÄMPÖTILAN DIFFUUSIO
    # ==========================================
    if diffusion_T > 0.0:
        for y in range(1, ny - 1):
            for x in range(nx):
                xm = (x - 1) % nx
                xp = (x + 1) % nx

                laplace_T = (
                    T[y, xm]
                    + T[y, xp]
                    + T[y - 1, x]
                    + T[y + 1, x]
                    - 4.0 * T[y, x]
                )

                Tout[y, x] = (
                    T[y, x] + diffusion_T * laplace_T
                )

    # ==========================================
    # 2. KOSTEUDEN DIFFUUSIO
    # ==========================================
    if diffusion_H > 0.0:
        for y in range(1, ny - 1):
            for x in range(nx):
                xm = (x - 1) % nx
                xp = (x + 1) % nx

                laplace_H = (
                    humidity[y, xm]
                    + humidity[y, xp]
                    + humidity[y - 1, x]
                    + humidity[y + 1, x]
                    - 4.0 * humidity[y, x]
                )

                Hout[y, x] = (
                    humidity[y, x] + diffusion_H * laplace_H
                )

    return Tout, Hout


import numpy as np
from numba import njit

@njit
def diffuse_temperature_and_humidity_scaled_advanced(
    T,
    q,
    material,
    K0,                  # Pohjadiffuusio (esim. 0.65)
    dt,
    omega_ratio=1.0,     # Planeetan pyörimisnopeus suhteessa Maahan (1.0 = Maa)
    v_hadley_max=0.05,   # Hadley-solun maksimituulennopeus skaalatussa avaruudessa
    k_ocean=1.0,
    k_land=0.2,
    k_sea_ice=0.2,
    k_snow_ice=0.2
):
    #dt=dt/(3600*24)
    ny, nx = T.shape
    
    if K0 <= 0.0 or dt <= 0.0:
        return T.copy(), q.copy()

    dlat = np.pi / ny
    dlon = 2.0 * np.pi / nx

    kmat = np.array([k_ocean, k_land, k_sea_ice, k_snow_ice], dtype=np.float64)
    Gmat = K0 * kmat

    # Alustetaan perussolut
    G_cell = np.empty((ny, nx), dtype=np.float64)
    for y in range(ny):
        for x in range(nx):
            m = material[y, x]
            if m < 0 or m > 3:
                m = 1
            G_cell[y, x] = Gmat[m]

    # Pinnan johtavuudet
    Ge = np.zeros((ny, nx), dtype=np.float64)
    Gn = np.zeros((ny - 1, nx), dtype=np.float64)
    
    # Hadley-solun tuulennopeus pohjois-etelä-rajapinnoilla [suhteellinen yksikkö]
    # Positiivinen arvo tarkoittaa virtausta etelästä pohjoiseen (kohti pohjoisnapaa)
    v_hadley = np.zeros(ny - 1, dtype=np.float64)

    # 1. ITÄ-LÄNSI VIRTAUKSEN GEOMETRIA (Coriolis ei estä itä-länsi sekoittumista)
    for y in range(ny):
        phi = -0.5 * np.pi + (y + 0.5) * dlat
        cos_phi = np.cos(phi)
        if cos_phi < 1e-5:
            cos_phi = 1e-5
        geom_factor_e = cos_phi * dlat / dlon

        for x in range(nx):
            xp = (x + 1) % nx
            g1, g2 = G_cell[y, x], G_cell[y, xp]
            if g1 > 0.0 and g2 > 0.0:
                Ge[y, x] = (2.0 * g1 * g2 / (g1 + g2)) * geom_factor_e

    # 2. POHJOIS-ETELÄ VIRTAUKSEN GEOMETRIA & FYSIIKKA (Coriolis + Hadley)
    # Pyörimisnopeuden vaimennuskerroin pohjois-etelä-suunnan diffuusiolle
    # Jos omega_ratio kasvaa, diffuusio kohti napoja heikkenee dramaattisesti
    rotational_damping = 1.0 / (1.0 + (omega_ratio * omega_ratio - 1.0))
    if rotational_damping < 0.01:
        rotational_damping = 0.01 # Estetään täydellinen pysähtyminen

    for y in range(ny - 1):
        phi_edge = -0.5 * np.pi + (y + 1) * dlat
        geom_factor_n = np.cos(phi_edge) * dlon / dlat

        # Lasketaan Hadley-solun nopeus rajapinnalla:
        # Tyypillinen Hadley-solu kuljettaa lämpöä yläilmakehässä poispäin päiväntasaajalta.
        # Käytetään yksinkertaista siniaalto-parametrointia, joka vaihtaa suuntaa hevosleveysasteilla (~30 astetta)
        # Maassa sin(3*phi) mallintaa Hadley-, Ferrel- ja Polaarisen solun rajoja.
        v_hadley[y] = v_hadley_max * np.sin(3.0 * phi_edge) * np.cos(phi_edge)

        for x in range(nx):
            g1, g2 = G_cell[y, x], G_cell[y + 1, x]
            if g1 > 0.0 and g2 > 0.0:
                gf = 2.0 * g1 * g2 / (g1 + g2)
                # Vaimennetaan pohjois-eteläsuuntaista diffuusiota pyörimisnopeuden mukaan
                Gn[y, x] = gf * geom_factor_n * rotational_damping

    # 3. AIKAINTEGROINTI (Hila-alueiden päivitys)
    T_new = T.copy()
    q_new = q.copy()

    for y in range(ny):
        phi_south = -0.5 * np.pi + y * dlat
        phi_north = -0.5 * np.pi + (y + 1) * dlat
        cell_area = dlon * (np.sin(phi_north) - np.sin(phi_south))
        if cell_area <= 0.0:
            continue

        for x in range(nx):
            xm = (x - 1) % nx
            xp = (x + 1) % nx

            # --- ITÄ-LÄNSI DIFFUUSIO (Puhdas diffuusio) ---
            flux_e_T = Ge[y, x] * (T[y, xp] - T[y, x])
            flux_w_T = Ge[y, xm] * (T[y, xm] - T[y, x])
            flux_e_q = Ge[y, x] * (q[y, xp] - q[y, x])
            flux_w_q = Ge[y, xm] * (q[y, xm] - q[y, x])

            # --- POHJOIS-ETELÄ DIFFUUSIO ---
            flux_n_diff_T = Gn[y, x] * (T[y + 1, x] - T[y, x]) if y < ny - 1 else 0.0
            flux_s_diff_T = Gn[y - 1, x] * (T[y - 1, x] - T[y, x]) if y > 0 else 0.0
            flux_n_diff_q = Gn[y, x] * (q[y + 1, x] - q[y, x]) if y < ny - 1 else 0.0
            flux_s_diff_q = Gn[y - 1, x] * (q[y - 1, x] - q[y, x]) if y > 0 else 0.0

            # --- POHJOIS-ETELÄ HADLEY-ADVEKTIO (Upwind-menetelmä) ---
            flux_n_adv_T = 0.0
            flux_s_adv_T = 0.0
            flux_n_adv_q = 0.0
            flux_s_adv_q = 0.0

            # Pohjoinen rajapinta (y -> y+1)
            if y < ny - 1:
                v_n = v_hadley[y]
                # Leveysasteen huomioiva reunapinta-alakerroin advektiolle
                edge_factor = np.cos(-0.5 * np.pi + (y + 1) * dlat) * dlon
                if v_n > 0.0: # Tuuli puhaltaa etelästä pohjoiseen (meiltä pois)
                    flux_n_adv_T = -v_n * T[y, x] * edge_factor
                    flux_n_adv_q = -v_n * q[y, x] * edge_factor
                else:         # Tuuli puhaltaa pohjoisesta etelään (meille sisään)
                    flux_n_adv_T = -v_n * T[y + 1, x] * edge_factor
                    flux_n_adv_q = -v_n * q[y + 1, x] * edge_factor

            # Eteläinen rajapinta (y-1 -> y)
            if y > 0:
                v_s = v_hadley[y - 1]
                edge_factor = np.cos(-0.5 * np.pi + y * dlat) * dlon
                if v_s > 0.0: # Tuuli puhaltaa etelästä pohjoiseen (meille sisään)
                    flux_s_adv_T = v_s * T[y - 1, x] * edge_factor
                    flux_s_adv_q = v_s * q[y - 1, x] * edge_factor
                else:         # Tuuli puhaltaa pohjoisesta etelään (meiltä pois)
                    flux_s_adv_T = v_s * T[y, x] * edge_factor
                    flux_s_adv_q = v_s * q[y, x] * edge_factor

            # Yhdistetään kaikki vuot
            net_T_flux = flux_e_T + flux_w_T + flux_n_diff_T + flux_s_diff_T + flux_n_adv_T + flux_s_adv_T
            net_q_flux = flux_e_q + flux_w_q + flux_n_diff_q + flux_s_diff_q + flux_n_adv_q + flux_s_adv_q

            # Päivitys suhteellisella solualalla
            T_new[y, x] += (net_T_flux * dt) / cell_area
            q_new[y, x] += (net_q_flux * dt) / cell_area

            if q_new[y, x] < 0.0:
                q_new[y, x] = 0.0

    return T_new, q_new





import numpy as np
from numba import njit

@njit
def diffuse_temperature_and_humidity_scaled(
    T,
    q,
    material,
    K0,          # Skaalattu pieni arvo, esim. 0.65
    dt,          # Suhteellinen tai aikaskaalattu aika-askel
    k_ocean=1.0,
    k_land=0.5,
    k_sea_ice=0.25,
    k_snow_ice=0.25
):
    ny, nx = T.shape

    if K0 <= 0.0 or dt <= 0.0:
        return T.copy(), q.copy()

    dlat = np.pi / ny
    dlon = 2.0 * np.pi / nx

    # Skaalatussa muodossa materiaalien kertoimet toimivat suorina suhteellisina johtavuuksina
    kmat = np.array([k_ocean, k_land, k_sea_ice, k_snow_ice], dtype=np.float64)
    Gmat = K0 * kmat

    # Esialustetaan hilan johtavuusmatriisit suhteelliselle avaruudelle
    G_cell = np.empty((ny, nx), dtype=np.float64)
    for y in range(ny):
        for x in range(nx):
            m = material[y, x]
            if m < 0 or m > 3:
                m = 1
            G_cell[y, x] = Gmat[m]

    # Pinnan johtavuudet (Face conductances) skaalatussa hilassa
    Ge = np.zeros((ny, nx), dtype=np.float64)
    Gn = np.zeros((ny - 1, nx), dtype=np.float64)

    # Itä-länsi-suuntaiset virrat (East face)
    # Korjataan leveysasteen kosini-kutistuminen (metrisen etäisyyden sijaan skaalataan virtausta)
    for y in range(ny):
        phi = -0.5 * np.pi + (y + 0.5) * dlat
        cos_phi = np.cos(phi)
        
        # Estetään nollalla jakaminen navoilla geometrisen vakauden vuoksi
        if cos_phi < 1e-5:
            cos_phi = 1e-5
            
        # Itä-länsi-etäisyyskerroin suhteellisessa kulma-avaruudessa
        geom_factor_e = cos_phi * dlat / dlon

        for x in range(nx):
            xp = (x + 1) % nx
            g1, g2 = G_cell[y, x], G_cell[y, xp]
            if g1 > 0.0 and g2 > 0.0:
                gf = 2.0 * g1 * g2 / (g1 + g2)
                Ge[y, x] = gf * geom_factor_e

    # Pohjois-etelä-suuntaiset virrat (North face)
    for y in range(ny - 1):
        phi_edge = -0.5 * np.pi + (y + 1) * dlat
        # Pohjois-etelä-virtausskaala riippuu rajapinnan pituudesta suhteessa dlat-askeleeseen
        geom_factor_n = np.cos(phi_edge) * dlon / dlat

        for x in range(nx):
            g1, g2 = G_cell[y, x], G_cell[y + 1, x]
            if g1 > 0.0 and g2 > 0.0:
                gf = 2.0 * g1 * g2 / (g1 + g2)
                Gn[y, x] = gf * geom_factor_n

    # Aikaintegrointi suhteellisessa pinta-ala-avaruudessa
    T_new = T.copy()
    q_new = q.copy()

    for y in range(ny):
        phi_south = -0.5 * np.pi + y * dlat
        phi_north = -0.5 * np.pi + (y + 1) * dlat
        
        # Solun suhteellinen pinta-alakerroin (vastaa pallonpinnan osaa ilman r²-termiä)
        cell_area = dlon * (np.sin(phi_north) - np.sin(phi_south))
        if cell_area <= 0.0:
            continue

        for x in range(nx):
            xm = (x - 1) % nx
            xp = (x + 1) % nx

            # --- LÄMMÖN JA KOSTEUDEN VUOT ---
            # Skaalatussa muodossa virtaussuhteet ovat identtiset, koska dlat/dlon suhteet määräävät siirron
            
            # Itä-Länsi
            flux_e_T = Ge[y, x] * (T[y, xp] - T[y, x])
            flux_w_T = Ge[y, xm] * (T[y, xm] - T[y, x])
            flux_e_q = Ge[y, x] * (q[y, xp] - q[y, x])
            flux_w_q = Ge[y, xm] * (q[y, xm] - q[y, x])
            
            # Pohjois-Etelä
            flux_n_T = Gn[y, x] * (T[y + 1, x] - T[y, x]) if y < ny - 1 else 0.0
            flux_s_T = Gn[y - 1, x] * (T[y - 1, x] - T[y, x]) if y > 0 else 0.0
            
            flux_n_q = Gn[y, x] * (q[y + 1, x] - q[y, x]) if y < ny - 1 else 0.0
            flux_s_q = Gn[y - 1, x] * (q[y - 1, x] - q[y, x]) if y > 0 else 0.0

            # Nettoerot ja päivitykset jaettuna suhteellisella pinta-alalla
            net_T_flux = flux_e_T + flux_w_T + flux_n_T + flux_s_T
            T_new[y, x] += (net_T_flux * dt) / cell_area

            net_q_flux = flux_e_q + flux_w_q + flux_n_q + flux_s_q
            q_new[y, x] += (net_q_flux * dt) / cell_area
            
            # Numeerinen suoja kosteudelle
            if q_new[y, x] < 0.0:
                q_new[y, x] = 0.0

    return T_new, q_new





import numpy as np
from numba import njit

@njit
def diffuse_temperature_and_humidity(
    T,
    q,
    material,
    K0,
    dt, rplanet_m,
    k_ocean=2.0,
    k_land=1.0,
    k_sea_ice=0.5,
    k_snow_ice=0.5,
    h_eff=10.0  # Tehokas ilmoitus/vaikutuskerroksen korkeus metreinä
):
    ny, nx = T.shape

    if K0 <= 0.0 or dt <= 0.0:
        return T.copy(), q.copy()

    dlat = np.pi / ny
    dlon = 2.0 * np.pi / nx

    # Materiaalin lämpökapasiteetti pinta-alayksikköä kohti [J/m²/K]
    Cmat = np.array([
        1030.0 * 4184.0 * 10.0,  # valtameri
        2700.0 * 750.0 * 2.0,    # manner
        917.0 * 2100.0 * 3.0,    # merijää
        400.0 * 880.0 * 3.0,     # lumi / mannerjää
    ])

    kmat = np.array([k_ocean, k_land, k_sea_ice, k_snow_ice])

    # Lämmön energiankuljetuskerroin
    Gmat_heat = Cmat * K0 * kmat
    
    # Kosteuden diffuusiokerroin [m²/s]. 
    # Käytetään pohjana K0 * kmat fysikaalisen kytkennän vuoksi.
    # (Voit tarvittaessa kertoa tämän omalla skaalauskertoimella)
    Dmat_hum = K0 * kmat 

    phi_edge = np.empty(ny + 1)
    area = np.empty(ny)

    for y in range(ny + 1):
        phi_edge[y] = -0.5 * np.pi + y * dlat

    for y in range(ny):
        area[y] = (
            rplanet_m * rplanet_m * dlon
            * (np.sin(phi_edge[y + 1]) - np.sin(phi_edge[y]))
        )

    # Alustetaan tilavuudet ja johtavuudet
    H_heat = np.empty((ny, nx))
    V_hum = np.empty((ny, nx))      # Solun tehokas tilavuus kosteudelle [m³]
    G_heat = np.empty((ny, nx))
    G_hum = np.empty((ny, nx))       # Kosteuden diffuutiokyky [m³/s]

    for y in range(ny):
        for x in range(nx):
            m = material[y, x]
            if m < 0 or m > 3:
                m = 1

            H_heat[y, x] = Cmat[m] * area[y]
            V_hum[y, x] = area[y] * h_eff
            G_heat[y, x] = Gmat_heat[m]
            G_hum[y, x] = Dmat_hum[m] * h_eff

    # Pinnan johtavuudet
    Ge_heat = np.zeros((ny, nx))
    Gn_heat = np.zeros((ny - 1, nx))
    Ge_hum = np.zeros((ny, nx))
    Gn_hum = np.zeros((ny - 1, nx))

    # Itä-länsi-suuntaiset virrat (East face)
    for y in range(ny):
        phi = -0.5 * np.pi + (y + 0.5) * dlat
        distance = rplanet_m * np.cos(phi) * dlon
        face_area = rplanet_m * dlat

        for x in range(nx):
            xp = (x + 1) % nx
            
            # Lämmön kytkentä
            gh1, gh2 = G_heat[y, x], G_heat[y, xp]
            if gh1 > 0.0 and gh2 > 0.0:
                gf_h = 2.0 * gh1 * gh2 / (gh1 + gh2)
                Ge_heat[y, x] = gf_h * face_area / distance

            # Kosteuden kytkentä
            gd1, gd2 = G_hum[y, x], G_hum[y, xp]
            if gd1 > 0.0 and gd2 > 0.0:
                gf_d = 2.0 * gd1 * gd2 / (gd1 + gd2)
                # Kosteusvirran pinta-ala on face_area * h_eff
                Ge_hum[y, x] = gf_d * (face_area * h_eff) / distance

    # Pohjois-etelä-suuntaiset virrat (North face)
    distance_ns = rplanet_m * dlat

    for y in range(ny - 1):
        phi = phi_edge[y + 1]
        face_area_ns = rplanet_m * np.cos(phi) * dlon

        for x in range(nx):
            # Lämmön kytkentä
            gh1, gh2 = G_heat[y, x], G_heat[y + 1, x]
            if gh1 > 0.0 and gh2 > 0.0:
                gf_h = 2.0 * gh1 * gh2 / (gh1 + gh2)
                Gn_heat[y, x] = gf_h * face_area_ns / distance_ns

            # Kosteuden kytkentä
            gd1, gd2 = G_hum[y, x], G_hum[y + 1, x]
            if gd1 > 0.0 and gd2 > 0.0:
                gf_d = 2.0 * gd1 * gd2 / (gd1 + gd2)
                Gn_hum[y, x] = gf_d * (face_area_ns * h_eff) / distance_ns

    # Aikaintegrointi (Explicit Finite Volume)
    T_new = T.copy()
    q_new = q.copy()

    for y in range(ny):
        for x in range(nx):
            if H_heat[y, x] <= 0.0 or V_hum[y, x] <= 0.0:
                continue

            xm = (x - 1) % nx
            xp = (x + 1) % nx

            # --- LÄMMÖN VUOT ---
            flux_e_h = Ge_heat[y, x] * (T[y, xp] - T[y, x])
            flux_w_h = Ge_heat[y, xm] * (T[y, xm] - T[y, x])
            
            flux_n_h = Gn_heat[y, x] * (T[y + 1, x] - T[y, x]) if y < ny - 1 else 0.0
            flux_s_h = Gn_heat[y - 1, x] * (T[y - 1, x] - T[y, x]) if y > 0 else 0.0

            net_heat_flux = flux_e_h + flux_w_h + flux_n_h + flux_s_h
            T_new[y, x] += (net_heat_flux * dt) / H_heat[y, x]

            # --- KOSTEUDEN VUOT (g/s) ---
            flux_e_q = Ge_hum[y, x] * (q[y, xp] - q[y, x])
            flux_w_q = Ge_hum[y, xm] * (q[y, xm] - q[y, x])
            
            flux_n_q = Gn_hum[y, x] * (q[y + 1, x] - q[y, x]) if y < ny - 1 else 0.0
            flux_s_q = Gn_hum[y - 1, x] * (q[y - 1, x] - q[y, x]) if y > 0 else 0.0

            net_hum_flux = flux_e_q + flux_w_q + flux_n_q + flux_s_q
            net_hum_flux=flux_e_q + flux_w_q  ##warning test only
            # Päivitetään absoluuttinen kosteus solun tilavuudella
            q_new[y, x] += (net_hum_flux * dt) / V_hum[y, x]
            
            # Estetään numeerinen alivuoto (kosteus ei voi tippua negatiiviseksi)
            if q_new[y, x] < 0.0:
                q_new[y, x] = 0.0

    return T_new, q_new




import numpy as np
import math
from numba import njit


@njit(cache=True)
def diffuse_temperature_and_humidity_coupled_iso_koodi(
    T,
    humidity,
    material,
    K0,
    dt,
    rplanet_m,
    k_ocean=1.0,
    k_land=0.15,
    k_sea_ice=0.10,
    k_snow_ice=0.05,
    k_humidity_ocean=1.0,
    k_humidity_land=0.5,
    k_humidity_sea_ice=0.2,
    k_humidity_snow_ice=0.2,
    D_humidity=0.1,
    alpha_heat_moisture=0.1,
):
    ny, nx = T.shape

    if (
        humidity.shape != T.shape
        or material.shape != T.shape
        or ny < 2
        or nx < 2
        or K0 <= 0.0
        or dt <= 0.0
        or rplanet_m <= 0.0
    ):
        return T.copy(), humidity.copy()

    dlat = np.pi / ny
    dlon = 2.0 * np.pi / nx

    # Tilavuuslämpökapasiteetti * kerroksen paksuus
    Cmat = np.array([
        1030.0 * 4184.0 * 10.0,
        2700.0 * 750.0 * 2.0,
        917.0 * 2100.0 * 3.0,
        400.0 * 880.0 * 3.0,
    ])

    kT = np.array([
        k_ocean, k_land, k_sea_ice, k_snow_ice
    ])

    kH = np.array([
        k_humidity_ocean,
        k_humidity_land,
        k_humidity_sea_ice,
        k_humidity_snow_ice,
    ])

    area = np.empty(ny)

    for y in range(ny):
        phi0 = -0.5 * np.pi + y * dlat
        phi1 = phi0 + dlat
        area[y] = (
            rplanet_m**2
            * dlon
            * (np.sin(phi1) - np.sin(phi0))
        )

    HT = np.empty((ny, nx))
    GT = np.empty((ny, nx))
    GH = np.empty((ny, nx))

    for y in range(ny):
        for x in range(nx):
            m = material[y, x]
            if m < 0 or m > 3:
                m = 1

            HT[y, x] = Cmat[m] * area[y]

            # Lämpöjohtavuus
            GT[y, x] = Cmat[m] * K0 * kT[m]

            # Kosteuden kuljetuskerroin
            GH[y, x] = K0 * D_humidity * kH[m] * area[y]

    GeT = np.zeros((ny, nx))
    GnT = np.zeros((ny - 1, nx))
    GeH = np.zeros((ny, nx))
    GnH = np.zeros((ny - 1, nx))

    # Itä-länsisuuntaiset kasvot
    for y in range(ny):
        phi = -0.5 * np.pi + (y + 0.5) * dlat
        distance = rplanet_m * np.cos(phi) * dlon
        face_area = rplanet_m * dlat

        if distance > 1e-12:
            for x in range(nx):
                xp = (x + 1) % nx

                a = GT[y, x]
                b = GT[y, xp]
                if a > 0.0 and b > 0.0:
                    GeT[y, x] = (
                        2.0 * a * b / (a + b)
                        * face_area / distance
                    )

                a = GH[y, x]
                b = GH[y, xp]
                if a > 0.0 and b > 0.0:
                    GeH[y, x] = (
                        2.0 * a * b / (a + b)
                        * face_area / distance
                    )

    # Pohjois-eteläsuuntaiset kasvot
    distance_ns = rplanet_m * dlat

    for y in range(ny - 1):
        phi = -0.5 * np.pi + (y + 1) * dlat
        face_area = rplanet_m * np.cos(phi) * dlon

        for x in range(nx):
            a = GT[y, x]
            b = GT[y + 1, x]
            if a > 0.0 and b > 0.0:
                GnT[y, x] = (
                    2.0 * a * b / (a + b)
                    * face_area / distance_ns
                )

            a = GH[y, x]
            b = GH[y + 1, x]
            if a > 0.0 and b > 0.0:
                GnH[y, x] = (
                    2.0 * a * b / (a + b)
                    * face_area / distance_ns
                )

    # Eksplisiittisen Euler-menetelmän stabiilisuusraja.
    max_rate = 0.0

    for y in range(ny):
        for x in range(nx):
            xm = (x - 1) % nx

            rateT = (
                GeT[y, x] + GeT[y, xm]
            ) / HT[y, x]

            rateH = (
                GeH[y, x] + GeH[y, xm]
            ) / area[y]

            if y > 0:
                rateT += GnT[y - 1, x] / HT[y, x]
                rateH += GnH[y - 1, x] / area[y]

            if y < ny - 1:
                rateT += GnT[y, x] / HT[y, x]
                rateH += GnH[y, x] / area[y]

            max_rate = max(max_rate, rateT, rateH)

    nsub = max(
        1,
        int(math.ceil(dt * max_rate / 0.40))
    )
    sub_dt = dt / nsub

    Tout = T.copy()
    Hout = humidity.copy()

    newT = np.empty((ny, nx))
    Hdiff = np.empty((ny, nx))
    newH = np.empty((ny, nx))

    for step in range(nsub):

        # 1. Kosteuden diffuusio
        Hdiff[:, :] = Hout[:, :]

        for y in range(ny):
            for x in range(nx):
                xp = (x + 1) % nx

                q = (
                    sub_dt * GeH[y, x]
                    * (Hout[y, x] - Hout[y, xp])
                )

                Hdiff[y, x] -= q / area[y]
                Hdiff[y, xp] += q / area[y]

                if y < ny - 1:
                    q = (
                        sub_dt * GnH[y, x]
                        * (Hout[y, x] - Hout[y + 1, x])
                    )

                    Hdiff[y, x] -= q / area[y]
                    Hdiff[y + 1, x] += q / area[y + 1]

        # 2. Lämpötilagradientin aiheuttama kosteuden kuljetus
        newH[:, :] = Hdiff[:, :]

        if alpha_heat_moisture != 0.0:
            for y in range(ny):
                for x in range(nx):
                    xp = (x + 1) % nx

                    # Kosteusvirta itä-länsisuunnassa.
                    q = (
                        sub_dt * alpha_heat_moisture
                        * GeT[y, x]
                        * (Tout[y, x] - Tout[y, xp])
                        * 0.5
                        * (Hdiff[y, x] + Hdiff[y, xp])
                    )

                    # Rajoita virta, jotta lähdesolun kosteus
                    # ei muutu negatiiviseksi.
                    if q > 0.0:
                        q = min(q, max(0.0, newH[y, x]) * area[y])
                    else:
                        q = -min(
                            -q,
                            max(0.0, newH[y, xp]) * area[y]
                        )

                    newH[y, x] -= q / area[y]
                    newH[y, xp] += q / area[y]

                    if y < ny - 1:
                        q = (
                            sub_dt * alpha_heat_moisture
                            * GnT[y, x]
                            * (Tout[y, x] - Tout[y + 1, x])
                            * 0.5
                            * (Hdiff[y, x] + Hdiff[y + 1, x])
                        )

                        if q > 0.0:
                            q = min(
                                q,
                                max(0.0, newH[y, x]) * area[y]
                            )
                        else:
                            q = -min(
                                -q,
                                max(0.0, newH[y + 1, x])
                                * area[y + 1]
                            )

                        newH[y, x] -= q / area[y]
                        newH[y + 1, x] += q / area[y + 1]

        # 3. Lämpödiffuusio
        newT[:, :] = Tout[:, :]

        for y in range(ny):
            for x in range(nx):
                xp = (x + 1) % nx

                q = (
                    sub_dt * GeT[y, x]
                    * (Tout[y, x] - Tout[y, xp])
                )

                newT[y, x] -= q / HT[y, x]
                newT[y, xp] += q / HT[y, xp]

                if y < ny - 1:
                    q = (
                        sub_dt * GnT[y, x]
                        * (Tout[y, x] - Tout[y + 1, x])
                    )

                    newT[y, x] -= q / HT[y, x]
                    newT[y + 1, x] += q / HT[y + 1, x]

        Tout, newT = newT, Tout
        Hout, newH = newH, Hout

    return Tout, Hout









@njit
def diffuse_humidity(
    humidity,
    material,
    K0,
    dt,
    rplanet_m,
    k_ocean=1.0,
    k_land=0.5,
    k_sea_ice=0.2,
    k_snow_ice=0.2,
):
    ny, nx = humidity.shape

    if K0 <= 0.0 or dt <= 0.0:
        return humidity.copy()

    dlat = np.pi / ny
    dlon = 2.0 * np.pi / nx

    # Suhteelliset diffuusiokertoimet materiaalin mukaan.
    # Vain meri diffundoi oletusarvoisesti.
    kmat = np.array([
        k_ocean,
        k_land,
        k_sea_ice,
        k_snow_ice,
    ])

    # Diffuusion geometrinen kerroin.
    # K0:n yksiköt ja suuruus riippuvat humidity-muuttujasta.
    Gmat = K0 * kmat

    phi_edge = np.empty(ny + 1)
    area = np.empty(ny)

    for y in range(ny + 1):
        phi_edge[y] = -0.5 * np.pi + y * dlat

    for y in range(ny):
        area[y] = (
            rplanet_m**2 * dlon
            * (
                np.sin(phi_edge[y + 1])
                - np.sin(phi_edge[y])
            )
        )

    # Tallennetaan solukohtaiset kapasiteetit ja kertoimet.
    # Tässä oletetaan sama kosteuskapasiteetti kaikissa soluissa.
    H = area.copy()
    G = np.empty((ny, nx))

    for y in range(ny):
        for x in range(nx):
            m = material[y, x]

            if m < 0 or m > 3:
                m = 1

            G[y, x] = Gmat[m]

    # Itä-länsisuuntaiset rajapinnat.
    Ge = np.zeros((ny, nx))
    Gn = np.zeros((ny - 1, nx))

    for y in range(ny):
        phi = -0.5 * np.pi + (y + 0.5) * dlat
        distance = rplanet_m * np.cos(phi) * dlon
        face_area = rplanet_m * dlat

        for x in range(nx):
            xp = (x + 1) % nx

            g1 = G[y, x]
            g2 = G[y, xp]

            if g1 > 0.0 and g2 > 0.0 and distance > 0.0:
                gf = 2.0 * g1 * g2 / (g1 + g2)
                Ge[y, x] = gf * face_area / distance

    # Pohjois-eteläsuuntaiset rajapinnat.
    distance_ns = rplanet_m * dlat

    for y in range(ny - 1):
        phi = phi_edge[y + 1]
        face_area = rplanet_m * np.cos(phi) * dlon

        for x in range(nx):
            g1 = G[y, x]
            g2 = G[y + 1, x]

            if g1 > 0.0 and g2 > 0.0:
                gf = 2.0 * g1 * g2 / (g1 + g2)
                Gn[y, x] = gf * face_area / distance_ns

    # Suurin eksplisiittisen päivityksen nopeus.
    max_rate = 0.0

    for y in range(ny):
        for x in range(nx):
            xm = (x - 1) % nx

            rate = (Ge[y, x] + Ge[y, xm]) / H[y]

            if y > 0:
                rate += Gn[y - 1, x] / H[y]

            if y < ny - 1:
                rate += Gn[y, x] / H[y]

            if rate > max_rate:
                max_rate = rate

    nsub = max(1, int(math.ceil(dt * max_rate / 0.45)))
    sub_dt = dt / nsub

    out = humidity.copy()

    for step in range(nsub):
        new = out.copy()

        for y in range(ny):
            for x in range(nx):
                xp = (x + 1) % nx

                # Itään
                q = sub_dt * Ge[y, x] * (
                    out[y, x] - out[y, xp]
                )

                new[y, x] -= q / H[y]
                new[y, xp] += q / H[y]

                # Pohjoiseen
                if y < ny - 1:
                    q = sub_dt * Gn[y, x] * (
                        out[y, x] - out[y + 1, x]
                    )

                    new[y, x] -= q / H[y]
                    new[y + 1, x] += q / H[y + 1]

        out = new

    return out




@njit
def budyko_transport_temperature(temperature, planet_radius_m, b, dt):
    """Budyko-tyyppinen lämpötilan relaksaatiotransportti suoralle globaalille

    rasterille.

    Parameters
    ----------
    temperature : 2D float array
        Pintalämpötila [K], muoto (ny, nx), missä y on leveysaste ja x pituusaste.
    planet_radius_m : float
        Planeetan säde [m] (ei käytetä tässä laskennassa).
    b : float
        Relaksaation nopeuskerroin [1/s].
    dt : float
        Aika-askel [s].

    Returns
    -------
    out : 2D float array
        Päivitetty lämpötilakenttä [K].
    """
    ny, nx = temperature.shape
    out = np.empty_like(temperature)

    weighted_sum = 0.0
    weight_sum = 0.0

    # Lasketaan solukeskipisteiden väli radiaaneina
    # Koko planeetta kattaa 180 astetta eli pi radiaania y-suunnassa
    dlat = np.pi / ny

    # 1. Lasketaan pinta-alapainotettu keskilämpötila
    for i in range(ny):
        # Lasketaan solun keskipisteen leveysaste radiaaneina.
        # i=0 on pohjoisessa (pi/2) ja i=ny-1 on etelässä (-pi/2)
        lat_rad = (np.pi / 2.0) - (i + 0.5) * dlat

        # Pallopinnan pinta-alaelementti on verrannollinen kosiniin
        weight = np.cos(lat_rad)

        weight_sum += weight * nx

        lon_sum = 0.0
        for j in range(nx):
            lon_sum += temperature[i, j]

        weighted_sum += lon_sum * weight

    mean_temperature = weighted_sum / weight_sum

    # 2. Eksponentiaalinen relaksaatio kohti keskilämpötilaa
    decay = np.exp(-b * dt)
    rate=1 ## own rate
    for i in range(ny):
        for j in range(nx):
            out[i, j] = mean_temperature + (temperature[i, j] - mean_temperature) * decay*rate

    return out


    
@njit
def diffuse_temperature(
    T,
    material,
    K0,
    dt, rplanet_m,
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
            rplanet_m * rplanet_m * dlon
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
        distance = rplanet_m * np.cos(phi) * dlon
        face_area = rplanet_m * dlat

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
        face_area = rplanet_m * np.cos(phi) * dlon

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
    nsub=2
    #sub_dt = dt / nsub
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



@njit
def cloud_rain_step(
    temperature,
    humidity,
    cloud_water,
    dt_seconds,

    # Ilmakehä
    pressure_pa=101325.0,
    air_density=1.2,
    layer_height_m=1000.0,

    # Uudet rasteriparametrit
    pressure_raster=None,
    vertical_velocity=None,

    # Kondensaatio ja pilviveden haihtuminen
    condensation_time_s=300.0,
    evaporation_time_s=600.0,

    # Pilvien ja sateen muodostuminen
    cloud_water_threshold=0.00005,
    rain_water_threshold=0.0005,
    rain_conversion_time_s=1800.0,

    # Pilvivettä sateeksi muuttuva osuus
    precipitation_efficiency=0.8,

    # Pystynopeuden vaikutus
    vertical_velocity_scale=0.5,
):
    """
    Near-surface cloud and precipitation model.

    Inputs:
        temperature: K
        humidity: kg water vapour / kg air
        cloud_water: kg liquid water / kg air
        dt_seconds: timestep in seconds
        pressure_raster: local pressure in Pa, or None
        vertical_velocity: vertical velocity in m/s.
            Positive = ascending air, negative = descending air.

    Returns:
        RH (%)
        cloud_fraction (0...1)
        rain_mm_h (mm/hour)
        humidity_new (kg/kg)
        cloud_water_new (kg/kg)
    """

    ny, nx = temperature.shape

    rh = np.zeros((ny, nx))
    cloud_fraction = np.zeros((ny, nx))
    rain_mm_h = np.zeros((ny, nx))

    humidity_new = humidity.copy()
    cloud_water_new = cloud_water.copy()

    eps = 0.622

    if dt_seconds <= 0.0:
        return (
            rh, cloud_fraction, rain_mm_h,
            humidity_new, cloud_water_new
        )

    cond_factor_base = 1.0 - np.exp(
        -dt_seconds / max(1.0, condensation_time_s)
    )

    evap_factor_base = 1.0 - np.exp(
        -dt_seconds / max(1.0, evaporation_time_s)
    )

    rain_factor = 1.0 - np.exp(
        -dt_seconds / max(1.0, rain_conversion_time_s)
    )

    for y in range(ny):
        for x in range(nx):

            T = max(180.0, temperature[y, x])
            q = max(0.0, humidity[y, x])
            cw = max(0.0, cloud_water[y, x])

            # Paine rasterista tai vakioarvosta
            if pressure_raster is not None:
                p = max(100.0, pressure_raster[y, x])
            else:
                p = pressure_pa

            # Pystynopeus
            if vertical_velocity is not None:
                w = vertical_velocity[y, x]
            else:
                w = 0.0

            # Paineen kyllästyshöyrynpaine
            Tc = T - 273.15

            es = 610.94 * np.exp(
                17.625 * Tc / (Tc + 243.04)
            )

            es = min(es, p * 0.99)

            qsat = eps * es / (
                p - (1.0 - eps) * es
            )

            qsat = max(qsat, 1e-12)

            RH = q / qsat
            rh[y, x] = min(100.0, RH * 100.0)

            # Pystynopeuden vaikutus prosesseihin
            ascent = max(0.0, w)
            descent = max(0.0, -w)

            cond_factor = 1.0 - np.exp(
                -dt_seconds
                / max(
                    1.0,
                    condensation_time_s
                    / (1.0 + vertical_velocity_scale * ascent)
                )
            )

            evap_factor = 1.0 - np.exp(
                -dt_seconds
                / max(
                    1.0,
                    evaporation_time_s
                    / (1.0 + vertical_velocity_scale * descent)
                )
            )

            # 1. Kondensaatio
            if q > qsat:
                condensed = (q - qsat) * cond_factor
                q -= condensed
                cw += condensed

            # 2. Pilviveden haihtuminen
            elif q < qsat and cw > 0.0:
                possible_evap = (
                    (qsat - q) * evap_factor
                )
                evaporated = min(cw, possible_evap)
                q += evaporated
                cw -= evaporated

            RH = q / qsat

            # 3. Pilvipeite
            if cw <= cloud_water_threshold:
                cf = 0.0
            else:
                cf = (
                    (cw - cloud_water_threshold)
                    / max(
                        1e-12,
                        rain_water_threshold
                        - cloud_water_threshold
                    )
                )
                cf = min(1.0, max(0.0, cf))

            if RH > 0.90:
                rh_cloud = min(
                    1.0,
                    (RH - 0.90) / 0.10
                )
                cf = max(cf, rh_cloud * 0.25)

            cloud_fraction[y, x] = cf

            # 4. Sateen muodostuminen
            rain_mixing_ratio = 0.0

            if cw > rain_water_threshold:
                excess_cloud = cw - rain_water_threshold

                rain_mixing_ratio = (
                    excess_cloud
                    * rain_factor
                    * precipitation_efficiency
                )

                rain_mixing_ratio = min(
                    cw, rain_mixing_ratio
                )

                cw -= rain_mixing_ratio

            # 5. Sademäärä mm/h
            rain_mm_h[y, x] = (
                rain_mixing_ratio
                * air_density
                * layer_height_m
                * 3600.0
                / dt_seconds
            )

            humidity_new[y, x] = max(0.0, q)
            cloud_water_new[y, x] = max(0.0, cw)

    return (
        rh,
        cloud_fraction,
        rain_mm_h,
        humidity_new,
        cloud_water_new,
    )




#Yksi tärkeä huomio ennen ajoa

#Tässä versiossa upper_wind_strength=1.0 ja vertical_circulation_strength=10000.0 ovat nyt kaksi tärkeintä säätöparametria.

#Jos ylävirta on liian heikkoa:

#upper_wind_strength=2.0

#Jos pystysuuntainen kierto on liian heikkoa:

#vertical_circulation_strength=20000.0






@njit
def evaporation_step(
    humidity,
    temperature,
    landmask,
    soil_moisture,
    dt_seconds,

    # Haihtumisen säätöparametrit
    ocean_evaporation_rate=2.0e-5/100,
    land_evaporation_rate=1.0e-5/100,
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
def enforce_polar_humidity(
    humidity,
    polar_humidity=0.0002,
    polar_rows=1,
):
    ny, nx = humidity.shape

    # Pohjoisnapa
    for y in range(polar_rows):
        for x in range(nx):
            humidity[y, x] = polar_humidity

    # Etelänapa
    for y in range(ny - polar_rows, ny):
        for x in range(nx):
            humidity[y, x] = polar_humidity

    return humidity
 
 



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
    nlon = nx - 1  # Viimeinen sarake on periodisen sauman kopio.

    dlat = np.pi / (ny - 1)
    dlon = 2.0 * np.pi / nlon

    polar_latitude_limit = 80.0
    polar_humidity = 0.0002

    # Tavoiteltu CFL-raja jokaiselle sisäiselle aliaskelmalle.
    cfl_target = 0.8

    # --------------------------------------------------
    # 1. Etsi suurin paikallinen Courant-luku
    # --------------------------------------------------
    max_courant = 0.0

    for y in range(1, ny - 1):
        latitude = 0.5 * np.pi - y * dlat
        cos_lat = abs(np.cos(latitude))

        for x in range(nlon):
            r = planet_radius_m + dem[y, x]

            if r <= 0.0:
                continue

            dx = r * cos_lat * dlon
            dy = r * dlat

            if dx <= 0.0 or dy <= 0.0:
                continue

            courant = dt_seconds * (
                abs(u[y, x]) / dx
                + abs(v[y, x]) / dy
            )

            if courant > max_courant:
                max_courant = courant

    # --------------------------------------------------
    # 2. Valitse aliaskelten määrä automaattisesti
    # --------------------------------------------------
    n_substeps = max(
        1,
        int(np.ceil(max_courant / cfl_target))
    )

    dt = dt_seconds / n_substeps

    temperature_old = temperature.copy()
    humidity_old = humidity.copy()

    temperature_new = temperature.copy()
    humidity_new = humidity.copy()

    # --------------------------------------------------
    # 3. Upwind-advektio aliaskel kerrallaan
    # --------------------------------------------------
    for step in range(n_substeps):

        # Jokainen uusi kenttä lasketaan edellisestä
        # kokonaisesta aika-askeleesta.
        temperature_new[:, :] = temperature_old
        humidity_new[:, :] = humidity_old

        for y in range(1, ny - 1):
            latitude = 0.5 * np.pi - y * dlat
            cos_lat = abs(np.cos(latitude))

            is_polar = (
                abs(latitude)
                >= np.deg2rad(polar_latitude_limit)
            )

            for x in range(nlon):

                xm = x - 1
                xp = x + 1

                if xm < 0:
                    xm = nlon - 1

                if xp >= nlon:
                    xp = 0

                r = planet_radius_m + dem[y, x]

                if r <= 0.0:
                    continue

                dx = r * cos_lat * dlon
                dy = r * dlat

                if dx <= 0.0 or dy <= 0.0:
                    continue

                uu = u[y, x]
                vv = v[y, x]

                T = temperature_old
                q = humidity_old

                # --------------------------------------
                # Lämpötila: upwind-gradientit
                # --------------------------------------
                if uu >= 0.0:
                    dTdx = (T[y, x] - T[y, xm]) / dx
                else:
                    dTdx = (T[y, xp] - T[y, x]) / dx

                # Oletus: v > 0 tarkoittaa pohjoiseen.
                if vv >= 0.0:
                    dTdy = (T[y, x] - T[y + 1, x]) / dy
                else:
                    dTdy = (T[y - 1, x] - T[y, x]) / dy

                temperature_new[y, x] = T[y, x] - dt * (
                    uu * dTdx + vv * dTdy
                )

                # --------------------------------------
                # Kosteus: upwind-gradientit
                # --------------------------------------
                if uu >= 0.0:
                    dqdx = (q[y, x] - q[y, xm]) / dx
                else:
                    dqdx = (q[y, xp] - q[y, x]) / dx

                if vv >= 0.0:
                    dqdy = (q[y, x] - q[y + 1, x]) / dy
                else:
                    dqdy = (q[y - 1, x] - q[y, x]) / dy

                q_new = q[y, x] - dt * (
                    uu * dqdx + vv * dqdy
                )

                # Kosteus ei voi olla negatiivinen.
                if q_new < 0.0:
                    q_new = 0.0

                # Alkuperäisen mallin napakosteusehto.
                if is_polar:
                    q_new = polar_humidity

                humidity_new[y, x] = q_new

        # Periodinen pituusasteen sauma.
        for y in range(ny):
            temperature_new[y, nx - 1] = temperature_new[y, 0]
            humidity_new[y, nx - 1] = humidity_new[y, 0]

        # Seuraava aliaskel käyttää juuri laskettua kenttää.
        temperature_old[:, :] = temperature_new
        humidity_old[:, :] = humidity_new

    return temperature_new, humidity_new



#surface_mass_weight = 0.7
#upper_mass_weight = 0.3
#pressure_mass_coupling = 1.0
#advection_cfl = 0.2



@njit
def atmosphere_step_v10(
    temperature,
    humidity,
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

    # Upper atmosphere
    upper_drag_timescale=1728.0,
    upper_wind_strength=0.003,

    # Vertical circulation diagnostic
    vertical_circulation_strength=100.0/10,

    # Surface drag
    surface_drag_timescale_land=2160.0,
    surface_drag_timescale_ocean=8640.0,

    # Column mass continuity
    surface_mass_weight=0.7/10,
    upper_mass_weight=0.3/10,
    pressure_mass_coupling=1.0/10,

    # Numerical controls
    equatorial_fmin=2.0e-6,
    max_wind_speed=100.0,
    min_pressure=100.0,
    max_pressure_fraction_step=0.005,
    advection_cfl=0.25,
):
    """
    V10: two-layer atmospheric circulation model.

    Upper wind:
        temperature gradients -> upper momentum balance

    Vertical circulation:
        upper spherical divergence -> diagnostic vertical velocity

    Surface pressure:
        weighted surface/upper horizontal mass-flux divergence

    Surface wind:
        time-dependent momentum
        pressure gradient
        Coriolis
        drag
        spherical horizontal advection

    Grid:
        y=0 north pole
        y=ny-1 south pole
        last longitude column duplicates column zero

    Important:
        This is a simplified two-layer closure.
        It is not a fully conservative 3D primitive-equation model.
    """
    rotation_period_days=rotation_period_days
    ny, nx = temperature.shape
    nlon = nx - 1

    speed = np.zeros((ny, nx), dtype=np.float64)

    if (
        ny < 5
        or nx < 5
        or dt_seconds <= 0.0
        or planet_radius_m <= 0.0
        or rotation_period_days <= 0.0
    ):
        for y in range(ny):
            for x in range(nx):
                speed[y, x] = np.sqrt(
                    u[y, x]**2 + v[y, x]**2
                )

        return (
            pressure_raster, humidity, temperature,
            u, v, speed,
            u_upper, v_upper, vertical_velocity
        )

    omega = (
        2.0 * np.pi
        / (rotation_period_days * 86400.0)
    )

    dlat = np.pi / (ny - 1)
    dlon = 2.0 * np.pi / nlon

    # Normalize the layer weights.
    weight_sum = surface_mass_weight + upper_mass_weight

    if weight_sum <= 0.0:
        surface_weight = 1.0
        upper_weight = 0.0
    else:
        surface_weight = surface_mass_weight / weight_sum
        upper_weight = upper_mass_weight / weight_sum

    # Work arrays.
    div_surface = np.zeros(
        (ny, nx), dtype=np.float64
    )
    div_upper = np.zeros(
        (ny, nx), dtype=np.float64
    )
    pressure_tendency = np.zeros(
        (ny, nx), dtype=np.float64
    )

    # ========================================================
    # STEP 1: UPPER WIND FROM TEMPERATURE GRADIENTS
    # ========================================================

    for x in range(nx):
        u_upper[0, x] = 0.0
        v_upper[0, x] = 0.0
        u_upper[ny - 1, x] = 0.0
        v_upper[ny - 1, x] = 0.0

        vertical_velocity[0, x] = 0.0
        vertical_velocity[ny - 1, x] = 0.0

    for y in range(1, ny - 1):

        lat = 0.5 * np.pi - y * dlat
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)

        f = 2.0 * omega * sin_lat

        if abs(f) < equatorial_fmin:
            if f >= 0.0:
                f_eff = equatorial_fmin
            else:
                f_eff = -equatorial_fmin
        else:
            f_eff = f

        for x in range(nlon):

            xm = (x - 1) % nlon
            xp = (x + 1) % nlon

            r = planet_radius_m + dem[y, x]

            if r <= 0.0:
                u_upper[y, x] = 0.0
                v_upper[y, x] = 0.0
                continue

            dx = max(
                r * max(abs(cos_lat), 1.0e-5) * dlon,
                1.0
            )
            dy = max(r * dlat, 1.0)

            dTdx = (
                temperature[y, xp]
                - temperature[y, xm]
            ) / (2.0 * dx)

            dTdy = (
                temperature[y - 1, x]
                - temperature[y + 1, x]
            ) / (2.0 * dy)

            ax = -R_AIR * upper_wind_strength * dTdx
            ay = -R_AIR * upper_wind_strength * dTdy

            tau = max(upper_drag_timescale, 1.0)
            ft = f_eff * tau
            den = 1.0 + ft * ft

            uu = tau * (ax + ft * ay) / den
            vv = tau * (ay - ft * ax) / den

            s = np.sqrt(uu * uu + vv * vv)

            if s > max_wind_speed:
                factor = max_wind_speed / s
                uu *= factor
                vv *= factor

            u_upper[y, x] = uu
            v_upper[y, x] = vv

        u_upper[y, nx - 1] = u_upper[y, 0]
        v_upper[y, nx - 1] = v_upper[y, 0]

    # ========================================================
    # STEP 2: SPHERICAL DIVERGENCE OF BOTH WIND FIELDS
    #
    # div(U) =
    #   1/(r*cos(lat))*du/dlon
    # + 1/(r*cos(lat))*d(v*cos(lat))/dlat
    # ========================================================

    for y in range(1, ny - 1):

        lat = 0.5 * np.pi - y * dlat
        cos_lat = np.cos(lat)
        cos_safe = max(abs(cos_lat), 1.0e-5)

        lat_n = lat + dlat
        lat_s = lat - dlat

        for x in range(nlon):

            xm = (x - 1) % nlon
            xp = (x + 1) % nlon

            r = planet_radius_m + dem[y, x]

            if r <= 0.0:
                continue

            du_s_dlon = (
                u[y, xp] - u[y, xm]
            ) / (2.0 * dlon)

            d_vcos_s = (
                v[y - 1, x] * np.cos(lat_n)
                - v[y + 1, x] * np.cos(lat_s)
            ) / (2.0 * dlat)

            div_surface[y, x] = (
                du_s_dlon + d_vcos_s
            ) / (r * cos_safe)

            du_u_dlon = (
                u_upper[y, xp] - u_upper[y, xm]
            ) / (2.0 * dlon)

            d_vcos_u = (
                v_upper[y - 1, x] * np.cos(lat_n)
                - v_upper[y + 1, x] * np.cos(lat_s)
            ) / (2.0 * dlat)

            div_upper[y, x] = (
                du_u_dlon + d_vcos_u
            ) / (r * cos_safe)

            # Positive vertical velocity denotes rising motion.
            # This is a diagnostic proxy, not a solved 3D velocity.
            vertical_velocity[y, x] = (
                vertical_circulation_strength
                * div_upper[y, x]
            )

        div_surface[y, nx - 1] = div_surface[y, 0]
        div_upper[y, nx - 1] = div_upper[y, 0]

        vertical_velocity[y, nx - 1] = (
            vertical_velocity[y, 0]
        )

    # ========================================================
    # STEP 3: COLUMN PRESSURE TENDENCY
    #
    # Pressure rises with column convergence.
    # Pressure falls with column divergence.
    #
    # The layer weights approximate the relative contribution
    # of each layer to the column horizontal mass transport.
    # ========================================================

    for y in range(ny):
        for x in range(nx):
            pressure_tendency[y, x] = 0.0

    for y in range(1, ny - 1):
        for x in range(nlon):

            column_divergence = (
                surface_weight * div_surface[y, x]
                + upper_weight * div_upper[y, x]
            )

            pressure_tendency[y, x] = (
                -pressure_mass_coupling
                * pressure_raster[y, x]
                * column_divergence
            )

    # Remove global mean pressure tendency using spherical
    # area weights. This prevents an artificial global drift.
    weighted_sum = 0.0
    weight_sum = 0.0

    for y in range(1, ny - 1):

        lat = 0.5 * np.pi - y * dlat
        area_weight = abs(np.cos(lat))

        for x in range(nlon):
            weighted_sum += (
                pressure_tendency[y, x] * area_weight
            )
            weight_sum += area_weight

    mean_tendency = 0.0

    if weight_sum > 0.0:
        mean_tendency = weighted_sum / weight_sum

    for y in range(1, ny - 1):
        for x in range(nlon):
            pressure_tendency[y, x] -= mean_tendency

    # ========================================================
    # STEP 4: UPDATE SURFACE PRESSURE
    # ========================================================

    for y in range(1, ny - 1):

        for x in range(nlon):

            p = pressure_raster[y, x]

            dp = (
                pressure_tendency[y, x]
                * dt_seconds
            )

            max_dp = (
                max_pressure_fraction_step
                * max(p, min_pressure)
            )

            if dp > max_dp:
                dp = max_dp
            elif dp < -max_dp:
                dp = -max_dp

            pressure_raster[y, x] = max(
                p + dp, min_pressure
            )

        pressure_raster[y, nx - 1] = (
            pressure_raster[y, 0]
        )

    # ========================================================
    # STEP 5: SURFACE MOMENTUM
    #
    # Time-dependent momentum with:
    #   pressure gradient
    #   spherical advection
    #   Coriolis
    #   drag
    # ========================================================

    u_old = u.copy()
    v_old = v.copy()

    min_spacing = 1.0e30
    max_initial_speed = 1.0

    for y in range(1, ny - 1):

        lat = 0.5 * np.pi - y * dlat
        cos_safe = max(abs(np.cos(lat)), 1.0e-5)

        for x in range(nlon):

            r = planet_radius_m + dem[y, x]

            if r <= 0.0:
                continue

            dx = max(r * cos_safe * dlon, 1.0)
            dy = max(r * dlat, 1.0)

            min_spacing = min(
                min_spacing, dx, dy
            )

            s = np.sqrt(
                u_old[y, x]**2
                + v_old[y, x]**2
            )

            max_initial_speed = max(
                max_initial_speed, s
            )

    cfl = min(max(advection_cfl, 0.05), 0.8)

    sub_dt_limit = (
        cfl * min_spacing / max_initial_speed
    )

    sub_dt_limit = max(sub_dt_limit, 1.0e-6)

    nsub = int(np.ceil(dt_seconds / sub_dt_limit))
    nsub = max(nsub, 1)

    # A capped substep count is a warning condition:
    # reduce the main timestep if this limit is reached.
    if nsub > 10000:
        nsub = 10000

    sub_dt = dt_seconds / nsub

    u_next = u.copy()
    v_next = v.copy()

    for substep in range(nsub):

        u_old[:, :] = u[:, :]
        v_old[:, :] = v[:, :]

        for y in range(1, ny - 1):

            lat = 0.5 * np.pi - y * dlat
            sin_lat = np.sin(lat)
            cos_lat = np.cos(lat)

            cos_safe = max(abs(cos_lat), 1.0e-3)

            tan_lat = sin_lat / cos_lat
            tan_lat = min(max(tan_lat, -100.0), 100.0)

            f = 2.0 * omega * sin_lat

            for x in range(nlon):

                xm = (x - 1) % nlon
                xp = (x + 1) % nlon

                r = planet_radius_m + dem[y, x]

                if r <= 0.0:
                    continue

                dx = max(r * cos_safe * dlon, 1.0)
                dy = max(r * dlat, 1.0)

                uu = u_old[y, x]
                vv = v_old[y, x]

                # Pressure-gradient acceleration.
                dpdx = (
                    pressure_raster[y, xp]
                    - pressure_raster[y, xm]
                ) / (2.0 * dx)

                dpdy = (
                    pressure_raster[y - 1, x]
                    - pressure_raster[y + 1, x]
                ) / (2.0 * dy)

                T = max(temperature[y, x], 1.0)

                rho = (
                    pressure_raster[y, x]
                    / (R_AIR * T)
                )

                if rho < 1.0e-8:
                    continue

                ax = -dpdx / rho
                ay = -dpdy / rho

                # Upwind longitude derivatives.
                if uu >= 0.0:

                    du_dlon = (
                        u_old[y, x] - u_old[y, xm]
                    ) / dlon

                    dv_dlon = (
                        v_old[y, x] - v_old[y, xm]
                    ) / dlon

                else:

                    du_dlon = (
                        u_old[y, xp] - u_old[y, x]
                    ) / dlon

                    dv_dlon = (
                        v_old[y, xp] - v_old[y, x]
                    ) / dlon

                # Upwind latitude derivatives.
                if vv >= 0.0:

                    du_dlat = (
                        u_old[y - 1, x] - u_old[y, x]
                    ) / dlat

                    dv_dlat = (
                        v_old[y - 1, x] - v_old[y, x]
                    ) / dlat

                else:

                    du_dlat = (
                        u_old[y, x] - u_old[y + 1, x]
                    ) / dlat

                    dv_dlat = (
                        v_old[y, x] - v_old[y + 1, x]
                    ) / dlat

                adv_u = (
                    uu * du_dlon / (r * cos_safe)
                    + vv * du_dlat / r
                    - uu * vv * tan_lat / r
                )

                adv_v = (
                    uu * dv_dlon / (r * cos_safe)
                    + vv * dv_dlat / r
                    + uu * uu * tan_lat / r
                )

                if landmask[y, x] > 0.5:
                    tau = surface_drag_timescale_land
                else:
                    tau = surface_drag_timescale_ocean

                tau = max(tau, 1.0)

                # Explicit pressure and advection forcing.
                rhs_u = uu + sub_dt * (ax - adv_u)
                rhs_v = vv + sub_dt * (ay - adv_v)

                # Semi-implicit Coriolis and drag.
                a = 1.0 + sub_dt / tau
                b = sub_dt * f
                den = a * a + b * b

                un = (a * rhs_u + b * rhs_v) / den
                vn = (a * rhs_v - b * rhs_u) / den

                s = np.sqrt(un * un + vn * vn)

                if s > max_wind_speed:
                    factor = max_wind_speed / s
                    un *= factor
                    vn *= factor

                u_next[y, x] = un
                v_next[y, x] = vn

            u_next[y, nx - 1] = u_next[y, 0]
            v_next[y, nx - 1] = v_next[y, 0]

        for y in range(1, ny - 1):
            for x in range(nx):
                u[y, x] = u_next[y, x]
                v[y, x] = v_next[y, x]

    # ========================================================
    # STEP 6: WIND SPEED
    # ========================================================

    for y in range(ny):

        for x in range(nlon):
            speed[y, x] = np.sqrt(
                u[y, x]**2 + v[y, x]**2
            )

        u[y, nx - 1] = u[y, 0]
        v[y, nx - 1] = v[y, 0]
        speed[y, nx - 1] = speed[y, 0]

    return (
        pressure_raster,
        humidity,
        temperature,
        u,
        v,
        speed,
        u_upper,
        v_upper,
        vertical_velocity
    )


def bilinear_global(raster, lon, lat):
    """
    Bilineaarinen interpolointi globaalista rasterista.

    raster: (ny, nx) tai (12, ny, nx)
    lon: pituusaste [-180, 180]
    lat: leveysaste [-90, 90]

    Oletus:
      - rasterin ensimmäinen rivi on pohjoisessa
      - rasterin pikselit kattavat alueen [-180, 180, -90, 90]
      - rasterin arvot sijaitsevat pikselien keskipisteissä
    """
    ny, nx = raster.shape[-2:]

    # Pituusaste muunnetaan jatkuvaksi pikseli-indeksiksi.
    # -180 ja +180 ovat sama maantieteellinen pituusaste.
    x = ((lon + 180) % 360) / 360 * nx - 0.5

    # Leveysaste: rasterin ensimmäinen rivi on pohjoisessa.
    y = (90 - lat) / 180 * ny - 0.5
    y = np.clip(y, 0, ny - 1)

    x0_raw = int(np.floor(x))
    y0 = int(np.floor(y))

    x1_raw = x0_raw + 1
    y1 = min(y0 + 1, ny - 1)

    # Pituusasteen indeksi kiertyy globaalisti.
    x0 = x0_raw % nx
    x1 = x1_raw % nx

    # Interpolointipainot
    dx = x - x0_raw
    dy = y - y0

    # Neljä ympäröivää rasteriarvoa
    q00 = raster[..., y0, x0]
    q10 = raster[..., y0, x1]
    q01 = raster[..., y1, x0]
    q11 = raster[..., y1, x1]

    return (
        q00 * (1 - dx) * (1 - dy)
        + q10 * dx * (1 - dy)
        + q01 * (1 - dx) * dy
        + q11 * dx * dy
    )



def calculate_biotemperature(monthly_temps):
    """
    Laskee vuotuisen biolämpötilan (T_bio).
    monthly_temps: 3D numpy-array (kuukaudet, leveysaste, pituusaste) tai 1D-vektori kuukausikeskilämpötiloista.
    """
    # Holdridgen sääntö: Kaikki alle 0°C menevät kuukaudet nollataan kasvuaktiivisuuden osalta
    clipped_temps = np.clip(monthly_temps, a_min=0, a_max=30) 
    return np.mean(clipped_temps, axis=0)




    

def plot_temperature_map(downscaled_temperature_C, accurate_dem):
    """
    Piirtää planeetan keskilämpötilakartan rantaviivoineen ja lämpötilakäyrineen.
    """
    plt.figure(figsize=(12, 5))

    # Merimaski: DEM <= 0
    sea_mask = accurate_dem <= 0
    data_min = np.nanmin(downscaled_temperature_C)
    data_max = np.nanmax(downscaled_temperature_C)

    max_abs = max(abs(data_min), abs(data_max))
    limit = np.ceil(max_abs / 10) * 10

    # Lämpötilakuva
    im = plt.imshow(
        downscaled_temperature_C,
        interpolation="bilinear",
        origin="lower", 
        vmin=-limit, 
        vmax=limit,
        extent=[-180, 180, -90, 90],
        aspect="auto",
        cmap="RdYlBu_r"
    )

    # Piirretään meri mustaksi / tummansiniseksi
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
    levels_list = [-150, -120, -100, -80, -50, -40, -30, -25, -20, -15, -10, -5,
                   0, 5, 10, 15, 20, 25, 30, 40, 50, 80, 100, 120, 150, 250]
    
    cont1 = plt.contour(
        downscaled_temperature_C,
        levels=levels_list,
        linewidths=1,
        colors="black",
        alpha=0.5,
        origin="lower",
        extent=[-180, 180, -90, 90]
    )

    plt.clabel(cont1, inline=1, fmt='%3.1f', fontsize=10, colors=["#3f0000"])

    plt.colorbar(im, label="Mean temperature (°C)")
    plt.xlabel("Longitude (°)")
    plt.ylabel("Latitude (°)")
    plt.title("Mean temperature of planet, °C")
    
    plt.tight_layout()
    plt.show()
def plot_precipitation_map(precipitation_data, accurate_dem, vmin=0, vmax=3500):
    """
    Piirtää planeetan sademääräkartan käyttäen YlGnBu-värikarttaa.
    """
    plt.figure(figsize=(12, 5))

    # Merimaski: DEM <= 0
    sea_mask = accurate_dem <= 0

    # Sademääräkuva (pohjaväritys)
    im = plt.imshow(
        precipitation_data,
        interpolation="bilinear",
        origin="lower", 
        vmin=vmin, 
        vmax=vmax,
        extent=[-180, 180, -90, 90],
        aspect="auto",
        cmap="YlGnBu"
    )

    # Piirretään meri tummansiniseksi (kuten lämpötilakuvassa)
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

    plt.colorbar(im, label="Precipitation (mm)")
    plt.xlabel("Longitude (°)")
    plt.ylabel("Latitude (°)")
    plt.title("Precipitation of planet, mm")
    
    plt.tight_layout()
    plt.show()






# ID, nimi, väri
HLZ = {
    0: ("Ei dataa / meri", "#a6cee3"),
    1: ("Polar desert", "#f7f7f7"),
    2: ("Subpolar dry tundra", "#d9ef8b"),
    3: ("Subpolar moist tundra", "#a6d96a"),
    4: ("Subpolar wet tundra", "#66bd63"),
    5: ("Subpolar rain tundra", "#1a9850"),
    6: ("Boreal desert", "#fff7bc"),
    7: ("Boreal dry scrub", "#d9e6a3"),
    8: ("Boreal moist forest", "#78c679"),
    9: ("Boreal wet forest", "#31a354"),
    10: ("Boreal rain forest", "#006837"),
    11: ("Cool temperate desert", "#fff7bc"),
    12: ("Cool temperate desert scrub", "#fee391"),
    13: ("Cool temperate steppe", "#fec44f"),
    14: ("Cool temperate moist forest", "#a1d99b"),
    15: ("Cool temperate wet forest", "#41ab5d"),
    16: ("Cool temperate rain forest", "#006d2c"),
    17: ("Warm temperate desert", "#fff7bc"),
    18: ("Warm temperate desert scrub", "#fdd49e"),
    19: ("Warm temperate thorn steppe", "#fc8d59"),
    20: ("Warm temperate dry forest", "#addd8e"),
    21: ("Warm temperate moist forest", "#78c679"),
    22: ("Warm temperate wet forest", "#31a354"),
    23: ("Warm temperate rain forest", "#006837"),
    24: ("Subtropical desert", "#ffffcc"),
    25: ("Subtropical desert scrub", "#fed976"),
    26: ("Subtropical thorn woodland", "#fd8d3c"),
    27: ("Subtropical dry forest", "#c2e699"),
    28: ("Subtropical moist forest", "#78c679"),
    29: ("Subtropical wet forest", "#31a354"),
    30: ("Subtropical rain forest", "#006837"),
    31: ("Tropical desert", "#ffffcc"),
    32: ("Tropical desert scrub", "#fed976"),
    33: ("Tropical thorn woodland", "#fd8d3c"),
    34: ("Tropical very dry forest", "#f03b20"),
    35: ("Tropical dry forest", "#c2e699"),
    36: ("Tropical moist forest", "#78c679"),
    37: ("Tropical wet forest", "#31a354"),
    38: ("Tropical rain forest", "#004529"),
}


def classify_holdridge_vectorized(t_bio, p_annual):
    """
    Holdridge-tyyppinen 38 luokan luokitus.
    t_bio: biolämpötila °C, mikä tahansa NumPy-taulukko
    p_annual: vuosisade mm, samanmuotoinen taulukko

    Palauttaa ID-taulukon (0 = puuttuva/virheellinen data).
    """
    t, p = np.broadcast_arrays(
        np.asarray(t_bio, dtype=float),
        np.asarray(p_annual, dtype=float)
    )

    out = np.zeros(t.shape, dtype=np.int16)
    valid = np.isfinite(t) & np.isfinite(p) & (p >= 0) & (t >= 0)

    # Jokainen rivi: lämpötilarajat, sadebins, ID:t
    zones = [
        (0, 1.5, [0, np.inf], [1]),
        (1.5, 3, [0, 125, 250, 500, np.inf], [2, 3, 4, 5]),
        (3, 6, [0, 125, 250, 500, 1000, np.inf],
         [6, 7, 8, 9, 10]),
        (6, 12, [0, 125, 250, 500, 1000, 2000, np.inf],
         [11, 12, 13, 14, 15, 16]),
        (12, 18, [0, 125, 250, 500, 1000, 2000, 4000, np.inf],
         [17, 18, 19, 20, 21, 22, 23]),
        (18, 24, [0, 125, 250, 500, 1000, 2000, 4000, np.inf],
         [24, 25, 26, 27, 28, 29, 30]),
        (24, np.inf, [0, 125, 250, 500, 1000, 2000, 4000, 8000, np.inf],
         [31, 32, 33, 34, 35, 36, 37, 38]),
    ]

    for tmin, tmax, edges, ids in zones:
        tm = valid & (t >= tmin) & (t < tmax)

        for j, zone_id in enumerate(ids):
            pm = (p >= edges[j]) & (p < edges[j + 1])
            out[tm & pm] = zone_id

    return out


def plot_climate_diagram(lon, lat, masl, temp_monthly, prec_monthly):
    months = [
        "Tammi", "Helmi", "Maalis", "Huhti",
        "Touko", "Kesä", "Heinä", "Elo",
        "Syys", "Loka", "Marras", "Joulu"
    ]

    fig, ax1 = plt.subplots(figsize=(11, 5))

    # Kuukausittainen sadanta
    ax1.bar(
        months,
        prec_monthly,
        color="cornflowerblue",
        alpha=0.75,
        label="Sadanta"
    )
    ax1.set_xlabel("Kuukausi")
    ax1.set_ylabel("Sadanta (mm/kk)", color="royalblue")
    ax1.tick_params(axis="y", labelcolor="royalblue")
    ax1.grid(axis="y", alpha=0.2)

    # Kuukausittainen keskilämpötila
    ax2 = ax1.twinx()
    ax2.plot(
        months,
        temp_monthly,
        color="firebrick",
        marker="o",
        linewidth=2,
        label="Keskilämpötila"
    )
    ax2.set_ylabel("Keskilämpötila (°C)", color="firebrick")
    ax2.tick_params(axis="y", labelcolor="firebrick")
    ax2.axhline(0, color="gray", linewidth=0.8, alpha=0.6)

    ax1.set_title(
        f"Ilmastodiagrammi: {lon:.2f}°, {lat:.2f}°\n"
        f"Korkeus {masl:.0f} m merenpinnasta"
    )

    fig.tight_layout()
    plt.show()



def plot_holdridge_map(
    holdridge_ids,
    landmask,
    t_bio,
    p_annual,
    title="Holdridge life zones"
):
    #ocean_mask = np.isnan(landmask).astype(float)
    ocean_mask=np.copy(landmask)
    ocean_mask=np.where(ocean_mask==0,1,np.nan)
    cmap1 = ListedColormap(["#001a33"])  # Tumma sininen
    # Värit ja luokkanimet HLZ-sanakirjasta
    ids = sorted(HLZ)
    cmap = mcolors.ListedColormap([HLZ[i][1] for i in ids])
    norm = mcolors.BoundaryNorm(
        np.arange(-0.5, len(ids) + 0.5, 1),
        cmap.N
    )

    # Maamaski ja rasterin käsittely
    land = np.isfinite(landmask) & (landmask == 1)
    mapped = np.where(
        land & np.isfinite(holdridge_ids),
        holdridge_ids, np.nan
    )

    fig = plt.figure(figsize=(19, 9))

    # 1. Maailmankartta ilman lats/lons-vektoreita
    ax = fig.add_subplot(
        1, 2, 1, projection=ccrs.Robinson()
    )

    ax.imshow(
        mapped,
        extent=[-180, 180, -90, 90],
        origin="upper",
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        interpolation="nearest"
    )

    ax.set_global()
    ax.set_title(title, fontsize=13)

    # 2. P/PET-luokitteludiagrammi
    ax2 = fig.add_subplot(1, 2, 2)

    t_edges = np.array(
        [0, 1.5, 3, 6, 12, 18, 24, 30]
    )

    # P/PET-suhteen rajat logaritmisella asteikolla
    ratio_edges = np.array([
        1/64, 1/32, 1/16, 1/8, 1/4,
        1/2, 1, 2, 4, 8
    ])

    # Luokittelumatriisin solujen keskipisteet
    tc = (t_edges[:-1] + t_edges[1:]) / 2
    rc = np.sqrt(ratio_edges[:-1] * ratio_edges[1:])

    TT, RR = np.meshgrid(tc, rc)

    # PET ja vastaava vuosisade jokaiselle solulle
    PET = 58.93 * TT
    PP = RR * PET

    # Käytetään samaa luokittelijaa kuin kartassa
    zone_grid = classify_holdridge_vectorized(TT, PP)

    ax2.pcolormesh(
        t_edges,
        ratio_edges,
        zone_grid,
        cmap=cmap,
        norm=norm,
        shading="flat"
    )

    ax2.set_yscale("log")
    ax2.set_xlabel("Biotemperature (°C)")
    ax2.set_ylabel("P/PET ( precipitation / potential evaporation)")
    ax2.set_title("Holdridge: temperature and moisture")
    ax2.set_xticks(t_edges)
    ax2.set_ylim(ratio_edges[0], ratio_edges[-1])
    ax2.grid(alpha=0.25, which="both")

    # Vyöhykkeiden selite
    handles = [
        Patch(
            facecolor=HLZ[i][1],
            label=f"{i}: {HLZ[i][0]}"
        )
        for i in ids if i != 0
    ]
    ax.imshow(
        ocean_mask,
        extent=[-180, 180, -90, 90],
        origin="upper",
        cmap=cmap1, vmin=0, vmax=0,
        transform=ccrs.PlateCarree(),
        interpolation="nearest"
    )
    #ax.contour(
    #    landmask, levels=[0,1],colors=["k"], lw=3,
    #    extent=[-180, 180, -90, 90],
    #    origin="upper", 
    #    transform=ccrs.PlateCarree()
    #) 
 
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.20)
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.show()




# ============================================================
# WIND VISUALIZATION
# ============================================================


def visualize_wind_v5(
    u,
    v,
    speed,
    quiver_step=2,
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

def run_model(dem,
    landmask, 
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
    fake_rain_system=0.0,
    clouds_present=1.0,
    cloud_strength=1.0,

    use_simple_ice=False,
    use_dynamic_ice=False,
    ice_temperature=263.0,

    diffusion=0.0,diffusion_scaled=0.0,
    budyko_heat_transport_coeff=0.0,
    save_every_hours=24
):

    use_ice=False
    if(use_simple_ice==True):
        use_ice=True
    if(use_dynamic_ice==True):
        use_ice=True
    # --------------------------------------------------------
    # arrays
    # --------------------------------------------------------
    
    demup0 = np.maximum(
    0.0,
    dem
    ).astype(np.float64)
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

    #temperature = np.full(
    #    (ny, nx),
    #    initial_temperature,
    #    dtype=np.float64
    #)

    # Ominaiskosteus kg/kg
    #humidity = np.full(
    #temperature.shape,
    #0.005,
    #dtype=np.float64,
    #)

    temperature, humidity=generate_planet_base(nx, ny,landmask, max_blur_distance=1)
    #plt.imshow(humidity)
    #plt.show()
    #quit(-1)
    # Ominaiskosteus kg/kg
    cloud_water = np.full(
    temperature.shape,
    0.000,
    dtype=np.float64,
    )
    soil_moisture = np.ones_like(
    temperature,
    dtype=np.float64, 
    )
    rain_mm_h = np.full(
    temperature.shape,
    0.000,
    dtype=np.float64,
    )
    rain_accumulated = np.full(
    temperature.shape,
    0.000,
    dtype=np.float64,
    )
    snow = np.full(
    temperature.shape,
    0.000,
    dtype=np.float64,
    )
    glacier = np.full(
    temperature.shape,
    0.000,
    dtype=np.float64,
    )
    sea_ice = np.full(
    temperature.shape,
    0.000,
    dtype=np.float64,
    )
    snow_mask = np.full(
    temperature.shape,
    0.000,
    dtype=np.float64,
    )
    glacier_mask = np.full(
    temperature.shape,
    0.000,
    dtype=np.float64,
    )
    sea_ice_mask = np.full(
    temperature.shape,
    0.000,
    dtype=np.float64,
    )
    snow_depth_m = np.full(
    temperature.shape,
    0.000,
    dtype=np.float64,
    )
    glacier_depth_m = np.full(
    temperature.shape,
    0.000,
    dtype=np.float64,
    )
    sea_ice_depth_m = np.full(
    temperature.shape,
    0.000,
    dtype=np.float64,
    )

    #snow = np.zeros((height, width), dtype=np.float64)
    #glacier = np.zeros((height, width), dtype=np.float64)
    #sea_ice = np.zeros((height, width), dtype=np.float64)
    
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
    rh_raster= np.zeros(
    (ny, nx),
    dtype=np.float64
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


    pressure_raster=pressure_raster*0+131325
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
    saved_rain = []
    cloudiness = create_cloud_raster(
    simu_ny,
    simu_nx,
    clouds_present,
    tilt,
    0)

    temperature_year_record=[]
    temperature_day_record=[]
    # --------------------------------------------------------
    # MAIN TIME LOOP
    # --------------------------------------------------------
    last_year=years-1
    last_day=(total_hours/dt_hours)-24
    nn=-1
    for step in range(total_hours):
        nn=nn+1
        time_hours = step * dt_hours
        time_years = time_hours/(24*365.25)
        
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

        temperature, ASR, decl, H, cosz= energy_step(
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
            cloudiness,
            use_ice,
            ice_temperature
        )
        
        crashing = np.isnan(temperature).any()
        if(crashing==True):
            print("Error. Temperature nan, break. Maybe dt too long.")
            quit(-1)
 

        # ----------------------------------------------------
        # optional horizontal heat transport
        # ----------------------------------------------------
        #diffusion_original=0.0
        ## use diffusion

        if (diffusion> 0.0):
           #if (diffusion > 0.0):
           #   temperature = diffuse_temperature(
           #   temperature,
           #   material,
           #   K0=diffusion,
           #   dt=dt_seconds, rplanet_m=planet_radius_m,
           #   k_ocean=1.0,
           #   k_land=0.15,
           #   k_sea_ice=0.10,
           #   k_snow_ice=0.05,
           #   ) 
           #   humidity_diffusion=1   
           #   K_humidity=3000
           #   if(humidity_diffusion>0):
           #   humidity = diffuse_humidity(
           #   humidity,
           #   material,
           #   K0=K_humidity,
           #   dt=dt_seconds,
           #   rplanet_m=planet_radius_m,
           #   )
           #   rain_mm_h=(humidity/dt_seconds) 
           #   rain_accumulated=rain_accumulated+rain_mm_h       
           #   #plt.imshow(humidity)
           #   #plt.show()
           #humidity_diffusion=1   
           #K_humidity=3000
           #temperature, humidity = diffuse_temperature_and_humidity(
           #T=temperature,
           #humidity=humidity,
           #material=material,
           #K0=diffusion,
           #dt=dt_seconds,
           #rplanet_m=planet_radius_m,
           # Lämpö
           #k_ocean=1.0,
           #k_land=0.15,
           #k_sea_ice=0.10,
           #k_snow_ice=0.05,
           # Kosteuden itsenäinen diffuusio
           #D_humidity=3000,
           # Lämmön ja kosteuden kytkentä
           #)
           #temperature, humidity=diffuse_temperature_and_humidity_coupled(
           #temperature,
           #humidity,
           #material,
           #K0=diffusion*1,
           #dt=dt_seconds,
           #rplanet_m=planet_radius_m,
           #k_ocean=1.0,
           #k_land=1.0,
           #k_sea_ice=1.0,
           #k_snow_ice=1.0,
           #k_humidity_ocean=1.0,
           #k_humidity_land=0.5,
           #k_humidity_sea_ice=0.2,
           #k_humidity_snow_ice=0.2,
           #D_humidity=0.1,
           #alpha_heat_moisture=0.1,
           #)
           temperature, humidity = diffuse_temperature_and_humidity(
           T=temperature,
           q=humidity,
           material=material,
           K0=diffusion,
           dt=dt_seconds,
           rplanet_m=planet_radius_m,
           h_eff=15.0  # Voit myös määrittää valinnaisen vaikutuskorkeuden (oletus 10.0)
           )
           #rain_mm_h=(humidity/dt_seconds) 
           #rain_accumulated=rain_accumulated+rain_mm_h          

        if(diffusion_scaled>0):
           #temperature, humidity=diffuse_temperature_and_humidity_scaled_advanced(
           #temperature,
           #humidity,
           #material,
           #diffusion,                  # Pohjadiffuusio (esim. 0.65)
           #dt_seconds,
           #omega_ratio=1.0,     # Planeetan pyörimisnopeus suhteessa Maahan (1.0 = Maa)
           #v_hadley_max=0.05,   # Hadley-solun maksimituulennopeus skaalatussa avaruudessa
           #k_ocean=1.0,
           #k_land=0.5,
           #k_sea_ice=0.25,
           #k_snow_ice=0.25
           #) 
           dt_diffusion=(1/(3600*24*365))*24
           temperature, humidity=diffuse_temperature_and_humidity_scaled(temperature,humidity,material,diffusion_scaled,dt_diffusion,k_ocean=1.0,k_land=0.5,k_sea_ice=0.25,k_snow_ice=0.25)


           #rain=(humidity/dt_seconds) 
           #rain_accumulated=rain_accumulated+rain_mm_h       
           #plt.imshow(humidity)
           #plt.show()

        if(budyko_heat_transport_coeff>0):
            temperature=budyko_transport_temperature(temperature,planet_radius_m,budyko_heat_transport_coeff,dt_seconds)

        if(fake_rain_system==True):
               #print("FR")
               T_c = temperature - 273.15
               e_sat = 611.2 * np.exp((17.67 * T_c) / (T_c + 243.5))
               max_humidity = (e_sat / (461.5 * temperature)) * 1000
               humidity0=max_humidity
               rain_mm_h=humidity0*1000*0.05
               if(rotation_period_days<=14):
                   #temperature, humidity, pressure_raster, rain_mm_h, rh_raster, cloudiness2=earth_like_weather(temperature,humidity0, pressure_raster,material,decl,dt_seconds,itcz_coeff=1.0,h_eff=10.0 )
                   temperature, humidity, pressure_raster,rain_mm_h, rh_raster, cloudiness,u_new, v_new= earth_like_weather(temperature, humidity0, pressure_raster, material, decl, dt_seconds,itcz_coeff=1.0,h_eff=10.0,u=None,v=None)
                   rain=rain_mm_h/dt_hours
                   rain=rain*10000 
               if(rotation_period_days>14):
                   (T_new_0000, humidity, pressure_raster,rain_mm_h, rh_raster, cloudiness,u_new, v_new) = locked_planet_weather(temperature, humidity0, pressure_raster, material,sun_decl=decl,dt_seconds=dt_seconds,substellar_lat=0.0,substellar_lon=0.0, rotation_period=rotation_period_days * 86400.0,T_day=np.max(temperature),T_night=np.min(temperature),tau_rad=rotation_period_days * 86400.0,tau_pressure=3 * 86400.0,u=None,v=None, central_hurricane=True,r_rainy=np.deg2rad(30.0),r_cloudy=np.deg2rad(60.0),hurricane_rain_max=5.0,hurricane_cloud_max=0.8,hurricane_wind_max=30.0)
                   rain=rain_mm_h/dt_hours
                   rain=rain*10000

        if (use_atmosphere==False):
            use_winds=False
            use_humidity=False
            use_rain=False
            use_ocean_currents=True
        if (use_atmosphere==True):
            use_winds=True
            use_humidity=True
            use_rain=True
            use_ocean_currents=True

        if (use_winds==True):
            (
            pressure_raster, humidity, temperature,
            u,
            v,
            wind_speed,
            u_upper,
            v_upper,
            vertical_velocity
            )=atmosphere_step_v10(
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

            humidity = enforce_polar_humidity(
            humidity,
            polar_humidity=0.0002,
            polar_rows=1,
            )
            #print("Pressure:",pressure_raster.min(),pressure_raster.max(),pressure_raster.mean())
            #print("Wind:",wind_speed.min(),wind_speed.max(),wind_speed.mean())

        if(use_rain==True):
            rh, cloud_fraction, rain_mm_h, humidity, cloud_water = (
            cloud_rain_step(
            temperature,
            humidity,
            cloud_water,
            dt_seconds,
            pressure_raster=pressure_raster,
            vertical_velocity=vertical_velocity,
            )
            )
            #rain_accumulated=rain_accumulated + 1*(rain_mm_h*dt_hours)/1000
            rain= 1*(rain_mm_h/dt_hours)/1000
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

            temperature = ocean_heat_transport_step(
            temperature,
            ocean_u_surface,
            ocean_v_surface,
            material,
            dem,
            dt_seconds,
            planet_radius_m,
            )

        #use_dynamic_ice=True
        if (use_dynamic_ice==True):
            (
            snow, glacier, sea_ice, material,
            snow_mask, snow_depth_m,
            glacier_mask, glacier_depth_m,
            sea_ice_mask, sea_ice_depth_m,
            ) = update_ice(
            demup0,
            landmask,
            temperature,
            rain_mm_h,
            snow,
            glacier,
            sea_ice,
            material,
            snow_mask,
            snow_depth_m,
            glacier_mask,
            glacier_depth_m,
            sea_ice_mask,
            sea_ice_depth_m,
            dt_seconds,
            )
            glacier=spread_ice(
            demup0, landmask, glacier, dt_seconds, planet_radius_m
            )
            #print(np.mean(snow_depth_m), np.min(snow_depth_m), np.max(snow_depth_m))       
        # ----------------------------------------------------
        # save
        # ----------------------------------------------------

        #if step % int(save_every_hours / dt_hours) == 0:
        #    saved_temperature.append(
        #        temperature.copy()
        #    )
        if(int(time_years)>=last_year):
            saved_temperature.append(
                temperature.copy()
            )			
            saved_rain.append(
                rain.copy()
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


    #visualize_wind(u,v,speed=None,temperature=None,quiver_step=2,stream_density=2.0,title="Planetary surface wind")
    #visualize_wind(u_upper,v_upper,speed=None,temperature=None,quiver_step=2,stream_density=2.0,title="Planetary upper wind")
    #visualize_wind(u,v,speed=vertical_velocity,temperature=vertical_velocity,quiver_step=2,stream_density=2.0,title="Planetary surface wind")

    #plt.imshow(humidity)
    #plt.show()
    #plt.imshow(pressure_raster)
    #plt.show()
    #plt.imshow(vertical_velocity)
    #plt.show()
    #plt.imshow(divergence)
    #plt.show()    
    #plt.imshow(ocean_temperature_surface, cmap="coolwarm")
    #rain_annual=humidity*1000*100
    #rain_annual=(humidity/nn)*1000*100*100
    #rain_annual=(rain_accumulated/years)*50
    #plt.imshow(rain_annual)
    #plt.show()
    #plt.imshow(glacier)
    #plt.imshow(glacier_depth_m)
    #plt.show()

    return (
        temperature,  rain, humidity,
        material,
        np.array(saved_temperature), np.array(saved_rain)
    )

def run_stubb(): 
    return(0)







def downscale_rainfall_naiive(
    simu_dem,
    accurate_dem,
    simu_rainfall,
    downscaling_type="coefficient",
    param=0.10,
):
    """
    Korkeuseroon perustuva naiivi sademäärän alas-/ylösskaalaus.

    Parameters
    ----------
    simu_dem : numpy.ndarray
        Simulaation korkeusmalli tarkemman mallin ruudukossa (m).
    accurate_dem : numpy.ndarray
        Tarkka korkeusmalli (m).
    simu_rainfall : numpy.ndarray
        Simulaation sademäärä samassa ruudukossa kuin korkeusmallit.
    downscaling_type : str
        "coefficient" tai "exponential".
    param : float
        coefficient: suhteellinen muutos per 100 m.
        exponential: eksponentin kerroin per metri.

    Returns
    -------
    numpy.ndarray
        Korkeuserolla korjattu sademäärä.
    """

    # Muunnetaan syötteet NumPy-taulukoiksi
    simu_dem = np.asarray(simu_dem, dtype=float)
    accurate_dem = np.asarray(accurate_dem, dtype=float)
    simu_rainfall = np.asarray(simu_rainfall, dtype=float)
    # ------------------------------------------------------------
    # 2. Skaalauskerroin simulaatioresoluutiosta accurate-resoluutioon
    # ------------------------------------------------------------

    zoom_y = accurate_dem.shape[0] / simu_rainfall.shape[0]
    zoom_x = accurate_dem.shape[1] / simu_rainfall.shape[1]

    zoom_factor = (zoom_y, zoom_x)

    # ------------------------------------------------------------
    # 3. Interpoloidaan simulaation lämpötila accurate-resoluutioon
    # ------------------------------------------------------------

    rainfall_base = zoom(
        simu_rainfall,
        zoom_factor,
        order=1,
    )
    zoom2_y = accurate_dem.shape[0] / simu_rainfall.shape[0]
    zoom2_x = accurate_dem.shape[1] / simu_rainfall.shape[1]

    zoom2_factor = (1/zoom2_y, 1/zoom2_x)

    # ------------------------------------------------------------
    # 3. Interpoloidaan simulaation lämpötila accurate-resoluutioon
    # ------------------------------------------------------------

    dem_inaccurate = zoom(
        accurate_dem,
        zoom2_factor,
        order=1,
    )   
    dem_inaccurate_expanded= zoom(
        dem_inaccurate,
        zoom_factor,
        order=1,
    )   
    
    elevation_diff=accurate_dem-dem_inaccurate_expanded
    

    downscaled_rainfall = np.full(
        simu_rainfall.shape, np.nan, dtype=float
    )

    if downscaling_type == "coefficient":

        # Esim. param=0.10 tarkoittaa +10 % / 100 m.
        factor = 1.0 + param * elevation_diff / 100.0

        # Sade ei voi muuttua negatiiviseksi.
        factor = np.maximum(factor, 0.0)

    elif downscaling_type == "exponential":

        # Esim. param=0.0005 tarkoittaa alpha=0.0005 / m.
        exponent = param * elevation_diff

        # Estetään numeerinen ylivuoto äärimmäisillä arvoilla.
        factor = np.exp(np.clip(exponent, -50, 50))

    else:
        raise ValueError(
            "downscaling_type pitää olla "
            "'coefficient' tai 'exponential'."
        )
    
    downscaled_rainfall =  rainfall_base * factor

    return downscaled_rainfall



def downscale_temperature_naiive(
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


import numpy as np

def downscale_monthly_rasters(
    temps12,
    rains12,
    simu_dem,
    accurate_dem,
    rain_param=0.12,
):
    """
    Downscale kuukausittaiset lämpötila- ja sadet rasterit.

    Parametrit
    ----------
    temps12 : ndarray, shape (12, ny, nx)
        Kuukausittainen lämpötila (°C) karkealla hilalla.
    rains12 : ndarray, shape (12, ny, nx)
        Kuukausittainen sademäärä (mm) karkealla hilalla.
    simu_dem : ndarray, shape (ny, nx)
        Karkean hilan korkeusmalli.
    accurate_dem : ndarray, shape (ny2, nx2)
        Tarkemman hilan korkeusmalli.
    rain_param : float
        Sademäärän suhteellinen kasvu per 100 m.
        Oletus 0.12 = 12 % / 100 m.

    Palauttaa
    ---------
    temps12_downscaled : ndarray, shape (12, ny2, nx2)
        Downscalatut kuukausilämpötilat (°C).
    rains12_downscaled : ndarray, shape (12, ny2, nx2)
        Downscalatut kuukausisademäärät (mm).
    """

    # Käsitellään DEM-arvot samalla tavalla kuin alkuperäisessä koodissa
    simu_dem_clean = np.where(simu_dem > 0, simu_dem, 0)
    accurate_dem_clean = np.where(accurate_dem > 0, accurate_dem, 0)

    if temps12.shape[0] != 12 or rains12.shape[0] != 12:
        raise ValueError("temps12- ja rains12-pinoissa pitää olla 12 kuukautta.")

    if temps12.shape[1:] != simu_dem.shape:
        raise ValueError("temps12-rasterin hilakoko ei vastaa simu_dem-rasteria.")

    if rains12.shape[1:] != simu_dem.shape:
        raise ValueError("rains12-rasterin hilakoko ei vastaa simu_dem-rasteria.")

    temps_out = []
    rains_out = []

    for month in range(12):
        # Lämpötila: käytetään samaa menetelmää kuin vuosiaineistolle
        temp_downscaled = downscale_temperature_naiive(
            temps12[month],
            simu_dem_clean,
            accurate_dem_clean,
        )

        # Sade: 12 % kasvu / 100 m
        rain_downscaled = downscale_rainfall_naiive(
            simu_dem=simu_dem_clean,
            accurate_dem=accurate_dem_clean,
            simu_rainfall=rains12[month],
            downscaling_type="coefficient",
            param=rain_param,
        )

        temps_out.append(temp_downscaled)
        rains_out.append(rain_downscaled)

    return np.stack(temps_out), np.stack(rains_out)


import numpy as np


# Köppen–Geiger-luokkien selitteet.
# Koodi 0 tarkoittaa puuttuvaa tai virheellistä dataa.
KOPPEN_CLASSES = {
    1:  ("Af",  "Trooppinen sademetsäilmasto"),
    2:  ("Am",  "Trooppinen monsuuni-ilmasto"),
    3:  ("Aw",  "Trooppinen savanni-ilmasto"),
    4:  ("BWh", "Kuuma aavikkoilmasto"),
    5:  ("BWk", "Kylmä aavikkoilmasto"),
    6:  ("BSh", "Kuuma aroilmasto"),
    7:  ("BSk", "Kylmä aroilmasto"),
    8:  ("Csa", "Kuuma kesä, kuiva kesä"),
    9:  ("Csb", "Lämmin kesä, kuiva kesä"),
    10: ("Csc", "Viileä kesä, kuiva kesä"),
    11: ("Cwa", "Kuuma kesä, kuiva talvi"),
    12: ("Cwb", "Lämmin kesä, kuiva talvi"),
    13: ("Cwc", "Viileä kesä, kuiva talvi"),
    14: ("Cfa", "Kuuma kesä, ei kuivaa vuodenaikaa"),
    15: ("Cfb", "Lämmin kesä, ei kuivaa vuodenaikaa"),
    16: ("Cfc", "Viileä kesä, ei kuivaa vuodenaikaa"),
    17: ("Dsa", "Kuuma kesä, kuiva kesä, kylmä mannerilmasto"),
    18: ("Dsb", "Lämmin kesä, kuiva kesä, kylmä mannerilmasto"),
    19: ("Dsc", "Viileä kesä, kuiva kesä, kylmä mannerilmasto"),
    20: ("Dsd", "Erittäin kylmä talvi, kuiva kesä"),
    21: ("Dwa", "Kuuma kesä, kuiva talvi, mannerilmasto"),
    22: ("Dwb", "Lämmin kesä, kuiva talvi, mannerilmasto"),
    23: ("Dwc", "Viileä kesä, kuiva talvi, mannerilmasto"),
    24: ("Dwd", "Erittäin kylmä talvi, kuiva talvi"),
    25: ("Dfa", "Kuuma kesä, ei kuivaa vuodenaikaa, mannerilmasto"),
    26: ("Dfb", "Lämmin kesä, ei kuivaa vuodenaikaa, mannerilmasto"),
    27: ("Dfc", "Viileä kesä, ei kuivaa vuodenaikaa, mannerilmasto"),
    28: ("Dfd", "Erittäin kylmä talvi, ei kuivaa vuodenaikaa"),
    29: ("ET",  "Tundra"),
    30: ("EF",  "Jäätikköilmasto"),
}


def koppen_geiger(
    temps12,
    rains12,
    hemisphere="north",
    missing_value=0,
):
    """
    Köppen-Geiger-luokittelu kuukausittaisista rasteripinoista.

    Parametrit
    ----------
    temps12 : ndarray, (12, ny, nx)
        Kuukausittainen keskilämpötila, °C.
    rains12 : ndarray, (12, ny, nx)
        Kuukausittainen sademäärä, mm/kk.
    hemisphere : str
        "north" tai "south". Määrittää kesän ja talven kuukaudet.
    missing_value : int
        Luokkarasterin puuttuvan datan koodi.

    Palauttaa
    ---------
    classes : ndarray, (ny, nx), int
        Luokkarasteri. 0 = puuttuva data.
    """

    T = np.asarray(temps12, dtype=float)
    P = np.asarray(rains12, dtype=float)

    if T.ndim != 3 or P.ndim != 3:
        raise ValueError("Lämpötila- ja sadeaineistojen pitää olla 3D-pinoja.")

    if T.shape != P.shape or T.shape[0] != 12:
        raise ValueError("Molempien aineistojen muodon pitää olla (12, ny, nx).")

    if hemisphere.lower() in ("north", "n", "pohjoinen"):
        summer = np.arange(3, 9)       # huhti-syyskuu
        winter = np.array([9, 10, 11, 0, 1, 2])
    elif hemisphere.lower() in ("south", "s", "etelä"):
        summer = np.array([9, 10, 11, 0, 1, 2])
        winter = np.arange(3, 9)
    else:
        raise ValueError("hemisphere: 'north' tai 'south'")

    # Kaikkien 12 kuukauden pitää olla kelvollisia.
    valid = (
        np.all(np.isfinite(T), axis=0)
        & np.all(np.isfinite(P), axis=0)
        & np.all(P >= 0, axis=0)
    )

    # Vuositason lämpötila- ja sademuuttujat.
    MAT = np.mean(T, axis=0)
    MAP = np.sum(P, axis=0)
    Tc = np.min(T, axis=0)
    Th = np.max(T, axis=0)
    Pdry = np.min(P, axis=0)

    # Kesän ja talven sademäärät.
    Ps = P[summer]
    Pw = P[winter]

    Psdry = np.min(Ps, axis=0)
    Pswet = np.max(Ps, axis=0)
    Pwdry = np.min(Pw, axis=0)
    Pwwet = np.max(Pw, axis=0)

    summer_rain = np.sum(Ps, axis=0)
    winter_rain = np.sum(Pw, axis=0)

    # Kuukausien määrä, jolloin lämpötila on yli 10 °C.
    nT10 = np.sum(T > 10, axis=0)

    # Kuivuutta koskeva raja: mm/vuosi ja lämpötilasta riippuva.
    # Kesä- ja talvisateiden osuus vuosisateesta ratkaisee kaavan.
    with np.errstate(divide="ignore", invalid="ignore"):
        winter_fraction = np.divide(
            winter_rain, MAP,
            out=np.zeros_like(MAP),
            where=MAP > 0,
        )
        summer_fraction = np.divide(
            summer_rain, MAP,
            out=np.zeros_like(MAP),
            where=MAP > 0,
        )

    Pthreshold = np.where(
        winter_fraction >= 0.70,
        2 * MAT,
        np.where(
            summer_fraction >= 0.70,
            2 * MAT + 28,
            2 * MAT + 14,
        ),
    )

    # Negatiivinen aridisuusraja ei ole mielekäs.
    # Nollan alittavassa lämpötilassa käytetään nollaa.
    Pthreshold = np.maximum(Pthreshold, 0)

    classes = np.full(T.shape[1:], missing_value, dtype=np.int16)

    # B: kuiva ilmasto. B-luokka tarkistetaan ennen A/C/D/E-luokkia.
    arid = valid & (MAP < 10 * Pthreshold)
    desert = arid & (MAP < 5 * Pthreshold)
    steppe = arid & ~desert

    classes[desert & (MAT >= 18)] = 4   # BWh
    classes[desert & (MAT < 18)] = 5    # BWk
    classes[steppe & (MAT >= 18)] = 6  # BSh
    classes[steppe & (MAT < 18)] = 7   # BSk

    non_arid = valid & ~arid

    # A: trooppinen, kylmimmän kuukauden lämpötila >= 18 °C.
    tropical = non_arid & (Tc >= 18)

    classes[tropical & (Pdry >= 60)] = 1  # Af

    monsoon_limit = 100 - MAP / 25
    classes[
        tropical & (Pdry < 60) & (Pdry >= monsoon_limit)
    ] = 2  # Am

    classes[
        tropical & (Pdry < 60) & (Pdry < monsoon_limit)
    ] = 3  # Aw

    # E: polaarinen ilmasto, lämpimin kuukausi <= 10 °C.
    polar = non_arid & (Th <= 10)
    classes[polar & (Th <= 0)] = 30  # EF
    classes[polar & (Th > 0)] = 29   # ET

    # C/D: lämpimin kuukausi > 10 °C.
    # Peel et al. (2007): C, jos Tc > 0 ja Tc < 18.
    # D, jos Tc <= 0.
    temperate = non_arid & (Th > 10) & (Tc > 0) & (Tc < 18)
    continental = non_arid & (Th > 10) & (Tc <= 0)

    # Kuiva kesä: kuivin kesäkuukausi < 40 mm ja
    # alle kolmasosa talven sateisimmasta kuukaudesta.
    dry_summer = (Psdry < 40) & (Psdry < Pwwet / 3)

    # Kuiva talvi: kuivin talvikuukausi alle 1/10
    # kesän sateisimmasta kuukaudesta.
    dry_winter = Pwdry < Pswet / 10

    # Kun molemmat ehdot täyttyvät, dry winter saa etusijan
    # tässä toteutuksessa, kuten yleisesti käytetyssä luokittelussa.
    season = np.zeros(T.shape[1:], dtype=np.int8)
    season[dry_summer] = 1
    season[dry_winter] = 2
    season[dry_summer & dry_winter] = 2

    # Kesälämpötilan kolmas kirjain.
    hot_summer = Th >= 22
    warm_summer = (~hot_summer) & (nT10 >= 4)
    cold_summer = (~hot_summer) & (~warm_summer) & (nT10 >= 1)
    very_cold_winter = (~hot_summer) & (Tc < -38)

    # C-luokan tunnukset:
    # s = kuiva kesä, w = kuiva talvi, f = ei kuivaa kautta.
    # a/b/c määritetään lämpimän kesän perusteella.
    for region, offset in ((temperate, 8), (continental, 17)):
        # offset-kartoitukset:
        # C: s=8..10, w=11..13, f=14..16
        # D: s=17..20, w=21..24, f=25..28
        is_c = offset == 8

        # Valitaan kuivuusryhmä.
        group_s = region & (season == 1)
        group_w = region & (season == 2)
        group_f = region & (season == 0)

        if is_c:
            # Csc on harvinainen, mutta kuuluu luokitteluun.
            classes[group_s & hot_summer] = 8
            classes[group_s & warm_summer] = 9
            classes[group_s & ~hot_summer & ~warm_summer] = 10

            classes[group_w & hot_summer] = 11
            classes[group_w & warm_summer] = 12
            classes[group_w & ~hot_summer & ~warm_summer] = 13

            classes[group_f & hot_summer] = 14
            classes[group_f & warm_summer] = 15
            classes[group_f & ~hot_summer & ~warm_summer] = 16

        else:
            classes[group_s & hot_summer] = 17
            classes[group_s & warm_summer] = 18
            classes[group_s & ~hot_summer & ~warm_summer & ~very_cold_winter] = 19
            classes[group_s & very_cold_winter] = 20

            classes[group_w & hot_summer] = 21
            classes[group_w & warm_summer] = 22
            classes[group_w & ~hot_summer & ~warm_summer & ~very_cold_winter] = 23
            classes[group_w & very_cold_winter] = 24

            classes[group_f & hot_summer] = 25
            classes[group_f & warm_summer] = 26
            classes[group_f & ~hot_summer & ~warm_summer & ~very_cold_winter] = 27
            classes[group_f & very_cold_winter] = 28

    return classes


def koppen_statistics(classes, pixel_area_km2=None):
    """
    Luokkien esiintymismäärät ja haluttaessa pinta-alat.

    pixel_area_km2:
        Yksi luku, jos jokainen pikseli edustaa samaa pinta-alaa.
        Maantieteellisessä leveys-pituusastehilassa pinta-ala
        voi vaihdella, joten käytä silloin erillistä pinta-alahilaa.
    """
    classes = np.asarray(classes)
    total = np.count_nonzero(classes > 0)

    rows = []

    for code, (symbol, description) in KOPPEN_CLASSES.items():
        count = int(np.count_nonzero(classes == code))

        row = {
            "code": code,
            "class": symbol,
            "description": description,
            "pixels": count,
            "percent": 100 * count / total if total else 0.0,
        }

        if pixel_area_km2 is not None:
            row["area_km2"] = count * pixel_area_km2

        rows.append(row)

    import pandas as pd
    return (
        pd.DataFrame(rows)
        .sort_values("pixels", ascending=False)
        .reset_index(drop=True)
    )

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


# Luokille erilliset värit. Värit on ryhmitelty ilmastotyypeittäin.
KOPPEN_COLORS = {
    1: "#0000FF",  2: "#0078FF",  3: "#46A9FA",
    4: "#FF0000",  5: "#FF9696",  6: "#F5A623",  7: "#FFDC64",
    8: "#FF00FF",  9: "#C000C0", 10: "#963296",
    11: "#FF69B4", 12: "#DB7093", 13: "#C080A0",
    14: "#FFFF00", 15: "#C8C800", 16: "#969600",
    17: "#AA0000", 18: "#730000", 19: "#550000", 20: "#400000",
    21: "#00FF00", 22: "#96FF96", 23: "#64C864", 24: "#328032",
    25: "#00C800", 26: "#00A000", 27: "#008000", 28: "#005000",
    29: "#B4FFFF", 30: "#808080",
}


def plot_koppen(
    classes,
    show_all=True,
    title="Köppen–Geiger-ilmastoluokitus",
    ax=None,
):
    """
    Piirtää Köppen-luokkarasterin.

    show_all=True: selitteessä kaikki 30 luokkaa.
    show_all=False: selitteessä vain rasterissa esiintyvät luokat.
    ax: halutessasi olemassa oleva matplotlib-akseli.
    """
    classes = np.asarray(classes)

    if classes.ndim != 2:
        raise ValueError("classes pitää olla kaksiulotteinen rasteri.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 9))
    else:
        fig = ax.figure

    present = set(np.unique(classes).tolist()) - {0}

    # Rajat ovat kokonaislukuluokkien keskikohdissa.
    color_list = ["#FFFFFF"] + [
        KOPPEN_COLORS[i] for i in range(1, 31)
    ]
    cmap = ListedColormap(color_list)
    cmap.set_bad("white")

    masked = np.ma.masked_equal(classes, 0)

    ax.imshow(
        masked,
        cmap=cmap,
        norm=BoundaryNorm(np.arange(-0.5, 31.5, 1), cmap.N),
        interpolation="nearest",
        origin="upper", extent=[-180, 180, -90, 90]
    )

    ax.set_title(title)
    ax.set_xlabel("X / sarake")
    ax.set_ylabel("Y / rivi")

    legend_codes = range(1, 31) if show_all else sorted(present)

    handles = [
        Patch(
            facecolor=KOPPEN_COLORS[i],
            edgecolor="black",
            linewidth=0.3,
            label=f"{KOPPEN_CLASSES[i][0]} – {KOPPEN_CLASSES[i][1]}",
        )
        for i in legend_codes
    ]

    ax.legend(
        handles=handles,
        title="Ilmastoluokka",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        title_fontsize=10,
        ncol=1,
        borderaxespad=0,
    )

    fig.tight_layout()
    return fig, ax

def collect_monthly_data(dt_hours, temperature_record, prec_record):
    # Kuukausien päivien lukumäärät (ei karkausvuosi)
    days_in_month = np.array([
        31, 28, 31, 30, 31, 30,
        31, 31, 30, 31, 30, 31
    ])

    # Kuukausien pituudet mittausaskelina
    steps_in_month = (
        days_in_month * 24 / dt_hours
    ).astype(int)

    # Rasterit muotoa (12, ny, nx)
    ny, nx = temperature_record.shape[1:]

    temps12 = np.zeros((12, ny, nx))
    precs12 = np.zeros((12, ny, nx))
    deltas_diurnal = np.zeros((12, ny, nx))

    start = 0

    for month in range(12):
        end = start + steps_in_month[month]

        # Kuukauden keskilämpötila
        temps12[month] = np.mean(
            temperature_record[start:end], axis=0
        )
        coeff=dt_hours/(24*30.5)
        # Kuukauden sadesumma
        precs12[month] = np.sum(
            prec_record[start:end], axis=0
        )*coeff

        # Vuorokausittainen lämpötilan vaihteluväli
        temp_month = temperature_record[start:end]

        # Muoto: (päivät, mittausaskeleet/päivä, ny, nx)
        steps_per_day = int(24 / dt_hours)

        temp_daily = temp_month.reshape(
            days_in_month[month],
            steps_per_day,
            ny,
            nx
        )

        # Päivän maksimi- ja minimilämpötilan erotus
        daily_delta = (
            np.max(temp_daily, axis=1)
            - np.min(temp_daily, axis=1)
        )

        # Kuukauden keskimääräinen vuorokautinen vaihtelu
        deltas_diurnal[month] = np.mean(
            daily_delta, axis=0
        )

        start = end

    return temps12, precs12, deltas_diurnal




import numpy as np
from scipy.ndimage import distance_transform_edt

def post_process_precipitation_v2(precipitation_map, dem_map, planet_radius_m=6371000.0):
    """
    Post-prosessoi sademääräkartan luomalla aavikkovyöhykkeet, joiden paikka
    siirtyy korkeammille leveysasteille (25 -> 45) kaukana rannikosta.
    
    Parameters:
    - precipitation_map: 2D numpy-taulukko (sademäärä)
    - dem_map: 2D numpy-taulukko (korkeusmalli, jossa maa > 0 ja meri <= 0)
    - planet_radius_m: Planeetan säde metreinä (oletuksena Maa = 6 371 000 m)
    """
    map_height, map_width = precipitation_map.shape
    
    # 1. Luodaan leveysasteet kullekin riville (2D-matriisiksi suoraan)
    lat_1d = np.linspace(90, -90, map_height)
    latitudes_2d = lat_1d[:, np.newaxis] * np.ones((1, map_width))
    
    # 2. Lasketaan etäisyys rannikosta (pikseleinä)
    # distance_transform_edt laskee etäisyyden nollapisteisiin (meri)
    land_mask = dem_map > 0
    pixel_distances = distance_transform_edt(land_mask)
    
    # 3. Muutetaan pikselietäisyys metreiksi
    # Yhden pystypikselin koko metreinä (pohjoisnavalta etelänavalle on pi * R)
    pixel_size_m = (np.pi * planet_radius_m) / map_height
    distance_to_coast_m = pixel_distances * pixel_size_m
    
    # 4. Määritetään aavikon keskipisteen siirtymä rannikkoetäisyyden mukaan
    # Asetetaan kynnysmatka (esim. 1500 km), jossa aavikko saavuttaa 45 asteen rajan
    max_inland_distance_m = 1500000.0 
    
    # Skaalataan etäisyys välille 0.0 - 1.0 ja rajoitetaan maksimiin
    distance_ratio = np.clip(distance_to_coast_m / max_inland_distance_m, 0.0, 1.0)
    
    # Dynaaminen keskipiste kullekin pikselille (25 astetta rannikolla -> 45 astetta sisämaassa)
    dynamic_center_lat = 25.0 + (45.0 - 25.0) * distance_ratio
    
    # 5. Lasketaan 2D Gaussin käyrä dynaamisella keskipisteellä
    width_sigma = 7.0  
    max_dryness = 0.85 
    
    north_desert = np.exp(-((latitudes_2d - dynamic_center_lat) ** 2) / (2 * width_sigma ** 2))
    south_desert = np.exp(-((latitudes_2d - (-dynamic_center_lat)) ** 2) / (2 * width_sigma ** 2))
    desert_influence = np.maximum(north_desert, south_desert)
    
    # 6. Sovelletaan korjauskerroin
    rain_modifier_2d = 1.0 - (desert_influence * max_dryness)
    processed_map = precipitation_map * rain_modifier_2d
    
    # 7. Varmistetaan, että aavikkokorjaus puree vain maalla (meri säilyttää alkuperäisen sateen)
    processed_map = np.where(land_mask, processed_map, precipitation_map)
    
    return np.clip(processed_map, 0.0, None)




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
                noise_dem[r, c] =pnoise3(
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
        dem_power=0.55
        dem = np.pow(normalized_dem, dem_power) * height_delta

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



def rebound_simulation_test(duration_years, step_hours, simu_dem, simu_landmask, initial_temperature):

    shape1=np.shape(simu_dem)
    nx=shape1[1]
    ny=shape1[0]
    
    material=simu_landmask
    solrad=np.copy(simu_dem).astype(float)*0
    temperature=np.copy(simu_dem).astype(float)*0

    planet_names = [
    body["name"]
    for body in bodies
    if body["kind"] == "planet"
    ]
    duration_days=duration_years*365.25
    step_days=step_hours/24
    print("System:")
    print(f"Stars: {sum(b['kind'] == 'star' for b in bodies)}")
    print(f"Planets: {len(planet_names)}")
    #print(f"Albedo: {albedo}")
    print() 

    temp_record=[] 
    for day in np.arange(0.0,duration_days + step_days * 0.5,step_days,):

        time_years = float(day / DAYS_PER_YEAR)
        sim.integrate(step_days)

        for planet_name in planet_names:

            state = planet_state(planet_name, time_years)
            stars = star_directions(planet_name, time_years)
            radiation = radiation_and_temperature(
            planet_name,
            albedo=albedo,
            )
            
            orbital_longitude=state['orbital_longitude']
            rotation_angle=state["rotation_angle"] 
            distance_to_parent=state['distance_to_parent']
            lenu=len(stars)
            #print(lenu)
            #print(planet_name)    
            solrad=solrad*0
            for mmm in range(0, lenu):
                star=stars[mmm]
                contribution=radiation["contributions"][mmm]
                #print(star)
                #print(contribution)
                star_name=star['star']
                star_name=contribution['star']
                star_longitude=star['longitude']
                star_latitude=star['latitude']
                star_distance=star['distance_au']
                star_S=contribution['flux'] 
                solrad=solrad+snapshot_solar_radiation(star_S*1365.2,star_longitude,star_latitude,ny,nx)

                #print(star_longitude, star_latitude, star_distance)        
        total_S=radiation['total_flux']
        Teq=radiation['equilibrium_temperature_K']
        
        #print("\n" + "=" * 72)
        #print(f"Time: {day:.2f} days: ", total_S,Teq)
        temperature=nbody_energy_balance_step(
        temperature,
        solrad,
        material,
        step_days)
        temp_record.append(temperature)
        #plt.imshow(solrad_earths)

    temperature=np.mean(np.array(temp_record), axis=0)
    plt.imshow(temperature, cmap="rainbow")
    plt.show()

    return(temperature, material)


        
# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":

    # 64 x 32 planet
    nx = simu_nx
    ny = simu_ny
    
    T = np.full((ny, nx), initial_temperature-273.15)
    accurate_dem, accurate_landmask, simu_dem, simu_landmask, \
    accurate_stats, simu_stats = create_dem_and_landmask(seed1=seed1,
    height=height,
    width=width,
    simu_ny=simu_ny,
    simu_nx=simu_nx,
    height_delta=height_delta,
    sea_level=sea_level,
    )
    
    accurate_landmask2=np.copy(accurate_landmask)
    accurate_landmask2=np.where(accurate_landmask2==0, np.nan, 1)    
    print(" Sea   : ", round(accurate_stats["sea_share"],3 ))
    print(" Land  : ",round(accurate_stats["land_share"],3))
    print(" Min height         : ",round(accurate_stats["min_height"],3))    
    print(" Max height         : ",round(accurate_stats["max_height"],3))
    print(" Max-min height     : ",round(accurate_stats["height_delta"],3))
    print(" Mean land height   : ",round(accurate_stats["mean_land_height"],3))    
    print(" Mean sea elevation : ",round(accurate_stats["mean_sea_elevation"],3))      
    #rebound_simulation_test(simu_run_years, simu_dt_hours, simu_dem, simu_landmask, initial_temperature)
    #quit(-1)
    
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

    #print(accurate_stats)
    #print(simu_stats)
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
    #plt.imshow(simu_landmask)
    #plt.show()
    temperature, simu_rain_annual, humidity, material, temperature_record, rain_record = run_model(simu_dem, simu_landmask, 
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
    dt_hours=simu_dt_hours, fake_rain_system=fake_rain_system,
    clouds_present=clouds_present,
    cloud_strength=cloud_strength,
    use_simple_ice=use_simple_ice,
    use_dynamic_ice=use_dynamic_ice,
    ice_temperature=ice_temperature,
    diffusion=diffusion,diffusion_scaled=diffusion_scaled,
    budyko_heat_transport_coeff=budyko_heat_transport_coeff,
    save_every_hours=simu_save_every_hours
    )
   

    # Lasketaan keskiarvo
    mean_temperature_K = cosine_weighted_average(temperature, ny, nx)

   
    print()
    print("RESULT")
    print("------")
    print(
        "Mean T:",
         mean_temperature_K - 273.15,
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
    print(
        "Mean rain annual:",
        np.mean(simu_rain_annual),
        "mm"
    )

    print(
        "Min rain annual:",
        np.min(simu_rain_annual),
        "mm"
    )

    print(
        "Max rain annual:",
        np.max(simu_rain_annual),
        "mm"
    )

    print(
        "Range rain annual:",
        np.max(simu_rain_annual)
        - np.min(simu_rain_annual),
        "mm"
    )



# ============================================================
# YEARLY MEAN TEMPERATURE RASTER
# ============================================================

# history:
#   time, latitude, longitude
#
# Lasketaan vuoden keskilämpötila jokaiselle grid-solulle.

lenrec=len(temperature_record)

simu_mean_temperature_K = np.mean(temperature_record, axis=0)

# Celsius
simu_mean_temperature_C = simu_mean_temperature_K - 273.15

#simu_rain_annual = np.mean(rain_record, axis=0)*(simu_run_years*365.25*24)/simu_dt_hours
simu_rain_annual = np.sum(rain_record, axis=0)/lenrec

temps12, rains12, deltat_diurnal=collect_monthly_data(simu_dt_hours, np.array(temperature_record), np.array(rain_record))

temps12=temps12-273.15

#plt.imshow(rains12[0])
#plt.show()

#quit(-1)

lon = 50.0
lat = 60.0

if not (-180 <= lon <= 180 and -90 <= lat <= 90):
    raise ValueError("Koordinaatit ovat globaalin alueen ulkopuolella.")

# Interpoloidaan paikan tiedot
masl = float(bilinear_global(simu_dem, lon, lat))
temp_monthly = bilinear_global(temps12, lon, lat)
prec_monthly = bilinear_global(rains12, lon, lat)

print(f"\nSijainti: {lon:.3f}°, {lat:.3f}°")
print(f"Korkeus: {masl:.0f} m merenpinnasta")
print(f"Vuoden keskilämpötila: {np.mean(temp_monthly):.1f} °C")
print(f"Vuosisadanta: {np.sum(prec_monthly):.0f} mm")

plot_climate_diagram(lon, lat, masl, temp_monthly, prec_monthly)

#print (np.shape(temps12))

#plt.imshow(temps12[0])
#plt.imshow(deltat_diurnal[0] )
#plt.show()



#quit(-1)



#simu_landmask=np.copy(simu_dem)
#simu_landmask=np.where(simu_landmask>0,1,0)


#rotation_days=rotation_period_days
#dem=simu_dem
#landmask=simu_landmask
#temperature_K=simu_mean_temperature_K






downscaled_temperature_C = downscale_temperature_naiive(
    simu_mean_temperature_C,
    np.where(simu_dem>0,simu_dem,0),
    np.where(accurate_dem>0,accurate_dem, 0)
)


downscaled_rain_annual = downscale_rainfall_naiive(
    simu_dem=np.where(simu_dem>0,simu_dem,0),
    accurate_dem=np.where(accurate_dem>0,accurate_dem, 0),
    simu_rainfall=simu_rain_annual,
    downscaling_type="coefficient",
    param=0.12  # 12% kasvu per 100m
)

temps12_downscaled, rains12_downscaled = downscale_monthly_rasters(
    temps12=temps12,
    rains12=rains12,
    simu_dem=simu_dem,
    accurate_dem=accurate_dem,
    rain_param=0.12,
)


#if (fake_deserts==1):
#    ###downscaled_rain_annual=fake_post_process_precipitation(downscaled_rain_annual, np.where(accurate_dem>0, accurate_dem, 0) )
#    downscaled_rain_annual=post_process_precipitation_v2(downscaled_rain_annual, dem_map=accurate_dem, planet_radius_m=6371000.0)
	


# Kun haluat piirtää lämpötilan:
plot_temperature_map(downscaled_temperature_C, accurate_dem)

# Kun haluat piirtää sademäärän (korvaa 'oma_sademäärä_data' omalla muuttujallasi):
plot_precipitation_map(downscaled_rain_annual, accurate_dem, vmin=0, vmax=3500)

t_bio_temp=np.copy(downscaled_temperature_C)
pet_bio=t_bio_temp*58.93

holdridge_ids = classify_holdridge_vectorized(t_bio_temp, downscaled_rain_annual)



# Halutessasi tallenna tilasto
#stats.to_csv("koppen_statistics.csv", index=False)

#plt.imshow(holdridge_map )
#plt.show()
#plot_holdridge_map(
#    holdridge_ids*accurate_landmask,
#    accurate_landmask,
#    t_bio_temp,
#    downscaled_rain_annual
#)

# Luokittelu: pohjoinen pallonpuolisko
koppen_classes = koppen_geiger(
    temps12=temps12_downscaled,
    rains12=rains12_downscaled,
    hemisphere="north",
)
stats = koppen_statistics(koppen_classes*accurate_landmask)
print(stats.to_string(index=False))

# Piirrä kaikki 30 luokkaa selitteeseen
fig, ax = plot_koppen(
    koppen_classes*accurate_landmask,
    show_all=True,
    title="Köppen–Geiger: tarkennettu ilmastoluokitus",
)

plt.show()

print(".")
