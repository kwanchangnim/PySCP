import numpy as np
from lib.utils.PlanetData import Moon

mass_0 = 2000.0 # kg
Isp = 325.0 #s Specific impulse 

#####################################################

# Comparison of lunar landing trajectory strategies using numirical simulations
descent_approach_az_base = np.pi/4 # rad

# LLO parking orbit 60 nautical miles = 111120 meters
alt_parking_orbit_base = 111120
# braking altitude 50000 ft = 15240 meters
alt_braking_base = 15240
# braking range 280 nautical miles = 518560 meters 
range_braking_base = 518560

# approach altitude 7000 ft = 2133.6 m
alt_approach_base = 2133.6
# approach range = 8300 ft = 2529.84 m
range_approach_base = 2529.84
# approach altitude rate = -200 fps = 60.96 mps
alt_rate_approach_base = -60.96
# approach velocity = 500 fps = 152.4 mps
vel_approach_base = 152.4

# landing altitude gate = 500 ft = 152.4 m
alt_land_base = 152.4
# landing altitude rate = -3 fps = -0.9144 mps
alt_rate_land_base = -0.9144

#####################################################

# Comparison of lunar landing trajectory strategies using numirical simulations
descent_approach_az = np.pi/4 # rad

# LLO parking orbit 60 nautical miles = 111120 meters
alt_parking_orbit = 111120
# braking altitude 50000 ft = 15240 meters
alt_braking = 15240
# braking range 280 nautical miles = 518560 meters 
range_braking = 518560

# approach altitude 7000 ft = 2133.6 m
alt_approach = 2133.6
# approach range = 8300 ft = 2529.84 m
range_approach = 2529.84
# approach altitude rate = -200 fps = 60.96 mps
alt_rate_approach = -60.96
# approach velocity = 500 fps = 152.4 mps
vel_approach = 152.4

# landing altitude gate = 500 ft = 152.4 m
alt_land = 152.4
# landing altitude rate = -3 fps = -0.9144 mps
alt_rate_land = -0.9144

# calculate velocity at braking initialization
# assume a velocity according to altitude according to optimal Hohman xfer
r_a = Moon.radius + alt_parking_orbit
r_p = Moon.radius + alt_braking
sma = 1/2*(r_a+r_p)
velocity_braking = np.sqrt(Moon.mu*(2/r_p-1/sma))
print(velocity_braking)

rx_brk_0 = np.cos(descent_approach_az)*range_braking
ry_brk_0 = np.sin(descent_approach_az)*range_braking
rz_brk_0 = alt_braking
range_rate_brk_0 = velocity_braking
vx_brk_0 = -np.cos(descent_approach_az)*range_rate_brk_0
vy_brk_0 = -np.sin(descent_approach_az)*range_rate_brk_0
vz_brk_0 = 0 # no radial velocity at perilune

rx_app_0 = np.cos(descent_approach_az)*range_approach
ry_app_0 = np.sin(descent_approach_az)*range_approach
rz_app_0 = alt_approach
range_rate_app_0 = np.sqrt(vel_approach**2 - alt_rate_approach**2)
vx_app_0 = -np.cos(descent_approach_az)*range_rate_app_0
vy_app_0 = -np.sin(descent_approach_az)*range_rate_app_0
vz_app_0 = alt_rate_approach


def get_dispersed():
    # Comparison of lunar landing trajectory strategies using numirical simulations
    descent_approach_az = np.random.normal(descent_approach_az_base, np.pi/4)

    # LLO parking orbit 60 nautical miles = 111120 meters
    alt_parking_orbit = np.random.normal(alt_parking_orbit_base, 50000)
    # braking altitude 50000 ft = 15240 meters
    alt_braking = np.random.normal(alt_braking_base, 2000)
    # braking range 280 nautical miles = 518560 meters 
    range_braking = np.random.normal(range_braking_base, 40000)

    # calculate velocity at braking initialization
    # assume a velocity according to altitude according to optimal Hohman xfer
    r_a = Moon.radius + alt_parking_orbit
    r_p = Moon.radius + alt_braking
    sma = 1/2*(r_a+r_p)
    velocity_braking = np.sqrt(Moon.mu*(2/r_p-1/sma))
    print(velocity_braking)

    rx_brk_0 = np.cos(descent_approach_az)*range_braking
    ry_brk_0 = np.sin(descent_approach_az)*range_braking
    rz_brk_0 = alt_braking
    range_rate_brk_0 = velocity_braking
    vx_brk_0 = -np.cos(descent_approach_az)*range_rate_brk_0
    vy_brk_0 = -np.sin(descent_approach_az)*range_rate_brk_0
    vz_brk_0 = 0 # no radial velocity at perilune

    rx_app_0 = np.cos(descent_approach_az)*range_approach
    ry_app_0 = np.sin(descent_approach_az)*range_approach
    rz_app_0 = alt_approach
    range_rate_app_0 = np.sqrt(vel_approach**2 - alt_rate_approach**2)
    vx_app_0 = -np.cos(descent_approach_az)*range_rate_app_0
    vy_app_0 = -np.sin(descent_approach_az)*range_rate_app_0
    vz_app_0 = alt_rate_approach

    return descent_approach_az, alt_parking_orbit, alt_braking, range_braking, rx_brk_0, ry_brk_0, rz_brk_0, range_rate_brk_0, vx_brk_0, vy_brk_0, vz_brk_0,rx_app_0, ry_app_0, rz_app_0, range_rate_app_0, vx_app_0, vy_app_0, vz_app_0