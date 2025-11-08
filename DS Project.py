# DS Project 
# Corey Miles

'''
Creat a model that successfully captures the motion of a torque powered pirate ship amusement park ride.
'''

# import necessary libraries
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Affine2D
from pathlib import Path
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


# define parameters
m = 5000  # kg
L = 14.5  # m
b = 20205  # N*m*s
g = 9.81  # m/s**2
t_motor_off = 120 # s
ride_time = 220  # s
tire_T = -20000  # N*m
brake_b = b*6  # N*m*s

# motor on ODE
def motor_on(t, z, torque):

    # define state
    theta = z[0]
    theta_dot = z[1]

    # forcing function
    if np.sign(theta_dot) != np.sign(torque):
        torque = -torque

    # equations
    z0_dot = theta_dot
    if abs(theta) <= np.radians(5):
        z1_dot = (torque/(m*L)) - (b/(m*L**2))*theta_dot - (g/L)*np.sin(theta)
    else:
        z1_dot = - (b/(m*L**2))*theta_dot - (g/L)*np.sin(theta)

    # return derivatives
    return [z0_dot, z1_dot]

def motor_off(t, z):

    # define state
    theta = z[0]
    theta_dot = z[1]

    # equations
    z0_dot = theta_dot
    z1_dot = - (brake_b/(m*L**2))*theta_dot - (g/L)*np.sin(theta)

    # return derivatives
    return [z0_dot, z1_dot]

# motor on initial conditions
mon_theta0 = [0, 0]

# time spans
mon_t = (0, t_motor_off)
moff_t = (t_motor_off, ride_time)
mon_t_span = np.linspace(0, t_motor_off, 1000)
moff_t_span = np.linspace(t_motor_off, ride_time, 1000)

# solve motor on ODE
mon_sol = solve_ivp(motor_on, mon_t, mon_theta0, t_eval = mon_t_span, args=(tire_T,))

# unpack motor on solutions
mon_time = mon_sol.t
mon_ap = mon_sol.y[0]
mon_av = mon_sol.y[1]

# motor off initial conditions
moff_theta0 = [mon_ap[-1], mon_av[-1]]

# solve motor off ODE
moff_sol = solve_ivp(motor_off, moff_t, moff_theta0, t_eval=moff_t_span)

# unpack motor off solutions
moff_time = moff_sol.t
moff_ap = moff_sol.y[0]
moff_av = moff_sol.y[1]

# combine motor on and motor off results
time = np.concatenate((mon_time, moff_time))
ap = np.degrees(np.concatenate((mon_ap, moff_ap)))
av = np.degrees(np.concatenate((mon_av, moff_av)))
x = L * np.sin(np.radians(ap)); y = -L * np.cos(np.radians(ap))

# plot results
plt.figure()
plt.plot(time, ap)
plt.xlabel("Time (s)", fontsize=16)
plt.xlim(0, (ride_time))
plt.ylabel("Angular Position (\u00B0)", fontsize=16)

# animate motion
ship_width, ship_height = 10.0, 3.5
post_width = 0.7
post_height = L + 2.0
post_offset = 3.5
crossbar_thickness = 0.6
pivot_radius = 0.45

fig, ax = plt.subplots(figsize=(8,6))
ax.set_aspect('equal', 'box')
max_extent = L + 6.0
ax.set_xlim(-max_extent, max_extent)
ax.set_ylim(-max_extent-2, 4)
ax.axis('off')
ax.set_facecolor("#87CEEB")

left_post = Rectangle(
    ( -post_offset - post_width/2, -post_height ),
    post_width, post_height,
    zorder=0, facecolor="#6b4f2b", ec="black", lw=0.8
)
right_post = Rectangle(
    ( post_offset - post_width/2, -post_height ),
    post_width, post_height,
    zorder=0, facecolor="#6b4f2b", ec="black", lw=0.8
)
ax.add_patch(left_post)
ax.add_patch(right_post)

crossbar = Rectangle(
    (-post_offset-1.2, -crossbar_thickness/2),
    2*(post_offset+1.2), crossbar_thickness,
    zorder=1, facecolor="#444444", ec="black", lw=0.8
)
ax.add_patch(crossbar)

pivot = Circle((0, 0), pivot_radius, facecolor="#222222", edgecolor="k", zorder=3)
ax.add_patch(pivot)

pivot_line, = ax.plot([0, x[0]], [0, y[0]], lw=5, color="#222222", zorder=4)

ship_rect = Rectangle((-ship_width/2, -ship_height/2), ship_width, ship_height,
                      zorder=5, ec='k', lw=1.2, facecolor="#c0392b", alpha=0.95)
ax.add_patch(ship_rect)

ax.plot([-max_extent, max_extent], [-post_height, -post_height], lw=2, color="#2c3e50", zorder=0)

time_text = ax.text(-max_extent+0.5, 3.0, '', fontsize=12, zorder=6)

def init():
    pivot_line.set_data([0, x[0]], [0, y[0]])
    ship_rect.set_xy((x[0]-ship_width/2, y[0]-ship_height/2))
    ship_rect.set_transform(ax.transData)
    time_text.set_text('')
    return pivot_line, ship_rect, time_text

def update(i):
    pivot_line.set_data([0, x[i]], [0, y[i]])
    angle = 0
    trans = Affine2D().rotate_around(0, 0, angle).translate(x[i], y[i]) + ax.transData
    ship_rect.set_transform(trans)
    ship_rect.set_xy((-ship_width/2, -ship_height/2))
    time_text.set_text(f"t={time[i]:.1f}s  θ={ap[i]:.2f}°")
    return pivot_line, ship_rect, time_text

ani = animation.FuncAnimation(fig, update, frames=range(0, len(time), 4), init_func=init, blit=True, interval=25)

out = Path(r"C:\Users\corey\Documents\Fall 2025\ME EN 335 DS\DS Project/pirate_ship.gif")
ani.save(out, writer='pillow', fps=30)

# show plots
plt.show()