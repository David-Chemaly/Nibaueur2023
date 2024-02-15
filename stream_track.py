import numpy as np
import pandas as pd

# Ensure that the unwrapped angles are monotonically increasing
def unwrap(angles):
    arg_decrease = np.where( np.diff(angles) <= 0 )[0]
    for i in arg_decrease:
        angles[i+1:] += 2 * np.pi

    return angles

# Get average of a column
def average_column(x,y):
    bins = np.linspace(np.min(x), np.max(x), 101)

    # Bin the x array
    binned_x = np.digitize(x, bins)

    # Create a DataFrame for easy calculation
    df = pd.DataFrame({'x': x, 'y': y, 'bin': binned_x})

    # Group by the bin and calculate mean
    average_y_per_bin = df.groupby('bin')['y'].mean()

    return average_y_per_bin.to_numpy()

# Get the track from the orbits
def get_track_from_orbits(x_pos, y_pos, args):

    first_theta = []
    for index, i in enumerate(args):
        x = x_pos[index, i//2:]
        y = y_pos[index, i//2:]
        first_theta.append(np.arctan2(y[0], x[0]))
    adjust_theta = unwrap( first_theta - first_theta[0])

    x_all = []
    y_all = []
    r_all = []
    theta_all = []
    for index, i in enumerate(args):
        x = x_pos[index, i//2:]
        y = y_pos[index, i//2:]
        x_all.extend(x)
        y_all.extend(y)
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta_shifted = theta - theta[0]
        theta_unwrapped = unwrap( theta_shifted + adjust_theta[index])

        r_all.extend(r)
        theta_all.extend(theta_unwrapped)

    r_mean = average_column(theta_all, r_all)
    theta_mean = np.linspace(np.min(theta_all), np.max(theta_all), len(r_mean))
    theta_norm = (theta_mean + first_theta[0] + np.pi) % (2 * np.pi) - np.pi 

    x_mean = r_mean*np.cos(theta_norm)
    y_mean = r_mean*np.sin(theta_norm)

    return x_mean, y_mean

