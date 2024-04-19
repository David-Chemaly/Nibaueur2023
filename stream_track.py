import numpy as np
from scipy.interpolate import interp1d

# Ensure that the unwrapped angles are monotonically increasing
def unwrap(angles):
    arg_decrease = np.where( np.diff(angles) < 0 )[0]
    for i in arg_decrease:
        angles[i+1:] += 2 * np.pi

    return angles

def histedges_equalN(x, nb_bin):
    """
    Creates bins containing (approximately) the same number of points in each bin.
    Input: x (array of float): values for which you want to get the bins containing the same number of points
           nb_bins (int): number of bins you want
    Output: bins (array of float): values of the edges of the bins that contain approximately the same number of points
    """
    nb_pt = len(x)
    bins = np.interp(np.linspace(0, nb_pt, nb_bin + 1),
                     np.arange(nb_pt),
                     np.sort(x))
    return bins

def get_average_column(x, y, type_bins='constant', n_bins=100):
    xx = x.flatten()
    yy = y.flatten()

    arg_keep = np.array(np.where(yy != 0))[0]
    xx_keep  = xx[arg_keep]
    yy_keep  = yy[arg_keep]

    if type_bins == 'linear':
        bins = np.linspace(xx[0], xx.max(), n_bins + 1)
    elif type_bins == 'constant':
        bins = histedges_equalN(xx_keep, n_bins)

    yy_mean = []
    for index, i in enumerate(bins[:-1]):
        arg_bin = np.where( (i <= xx_keep) & (xx_keep <=bins[index+1]))[0]
        yy_mean.append( np.median(yy_keep[arg_bin]) )
    yy_mean = np.array(yy_mean)
    xx_mean = bins[:-1] + np.diff(bins)/2

    # Make sure no nan in final output
    arg_nan = np.isnan(yy_mean)
    xx_mean = xx_mean[arg_nan == 0]
    yy_mean = yy_mean[arg_nan == 0]

    return xx_mean, yy_mean

class average_stream():

    def __init__(self, stream_pos_N, gamma, type_bins='constant', n_bins=100):
        self.stream_pos_N = stream_pos_N
        self.gamma  = gamma
        self.n_bins = n_bins
        self.type_bins = type_bins

    def forward(self):

        arg_order = np.argsort(self.gamma)

        ordered_gamma = self.gamma[arg_order]
        ordered_stream_pos_N = self.stream_pos_N[arg_order]

        x = ordered_stream_pos_N[:,0]
        y = ordered_stream_pos_N[:,1]

        mean_gamma, mean_x = get_average_column(ordered_gamma, x.value, self.type_bins, self.n_bins)

        f_y = interp1d(ordered_gamma, y)
        mean_y = f_y(mean_gamma)

        return mean_x, mean_y

class average_orbit():

    def __init__(self, orbit_pos_N, arg, n_orbits=11, n_bins=100):
        self.orbit_pos_N = orbit_pos_N
        self.arg         = arg
        self.n_orbits    = n_orbits
        self.n_bins      = n_bins
    

    def get_polar_coord(self):

        keep_arg    = np.linspace(0, len(self.arg), self.n_orbits)
        leading_idx = np.array(self.arg)[keep_arg.astype(int)[:-1]]

        x_leading = self.orbit_pos_N[:,leading_idx, 0].T.value
        y_leading = self.orbit_pos_N[:,leading_idx, 1].T.value

        r_leading_corrected = np.sqrt(x_leading**2 + y_leading**2) 
        first = np.where( r_leading_corrected[0] != 0)[0][0]

        x_leading = self.orbit_pos_N[first:,leading_idx, 0].T.value
        y_leading = self.orbit_pos_N[first:,leading_idx, 1].T.value

        r_leading_corrected = np.sqrt(x_leading**2 + y_leading**2) 

        a = np.sign(x_leading[0, 0])
        b = np.sign(y_leading[0, 1] - y_leading[0, 0])

        x_leading_corrected = a * x_leading
        y_leading_corrected = b * y_leading 

        theta_leading_corrected = np.arctan2(y_leading_corrected, x_leading_corrected)

        theta_leading_unwrapped = np.zeros(theta_leading_corrected.shape)
        for i in range(self.n_orbits-1):
            theta_leading_unwrapped[i] = unwrap(theta_leading_corrected[i])
        for i in range(self.n_orbits-2):
            if np.diag(theta_leading_unwrapped[:,leading_idx])[i+1] < np.diag(theta_leading_unwrapped[:,leading_idx])[i]:
                theta_leading_unwrapped[i+1:] += 2*np.pi

        
        theta_0 = theta_leading_unwrapped[0, 0]
        theta_leading_unwrapped -= theta_0

        return r_leading_corrected, theta_leading_unwrapped, a, b, theta_0
    
    def forward(self):
        r, theta, a, b, theta_0 = self.get_polar_coord()
        theta_mean, r_mean = get_average_column(theta, r)

        x_mean = r_mean * np.cos(theta_mean + theta_0) / a
        y_mean = r_mean * np.sin(theta_mean + theta_0) / b

        return x_mean, y_mean
