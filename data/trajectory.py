"""Code to analyse trajectory, speed, etc. """
import numpy as np

import scipy.optimize as optimize
import functools


def general_lens_correction(xy, k, f, l=0.75, xy_c=(1023.5, 1023.5),
                            XY_c=(0.6, 0.6)):
    '''This function computes the correction for any type of lens. The fisheye
    factor k modifies the type of distortion / correction.
    see PTGui 11 fisheye in https://wiki.panotools.org/Fisheye_Projection

    xy are pixel coordinates on the image in px
    XY are real world coordinates in m

    xy_c are the coordinates of the center of the image in px
    XY_c are the coordinates of the center of the real world (i.e. the setup)

    l = 0.75 is default because the lens was exactly 0.75 m from the platforms

    k has to be > 1449 or else np.arctan would return an invalid number

    k should be between 0 and 1
    '''
    x, y = xy

    # we first center x and y
    x_c = x - xy_c[0]
    y_c = y - xy_c[1]

    # now we convert to polar coordinates in the image
    r = np.sqrt(x_c**2 + y_c**2)
    theta = np.arctan2(y_c, x_c)

    # R is the real world polar coordinate R
    if -1 <= k and k < 0:
        R = l * np.tan((-np.arcsin((k*r)/f)+np.pi)/k)
    elif 0 < k and k <= 1:
        R = l * np.tan(np.arctan((k*r)/f)/k)
    elif k == 0:
        R = l * np.tan(r/f)
    else:
        raise Exception('invalid k value. k must be a number between -1 and 1')

    # now we calculate the real world centered XY coordinates
    X = R * np.cos(theta)
    Y = R * np.sin(theta)

    # now we put it back into axes with the origin in the corner of the real
    # world, i.e. decenter again
    X += XY_c[0]
    Y += XY_c[1]
    return (X, Y)


##
# Discover correction coefficients for the lens
def loss_function(k, f, platform_centroid_px_coords_dic):
    '''This loss function is what we will minimize over r to find the correct
    radius of the imaginary sphere that undistorts the images.
    We will compute straightness of horizontal and vertical lines by taking
    the sum squared distance of each point in the line to the middle of it.
    We then take the sum accross all vertical and horizontal lines.
    '''
    # We know that the center of the outermost tower is at (15cm, 15cm).
    # since the towers are 10cm wide and the corridors 8cm long, then we know
    # the center coordinates for all towers in a row as 15 + n * 18
    # We are considering a square of 120cm * 120cm which is the size of the
    # maze
    #
    # The Y axis is "flipped", i.e., Y_(:,6) < Y_(:,5) < ... < Y_(:0) but the
    # X is not, i.e. X_(:,6) > X_(:,5) > ... > X_(:0) but the
    xy_coord_m = lambda i, j: (0.15 + i * 0.18, 1.2 - (0.15 + j * 0.18))

    # we initialize the Root Mean Square Error, which will be
    # RMSE = np.sqrt((1/36) * \Sum{Euclidian error**2})

    MSE = 0
    for i in range(6):
        # dim==0 is the x axis; dim==1 is the y axis
        for j in range(6):
            # we then get the y coord of the centroid of each port for rows and
            # the x coord of the centroid of each port for columns
            xy = platform_centroid_px_coords_dic[(i, j)]

            projected_xy = general_lens_correction(xy, k, f)

            MSE += np.sqrt((xy_coord_m(i, j)[0]-projected_xy[0])**2 + (
                xy_coord_m(i, j)[1]-projected_xy[1])**2)

    RMSE = np.sqrt(MSE/36)

    return RMSE


def find_optimal_params_correction(platform_centroid_px_coords, k_init=0.5,
                                   f_init=1500):
    partial_loss_function = functools.partial(
        loss_function, platform_centroid_px_coords_dic=platform_centroid_px_coords)
    f = lambda kf: partial_loss_function(kf[0], kf[1])

    initial_guess = np.array([[k_init], [f_init]])
    result = optimize.minimize(f, initial_guess, bounds=((0,0.999),(0, None)))
    k_optimal, f_optimal = result.x

    return k_optimal, f_optimal
