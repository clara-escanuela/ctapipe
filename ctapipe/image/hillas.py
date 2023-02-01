# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: UTF-8 -*-
"""
Hillas-style moment-based shower image parametrization.
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from ctapipe.image.cleaning import dilate
import scipy.optimize as opt

from ctapipe.image.pixel_likelihood import chi_squared

from ..containers import CameraHillasParametersContainer, HillasParametersContainer

HILLAS_ATOL = np.finfo(np.float64).eps


__all__ = ["hillas_parameters", "HillasParameterizationError"]


def camera_to_shower_coordinates(x, y, cog_x, cog_y, psi):
    """
    Return longitudinal and transverse coordinates for x and y
    for a given set of hillas parameters

    Parameters
    ----------
    x: u.Quantity[length]
        x coordinate in camera coordinates
    y: u.Quantity[length]
        y coordinate in camera coordinates
    cog_x: u.Quantity[length]
        x coordinate of center of gravity
    cog_y: u.Quantity[length]
        y coordinate of center of gravity
    psi: Angle
        orientation angle

    Returns
    -------
    longitudinal: astropy.units.Quantity
        longitudinal coordinates (along the shower axis)
    transverse: astropy.units.Quantity
        transverse coordinates (perpendicular to the shower axis)
    """
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    delta_x = x - cog_x
    delta_y = y - cog_y

    longi = delta_x * cos_psi + delta_y * sin_psi
    trans = delta_x * -sin_psi + delta_y * cos_psi

    return longi, trans


class HillasParameterizationError(RuntimeError):
    pass


def hillas_parameters(geometry, dl1_image, cleaned_mask, pedestal_variance, gaussian=True):
    """
    Compute Hillas parameters for a given shower image.

    Implementation uses a PCA analogous to the implementation in
    src/main/java/fact/features/HillasParameters.java
    from
    https://github.com/fact-project/fact-tools

    The recommended form is to pass only the sliced geometry and image
    for the pixels to be considered.

    Each method gives the same result, but vary in efficiency

    Parameters
    ----------
    geom: ctapipe.instrument.CameraGeometry
        Camera geometry, the cleaning mask should be applied to improve performance
    image : array_like
        Charge in each pixel, the cleaning mask should already be applied to
        improve performance.

    Returns
    -------
    HillasParametersContainer:
        container of hillas parametesr
    """
    geom = geometry[cleaned_mask]
    unit = geom.pix_x.unit
    pix_x = geom.pix_x.to_value(unit)
    pix_y = geom.pix_y.to_value(unit)
    image = np.asanyarray(dl1_image.copy()[cleaned_mask], dtype=np.float64)

    if isinstance(image, np.ma.masked_array):
        image = np.ma.filled(image, 0)

    if not (pix_x.shape == pix_y.shape == image.shape):
        raise ValueError("Image and pixel shape do not match")

    size = np.sum(image)

    if size == 0.0:
        raise HillasParameterizationError("size=0, cannot calculate HillasParameters")

    # calculate the cog as the mean of the coordinates weighted with the image
    cog_x = np.average(pix_x, weights=image)
    cog_y = np.average(pix_y, weights=image)

    # polar coordinates of the cog
    cog_r = np.linalg.norm([cog_x, cog_y])
    cog_phi = np.arctan2(cog_y, cog_x)

    # do the PCA for the hillas parameters
    delta_x = pix_x - cog_x
    delta_y = pix_y - cog_y

    # The ddof=0 makes this comparable to the other methods,
    # but ddof=1 should be more correct, mostly affects small showers
    # on a percent level
    cov = np.cov(delta_x, delta_y, aweights=image, ddof=0)
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    # round eig_vals to get rid of nans when eig val is something like -8.47032947e-22
    near_zero = np.isclose(eig_vals, 0, atol=HILLAS_ATOL)
    eig_vals[near_zero] = 0

    # width and length are eigen values of the PCA
    width, length = np.sqrt(eig_vals)

    # psi is the angle of the eigenvector to length to the x-axis
    vx, vy = eig_vecs[0, 1], eig_vecs[1, 1]

    # avoid divide by 0 warnings
    # psi will be consistently defined in the range (-pi/2, pi/2)
    if length == 0:
        psi = skewness_long = kurtosis_long = np.nan
    else:
        if vx != 0:
            psi = np.arctan(vy / vx)
        else:
            psi = np.pi / 2

        # calculate higher order moments along shower axes
        longitudinal = delta_x * np.cos(psi) + delta_y * np.sin(psi)

        m3_long = np.average(longitudinal**3, weights=image)
        skewness_long = m3_long / length**3

        m4_long = np.average(longitudinal**4, weights=image)
        kurtosis_long = m4_long / length**4

    # Compute of the Hillas parameters uncertainties.
    # Implementation described in [hillas_uncertainties]_ This is an internal MAGIC document
    # not generally accessible.

    # intermediate variables
    cos_2psi = np.cos(2 * psi)
    a = (1 + cos_2psi) / 2
    b = (1 - cos_2psi) / 2
    c = np.sin(2 * psi)

    A = ((delta_x**2.0) - cov[0][0]) / size
    B = ((delta_y**2.0) - cov[1][1]) / size
    C = ((delta_x * delta_y) - cov[0][1]) / size

    # Hillas's uncertainties

    # avoid divide by 0 warnings
    if length == 0:
        length_uncertainty = np.nan
    else:
        length_uncertainty = np.sqrt(
            np.sum(((((a * A) + (b * B) + (c * C))) ** 2.0) * image)
        ) / (2 * length)

    if width == 0:
        width_uncertainty = np.nan
    else:
        width_uncertainty = np.sqrt(
            np.sum(((((b * A) + (a * B) + (-c * C))) ** 2.0) * image)
        ) / (2 * width)

    if gaussian == False:

        if unit.is_equivalent(u.m):
            return CameraHillasParametersContainer(
                x=u.Quantity(cog_x, unit),
                y=u.Quantity(cog_y, unit),
                r=u.Quantity(cog_r, unit),
                phi=Angle(cog_phi, unit=u.rad),
                intensity=size,
                length=u.Quantity(length, unit),
                length_uncertainty=u.Quantity(length_uncertainty, unit),
                width=u.Quantity(width, unit),
                width_uncertainty=u.Quantity(width_uncertainty, unit),
                psi=Angle(psi, unit=u.rad),
                skewness=skewness_long,
                kurtosis=kurtosis_long,
            )
        return HillasParametersContainer(
            fov_lon=u.Quantity(cog_x, unit),
            fov_lat=u.Quantity(cog_y, unit),
            r=u.Quantity(cog_r, unit),
            phi=Angle(cog_phi, unit=u.rad),
            intensity=size,
            length=u.Quantity(length, unit),
            length_uncertainty=u.Quantity(length_uncertainty, unit),
            width=u.Quantity(width, unit),
            width_uncertainty=u.Quantity(width_uncertainty, unit),
            psi=Angle(psi, unit=u.rad),
            skewness=skewness_long,
            kurtosis=kurtosis_long,
        )

    #Add rows of pixels to the cleaned image

    n = 2
    m = cleaned_mask.copy()
    for ii in range(n):
        m = dilate(geometry, m)

    mask = np.array((m.astype(int) + cleaned_mask.astype(int)), dtype=bool)
    cleaned_image = dl1_image.copy()
    cleaned_image[~mask] = 0.0
    cleaned_image[cleaned_image<0] = 0.0
    #cleaned_image = cleaned_image[mask]
    #geometry = geometry[mask][cleaned_image>0]
    #pedestal_variance = pedestal_variance[mask][cleaned_image>0]
    #cleaned_image = cleaned_image[cleaned_image>0]
    size = np.sum(cleaned_image)

    def fit(z, xi, yi, image, pedestal_variance, emf=1.585):
        delta_x = (xi - z[0])
        delta_y = (yi - z[1])

        longitudinal = delta_x * np.cos(z[2]) + delta_y * np.sin(z[2])
        transverse = delta_x * np.sin(z[2]) - delta_y * np.cos(z[2])

        x = (longitudinal/z[3])**2
        y = (transverse/z[4])**2

        prediction = (z[5])*np.exp(-(1/2)*(x + y))

        chi_square = chi_squared(cleaned_image, prediction, pedestal_variance, emf)

        return chi_square

    x0 = [cog_x, cog_y, psi, length, width, np.max(cleaned_image)/(2*np.pi*cog_x*cog_y)]

    bnds = ((-1.2, 1.2), (-1.2, 1.2), (-np.pi/2, np.pi/2), (0, 2), (0, 2), (0, np.max(cleaned_image)))

    result = opt.minimize(fit, x0=x0, args=(geometry.pix_x.value, geometry.pix_y.value, cleaned_image, pedestal_variance), bounds=bnds)
    results = result.x

    fit_xcog = results[0]
    fit_ycog = results[1]
    fit_psi = results[2]
    fit_length = results[3]
    fit_width = results[4]

    fit_rcog = np.linalg.norm([fit_xcog, fit_ycog])
    fit_phi = np.arctan2(fit_ycog, fit_xcog)

    if fit_length <= 0 or fit_width <= 0:
        fit_xcog = cog_x
        fit_ycog = cog_y
        fit_rcog = cog_r
        fit_phi = cog_phi
        fit_length = length
        fit_width = width
        fit_psi = psi

    delta_x = geometry.pix_x.value - fit_xcog
    delta_y = geometry.pix_y.value - fit_ycog

    cov = np.cov(delta_x, delta_y, aweights=cleaned_image, ddof=0)
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    longitudinal = delta_x * np.cos(fit_psi) + delta_y * np.sin(fit_psi)

    m3_long = np.average(longitudinal**3, weights=cleaned_image)
    skewness_long = m3_long / fit_length**3

    m4_long = np.average(longitudinal**4, weights=cleaned_image)
    kurtosis_long = m4_long / fit_length**4

    cos_2psi = np.cos(2 * fit_psi)
    a = (1 + cos_2psi) / 2
    b = (1 - cos_2psi) / 2
    c = np.sin(2 * fit_psi)

    A = ((delta_x**2.0) - cov[0][0]) / size
    B = ((delta_y**2.0) - cov[1][1]) / size
    C = ((delta_x * delta_y) - cov[0][1]) / size

    if fit_length == 0:
        length_uncertainty = np.nan
    else:
        length_uncertainty = np.sqrt(
            np.sum(((((a * A) + (b * B) + (c * C))) ** 2.0) * cleaned_image)
        ) / (2 * fit_length)

    if fit_width == 0:
        width_uncertainty = np.nan
    else:
        width_uncertainty = np.sqrt(
            np.sum(((((b * A) + (a * B) + (-c * C))) ** 2.0) * cleaned_image)
                ) / (2 * fit_width)
    
    if unit.is_equivalent(u.m):
        return CameraHillasParametersContainer(
            x=u.Quantity(fit_xcog, unit),
            y=u.Quantity(fit_ycog, unit),
            r=u.Quantity(fit_rcog, unit),
            phi=Angle(fit_phi, unit=u.rad),
            intensity=size,
            length=u.Quantity(fit_length, unit),
            length_uncertainty=u.Quantity(length_uncertainty, unit),
            width=u.Quantity(fit_width, unit),
            width_uncertainty=u.Quantity(width_uncertainty, unit),
            psi=Angle(fit_psi, unit=u.rad),
            skewness=skewness_long,
            kurtosis=kurtosis_long,
            )
    return HillasParametersContainer(
        fov_lon=u.Quantity(fit_xcog, unit),
        fov_lat=u.Quantity(fit_ycog, unit),
        r=u.Quantity(fit_rcog, unit),
        phi=Angle(fit_phi, unit=u.rad),
        intensity=size,
        length=u.Quantity(fit_length, unit),
        length_uncertainty=u.Quantity(length_uncertainty, unit),
        width=u.Quantity(fit_width, unit),
        width_uncertainty=u.Quantity(width_uncertainty, unit),
        psi=Angle(fit_psi, unit=u.rad),
        skewness=skewness_long,
        kurtosis=kurtosis_long,
        )

