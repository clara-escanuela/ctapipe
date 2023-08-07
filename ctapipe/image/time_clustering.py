from abc import abstractmethod

import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

from ctapipe.image.cleaning import dilate
from ctapipe.image.extractor import deconvolve

from ..core import TelescopeComponent
from ..core.traits import (
    BoolTelescopeParameter,
    FloatTelescopeParameter,
    IntTelescopeParameter,
)

__all__ = ["WaveformCleaner", "WaveformVolumeReducer"]


def get_cluster(subarray, broken_pixels, tel_id, waveform, cut):
    traces = waveform
    diff_traces = deconvolve(traces, 0.0, 4, 1.0)[~broken_pixels]

    snr = (
        diff_traces
        / np.array(subarray.tel[tel_id].camera.noise)[~broken_pixels][:, None]
    )
    x_pos = np.array(subarray.tel[tel_id].camera.geometry.pix_x)[~broken_pixels][
        :, None
    ] * np.ones(len(snr[0]))
    y_pos = np.array(subarray.tel[tel_id].camera.geometry.pix_y)[~broken_pixels][
        :, None
    ] * np.ones(len(snr[0]))
    pix_id = np.array(subarray.tel[tel_id].camera.geometry.pix_id)[~broken_pixels][
        :, None
    ] * np.ones(len(snr[0]))

    time = np.array([])
    x = np.array([])
    y = np.array([])
    pix_ids = np.array([])
    snrs = np.array([])
    for i in range(len(snr)):
        local_max_pos = find_peaks(snr[i])[0]
        local_max = snr[i][local_max_pos]
        pos = local_max_pos[local_max > cut]

        x = np.append(x, x_pos[i][pos])
        y = np.append(y, y_pos[i][pos])
        pix_ids = np.append(pix_ids, pix_id[i][pos])
        time = np.append(time, pos)

        snrs = np.append(snrs, snr[i][pos])

    return time, x, y, pix_ids, snrs


PIXEL_SPACING = {
    "LSTCam": 0.05,
    "NectarCam": 0.05,
    "FlashCam": 0.05,
    "SST-Camera": 0.5,
    "CHEC": 0.05,
    "DUMMY": 0.05,
    "testcam": 0.05,
}


def time_clustering(
    subarray,
    tel_id,
    r0_waveform,
    broken_pixels,
    cut=3.5,
    n_min=5,
    dd=1.0,
    t_scale=4.0,
    d_scale=2.5,
    rows=0.0,
    scale=4.0,
    shift=1.5,
    n_norm=2.0,
):

    time, x, y, pix_ids, snrs = get_cluster(
        subarray, broken_pixels, tel_id, r0_waveform, cut
    )
    geom = subarray.tel[tel_id].camera.geometry

    arr = np.zeros(len(time), dtype=float)
    pix_arr = -np.ones(geom.n_pixels, dtype=int)

    pix_x = x / (d_scale * PIXEL_SPACING[geom.name])
    pix_y = y / (d_scale * PIXEL_SPACING[geom.name])

    X = np.column_stack((time / t_scale, pix_x, pix_y))

    db = DBSCAN(eps=dd, min_samples=n_min).fit(
        X, sample_weight=n_norm / (1 + np.exp(-(snrs + shift) / scale))
    )
    labels = db.labels_

    ret_labels = -np.ones(geom.n_pixels, dtype=int)
    ret_labels[np.array(pix_ids)[np.array(arr) == 0].astype(int)] = labels

    arr[(labels == -1)] = -1
    pix_arr[np.array(pix_ids)[np.array(arr) == 0].astype(int)] = 0

    mask = pix_arr == 0  # we keep these events
    for _ in range(int(rows)):
        mask = dilate(geom, mask)

    return mask, ret_labels


"""
Image Cleaning
"""


class WaveformCleaner(TelescopeComponent):
    """
    Abstract class for all configurable Image Cleaning algorithms.   Use
    ``ImageCleaner.from_name()`` to construct an instance of a particular algorithm
    """

    @abstractmethod
    def __call__(
        self, tel_id: int, waveform: np.ndarray, broken_pixels: np.ndarray = None
    ) -> np.ndarray:
        """
        Identify pixels with signal, and reject those with pure noise.
        Parameters
        ----------
        tel_id: int
            which telescope id in the subarray is being used (determines
            which cut is used)
        image : np.ndarray
            image pixel data corresponding to the camera geometry
        arrival_times: np.ndarray
            image of arrival time (not used in this method)
        Returns
        -------
        np.ndarray
            boolean mask of pixels passing cleaning
        """
        pass


class TimeCleaner(WaveformCleaner):
    """
    Clean images using the standard picture/boundary technique. See
    `ctapipe.image.tailcuts_clean`
    """

    cut = FloatTelescopeParameter(
        default_value=3.5, help="top-level threshold in photoelectrons"
    ).tag(config=True)
    dd = FloatTelescopeParameter(
        default_value=1.0, help="top-level threshold in photoelectrons"
    ).tag(config=True)

    rows = FloatTelescopeParameter(
        default_value=0.0, help="second-level threshold in photoelectrons"
    ).tag(config=True)

    t_scale = FloatTelescopeParameter(
        default_value=4.0,
        help="Minimum number of neighbors above threshold to consider",
    ).tag(config=True)

    d_scale = FloatTelescopeParameter(
        default_value=2.5,
        help="Minimum number of neighbors above threshold to consider",
    ).tag(config=True)

    n_min = IntTelescopeParameter(
        default_value=5, help="Minimum number of neighbors above threshold to consider"
    ).tag(config=True)

    scale = FloatTelescopeParameter(
        default_value=4.0,
        help="Scale for weighting",
    ).tag(config=True)

    shift = FloatTelescopeParameter(
        default_value=1.5,
        help="Shift for weighting",
    ).tag(config=True)

    n_norm = FloatTelescopeParameter(
        default_value=2.0,
        help="Scale for weighting",
    ).tag(config=True)

    def __call__(
        self, tel_id: int, waveform: np.ndarray, broken_pixels: np.ndarray = None
    ) -> np.ndarray:
        """
        Apply standard picture-boundary cleaning. See `ImageCleaner.__call__()`
        """

        mask, labels = time_clustering(
            self.subarray,
            tel_id,
            waveform,
            broken_pixels,
            cut=self.cut.tel[tel_id],
            n_min=self.n_min.tel[tel_id],
            dd=self.dd.tel[tel_id],
            rows=self.rows.tel[tel_id],
            t_scale=self.t_scale.tel[tel_id],
            d_scale=self.d_scale.tel[tel_id],
            scale=self.scale.tel[tel_id],
            shift=self.shift.tel[tel_id],
            n_norm=self.n_norm.tel[tel_id],
        )

        return mask


"""
Data Volume Reduction
"""


class WaveformVolumeReducer(TelescopeComponent):
    """
    Base component for data volume reducers.
    """

    def __init__(self, subarray, config=None, parent=None, **kwargs):
        """
        Parameters
        ----------
        subarray: ctapipe.instrument.SubarrayDescription
            Description of the subarray
        config: traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        kwargs
        """
        self.subarray = subarray
        super().__init__(config=config, parent=parent, subarray=subarray, **kwargs)

    def __call__(
        self, waveforms, broken_pixels=None, tel_id=None, selected_gain_channel=None
    ):
        """
        Call the relevant functions to perform data volume reduction on the
        waveforms.
        Parameters
        ----------
        waveforms: ndarray
            Waveforms stored in a numpy array of shape
            (n_pix, n_samples).
        tel_id: int
            The telescope id. Required for the 'image_extractor' and
            'camera.geometry' in 'TailCutsDataVolumeReducer'.
        selected_gain_channel: ndarray
            The channel selected in the gain selection, per pixel. Required for
            the 'image_extractor' in 'TailCutsDataVolumeReducer'.
            extraction.
        Returns
        -------
        mask: array
            Mask of selected pixels.
        """
        mask = self.select_pixels(
            waveforms,
            broken_pixels=broken_pixels,
            tel_id=tel_id,
            selected_gain_channel=selected_gain_channel,
        )
        return mask

    @abstractmethod
    def select_pixels(
        self, waveforms, broken_pixels=None, tel_id=None, selected_gain_channel=None
    ):
        """
        Abstract method to be defined by a DataVolumeReducer subclass.
        Call the relevant functions for the required pixel selection.
        Parameters
        ----------
        waveforms: ndarray
            Waveforms stored in a numpy array of shape
            (n_pix, n_samples).
        tel_id: int
            The telescope id. Required for the 'image_extractor' and
            'camera.geometry' in 'TailCutsDataVolumeReducer'.
        selected_gain_channel: ndarray
            The channel selected in the gain selection, per pixel. Required for
            the 'image_extractor' in 'TailCutsDataVolumeReducer'.
        Returns
        -------
        mask: array
            Mask of selected pixels.
        """


class ClusteringWfDataVolumeReducer(WaveformVolumeReducer):

    n_end_dilates = IntTelescopeParameter(
        default_value=1, help="Number of how many times to dilate at the end."
    ).tag(config=True)

    do_boundary_dilation = BoolTelescopeParameter(
        default_value=True,
        help="If set to 'False', the iteration steps in 2) are skipped and"
        "normal TailcutCleaning is used.",
    ).tag(config=True)

    def __init__(
        self,
        subarray,
        config=None,
        parent=None,
        cleaner=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        subarray: ctapipe.instrument.SubarrayDescription
            Description of the subarray
        config: traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        kwargs
        """
        super().__init__(config=config, parent=parent, subarray=subarray, **kwargs)

        if cleaner is None:
            self.cleaner = TimeCleaner(parent=self, subarray=self.subarray)
        else:
            self.cleaner = cleaner

    def select_pixels(
        self, waveforms, broken_pixels=None, tel_id=None, selected_gain_channel=None
    ):
        camera_geom = self.subarray.tel[tel_id].camera.geometry

        # 1) Step: Clustering cleaning
        mask = self.cleaner(tel_id, waveforms, broken_pixels=broken_pixels)

        # 2) Step: Adding Pixels with 'dilate' to get more conservative.
        for _ in range(self.n_end_dilates.tel[tel_id]):
            mask = dilate(camera_geom, mask)

        return mask
