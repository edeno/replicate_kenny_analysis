from logging import getLogger

import numpy as np
import pandas as pd
from loren_frank_data_processing import (get_all_multiunit_indicators,
                                         get_all_spike_indicators,
                                         get_interpolated_position_dataframe,
                                         get_LFPs, get_trial_time,
                                         make_neuron_dataframe,
                                         make_tetrode_dataframe)
from ripple_detection import (Kay_ripple_detector, filter_ripple_band,
                              get_multiunit_population_firing_rate)

from spectral_rhythm_detector import detect_spectral_rhythm
from src.parameters import _MARKS, ANIMALS, BRAIN_AREAS, SAMPLING_FREQUENCY

logger = getLogger(__name__)


def get_theta_times(epoch_key, sampling_frequency=1500):
    THETA_BAND = (6, 12)
    TIME_WINDOW_STEP = TIME_WINDOW_DURATION = 0.300
    TIME_HALFBANDWIDTH_PRODUCT = 1

    position_info = (
        get_interpolated_position_dataframe(epoch_key, ANIMALS)
        .dropna(subset=['linear_distance', 'linear_speed']))
    time = position_info.index
    tetrode_info = make_tetrode_dataframe(ANIMALS, epoch_key=epoch_key)
    tetrode_keys = tetrode_info.loc[tetrode_info.area == 'Reference'].index

    lfps = get_LFPs(tetrode_keys, ANIMALS).reindex(time)

    multitaper_params = dict(
        time_halfbandwidth_product=TIME_HALFBANDWIDTH_PRODUCT,
        time_window_duration=TIME_WINDOW_DURATION,
        time_window_step=TIME_WINDOW_STEP,
        start_time=(time.values / np.timedelta64(1, 's')).min(),
    )

    df, model = detect_spectral_rhythm(
        time=time.values / np.timedelta64(1, 's'),
        lfps=lfps.values,
        sampling_frequency=sampling_frequency,
        multitaper_params=multitaper_params,
        frequency_band=THETA_BAND
    )

    return df.is_spectral_rhythm


def get_ripple_times2(epoch_key, sampling_frequency=1500,
                      brain_areas=BRAIN_AREAS):
    RIPPLE_BAND = (150, 250)
    TIME_WINDOW_STEP = TIME_WINDOW_DURATION = 0.020
    TIME_HALFBANDWIDTH_PRODUCT = 1

    position_info = (
        get_interpolated_position_dataframe(epoch_key, ANIMALS)
        .dropna(subset=['linear_distance', 'linear_speed']))
    time = position_info.index
    tetrode_info = make_tetrode_dataframe(ANIMALS).xs(
        epoch_key, drop_level=False)
    if ~np.all(np.isnan(tetrode_info.validripple.astype(float))):
        tetrode_keys = tetrode_info.loc[
            (tetrode_info.validripple == 1)].index
    else:
        is_brain_areas = (
            tetrode_info.area.astype(str).str.upper().isin(brain_areas))
        tetrode_keys = tetrode_info.loc[is_brain_areas].index

    lfps = get_LFPs(tetrode_keys, ANIMALS).reindex(time)

    multitaper_params = dict(
        time_halfbandwidth_product=TIME_HALFBANDWIDTH_PRODUCT,
        time_window_duration=TIME_WINDOW_DURATION,
        time_window_step=TIME_WINDOW_STEP,
        start_time=(time.values / np.timedelta64(1, 's')).min(),
    )

    df, model = detect_spectral_rhythm(
        time=time.values / np.timedelta64(1, 's'),
        lfps=lfps.values,
        sampling_frequency=sampling_frequency,
        multitaper_params=multitaper_params,
        frequency_band=RIPPLE_BAND
    )

    return df.is_spectral_rhythm


def get_ripple_times(epoch_key, sampling_frequency=1500,
                     brain_areas=BRAIN_AREAS):
    position_info = (
        get_interpolated_position_dataframe(epoch_key, ANIMALS)
        .dropna(subset=['linear_distance', 'linear_speed']))
    speed = position_info['speed']
    time = position_info.index
    tetrode_info = make_tetrode_dataframe(ANIMALS, epoch_key=epoch_key)
    if ~np.all(np.isnan(tetrode_info.validripple.astype(float))):
        tetrode_keys = tetrode_info.loc[
            (tetrode_info.validripple == 1)].index
    else:
        is_brain_areas = (
            tetrode_info.area.astype(str).str.upper().isin(brain_areas))
        tetrode_keys = tetrode_info.loc[is_brain_areas].index

    lfps = get_LFPs(tetrode_keys, ANIMALS).reindex(time)
    return Kay_ripple_detector(
        time, lfps.values, speed.values, sampling_frequency,
        zscore_threshold=2.0, close_ripple_threshold=np.timedelta64(0, 'ms'),
        minimum_duration=np.timedelta64(15, 'ms'))


def load_data(epoch_key, brain_areas=None):

    if brain_areas is None:
        brain_areas = BRAIN_AREAS

    time = get_trial_time(epoch_key, ANIMALS)
    time = (pd.Series(np.ones_like(time, dtype=np.float), index=time)
            .resample('2ms').mean()
            .index)

    def _time_function(*args, **kwargs):
        return time

    position_info = (
        get_interpolated_position_dataframe(
            epoch_key, ANIMALS, _time_function)
        .dropna(subset=['linear_distance', 'linear_speed']))

    time = position_info.index

    tetrode_info = make_tetrode_dataframe(ANIMALS, epoch_key=epoch_key)
    is_brain_areas = (
        tetrode_info.area.astype(str).str.upper().isin(brain_areas))
    tetrode_keys = tetrode_info.loc[is_brain_areas].index
    lfps = get_LFPs(tetrode_keys, ANIMALS)
    lfps = lfps.resample('2ms').mean().fillna(method='pad').reindex(time)

    try:
        neuron_info = make_neuron_dataframe(ANIMALS).xs(
            epoch_key, drop_level=False)
        neuron_info = neuron_info.loc[
            (neuron_info.numspikes > 100) &
            neuron_info.area.isin(brain_areas) &
            (neuron_info.type == 'principal')]
        spikes = get_all_spike_indicators(
            neuron_info.index, ANIMALS, _time_function).reindex(time)
    except KeyError:
        spikes = None

    tetrode_info = tetrode_info.loc[is_brain_areas]
    multiunit = (get_all_multiunit_indicators(
        tetrode_info.index, ANIMALS, _time_function)
        .sel(features=_MARKS)
        .reindex({'time': time}))
    multiunit_spikes = (np.any(~np.isnan(multiunit.values), axis=1)
                        ).astype(np.float)
    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_population_firing_rate(
            multiunit_spikes, SAMPLING_FREQUENCY, smoothing_sigma=0.020),
        index=time, columns=['firing_rate'])

    logger.info('Finding ripple times...')
    ripple_times = get_ripple_times(epoch_key)

    ripple_band_lfps = pd.DataFrame(
        np.stack([filter_ripple_band(lfps.values[:, ind])
                  for ind in np.arange(lfps.shape[1])], axis=1),
        index=lfps.index)

    is_ripple = pd.DataFrame(np.zeros_like(time, dtype=np.int), index=time,
                             columns=['ripple_number'])
    for replay_number, start_time, end_time in ripple_times.itertuples():
        is_ripple.loc[start_time:end_time] = replay_number

    is_theta = get_theta_times(epoch_key)

    return {
        'position_info': position_info,
        'ripple_times': ripple_times,
        'spikes': spikes,
        'multiunit': multiunit,
        'lfps': lfps,
        'tetrode_info': tetrode_info,
        'ripple_band_lfps': ripple_band_lfps,
        'multiunit_firing_rate': multiunit_firing_rate,
        'sampling_frequency': SAMPLING_FREQUENCY,
        'is_ripple': is_ripple,
        'is_theta': is_theta,
    }
