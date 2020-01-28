from os.path import abspath, dirname, join, pardir

import numpy as np
from loren_frank_data_processing import Animal

# LFP sampling frequency
SAMPLING_FREQUENCY = 500

# Data directories and definitions
ROOT_DIR = join(abspath(dirname(__file__)), pardir)
RAW_DATA_DIR = join(ROOT_DIR, 'Raw-Data')
PROCESSED_DATA_DIR = join(ROOT_DIR, 'Processed-Data')
FIGURE_DIR = join(ROOT_DIR, 'figures')

ANIMALS = {
    'bon': Animal(directory=join(RAW_DATA_DIR, 'Bond'), short_name='bon'),
    'fra': Animal(directory=join(RAW_DATA_DIR, 'Frank'), short_name='fra'),
}

_MARKS = ['channel_1_max', 'channel_2_max', 'channel_3_max', 'channel_4_max']
BRAIN_AREAS = ['CA1', 'CA2', 'CA3']

detector_parameters = {
    'movement_var': 25.0,
    'replay_speed': 1,
    'place_bin_size': 1.0,
    'spike_model_knot_spacing': 8.0,
    'spike_model_penalty': 0.5,
    'movement_state_transition_type': 'w_track_1D_random_walk',
    'multiunit_model_kwargs': {
        'bandwidth': np.array([20.0, 20.0, 20.0, 20.0, 8.0])},
    'multiunit_occupancy_kwargs': {'bandwidth': np.array([8.0])},
    'discrete_state_transition_type': 'constant',
}

classifier_parameters = {
    'movement_var': 25.0,
    'replay_speed': 1,
    'place_bin_size': 1.0,
    'continuous_transition_types': [['w_track_1D_random_walk', 'uniform'],
                                    ['uniform',                'uniform']],
    'model_kwargs': {
        'bandwidth': np.array([20.0, 20.0, 20.0, 20.0, 8.0])}

}

discrete_state_transition = np.array([[0.99, 0.01],
                                      [0.99, 0.01]])
