import os
import pickle
from itertools import product
import logging
import copy
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import networkx as nx
from tqdm import tqdm
from scipy.io import wavfile
from scipy.signal import fftconvolve
from skimage.measure import block_reduce
from torchaudio.functional import melscale_fbanks

from ss_baselines.common.utils import to_tensor
from soundspaces.mp3d_utils import CATEGORY_INDEX_MAPPING
from soundspaces.audio_utils import compute_spectrogram
from soundspaces.utils import next_greater_power_of_2


class AudioGoalDataset(Dataset):
    def __init__(self, config, scene_graphs, scenes, split, use_polar_coordinates=False, use_cache=False, filter_rule=''):
        self.use_cache = use_cache
        self.files = list()
        self.goals = list()
        self.binaural_rir_dir = 'data/binaural_rirs/mp3d'
        self.source_sound_dir = f'data/sounds/semantic_splits/{split}'
        self.source_sound_dict = dict()
        self.rir_sampling_rate = self.config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE
        sound_files = os.listdir(self.source_sound_dir)

        spec_config = self.config.TASK_CONFIG.SPECTROGRAM_SENSOR
        self.hop_length = int(self.rir_sampling_rate * (spec_config.HOP_SIZE_MS / 1000.0))
        self.win_length = int(self.rir_sampling_rate * (spec_config.WIN_SIZE_MS / 1000.0))
        self.n_mels = int(spec_config.NUM_MELS)
        self.n_fft = int(next_greater_power_of_2(self.win_length))
        self.downsample = spec_config.DOWNSAMPLE
        self.include_gcc_phat = bool(spec_config.GCC_PHAT)
        self.window = torch.hann_window(self.win_length, device="cpu")
        self.mel_scale = melscale_fbanks(
            n_freqs=(self.n_fft // 2) + 1,
            f_min=0,
            f_max=self.rir_sampling_rate / 2, # nyquist
            n_mels=self.n_mels,
            sample_rate=self.rir_sampling_rate,
            mel_scale="htk",
            norm=None,
        ).to(device="cpu", dtype=torch.float32) if self.n_mels else None

        for scene in tqdm(scenes):
            scene_graph = scene_graphs[scene]
            goals = []
            subgraphs = list(nx.connected_components(scene_graph))
            sr_pairs = list()
            for subgraph in subgraphs:
                sr_pairs += list(product(subgraph, subgraph))
            random.shuffle(sr_pairs)
            for s, r in sr_pairs[:50000]:
                sound_file = random.choice(sound_files)
                index = CATEGORY_INDEX_MAPPING[sound_file[:-4]]
                angle = random.choice([0, 90, 180, 270])
                rir_file = os.path.join(self.binaural_rir_dir, scene, str(angle), f"{r}_{s}.wav")

                self.files.append((rir_file, sound_file))
                delta_x = scene_graph.nodes[s]['point'][0] - scene_graph.nodes[r]['point'][0]
                delta_y = scene_graph.nodes[s]['point'][2] - scene_graph.nodes[r]['point'][2]
                goal_xy = self._compute_goal_xy(delta_x, delta_y, angle, use_polar_coordinates)

                goal = to_tensor(np.zeros(3))
                goal[0] = index # ground truth category label
                goal[1:] = goal_xy # ground truth ∆x, ∆y
                goals.append(goal)

            self.goals += goals

        self.data = [None] * len(self.goals)
        self.load_source_sounds()

    def audio_length(self, sound):
        return self.source_sound_dict[sound].shape[0] // self.rir_sampling_rate

    def load_source_sounds(self):
        sound_files = os.listdir(self.source_sound_dir)
        for sound_file in sound_files:
            audio_data, sr = librosa.load(
                os.path.join(self.source_sound_dir, sound_file),
                sr=self.rir_sampling_rate
            )
            self.source_sound_dict[sound_file] = audio_data

    @staticmethod
    def _compute_goal_xy(delta_x, delta_y, angle, use_polar_coordinates):
        """
        -Y is forward, X is rightward, agent faces -Y
        """
        if angle == 0:
            x = delta_x
            y = delta_y
        elif angle == 90:
            x = delta_y
            y = -delta_x
        elif angle == 180:
            x = -delta_x
            y = -delta_y
        else:
            x = -delta_y
            y = delta_x

        if use_polar_coordinates:
            theta = np.arctan2(y, x)
            distance = np.linalg.norm([y, x])
            goal_xy = to_tensor([theta, distance])
        else:
            goal_xy = to_tensor([x, y])
        return goal_xy

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        if (self.use_cache and self.data[item] is None) or not self.use_cache:
            rir_file, sound_file = self.files[item] # input source
            audiogoal = self.compute_audiogoal(rir_file, sound_file)
            spectrogram = compute_spectrogram(
                audio_data=audiogoal,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                window=self.window,
                mel_scale=self.mel_scale,
                downsample=self.downsample,
                include_gcc_phat=self.include_gcc_phat,
                backend="torch",
            )
            inputs_outputs = ([spectrogram], self.goals[item]) # (inputs, labels)

            if self.use_cache: # cache new data for future use
                self.data[item] = inputs_outputs
        else: # when data is cached !None
            inputs_outputs = self.data[item]

        return inputs_outputs

    def compute_audiogoal(self, binaural_rir_file, sound_file):
        sampling_rate = self.rir_sampling_rate
        try:
            sampling_freq, binaural_rir = wavfile.read(binaural_rir_file)  # float32
        except ValueError:
            logging.warning("{} file is not readable".format(binaural_rir_file))
            binaural_rir = np.zeros((sampling_rate, 2)).astype(np.float32)
        if len(binaural_rir) == 0:
            logging.debug("Empty RIR file at {}".format(binaural_rir_file))
            binaural_rir = np.zeros((sampling_rate, 2)).astype(np.float32)

        current_source_sound = self.source_sound_dict[sound_file]
        index = random.randint(0, self.audio_length(sound_file) - 2)
        if index * sampling_rate - binaural_rir.shape[0] < 0:
            source_sound = current_source_sound[: (index + 1) * sampling_rate]
            binaural_convolved = np.array([
                fftconvolve(
                    source_sound,
                    binaural_rir[:, channel],
                )
                for channel in range(binaural_rir.shape[-1])
            ])
            audiogoal = binaural_convolved[:, index * sampling_rate: (index + 1) * sampling_rate]
        else:
            # include reverb from previous time step
            source_sound = current_source_sound[index * sampling_rate - binaural_rir.shape[0]
                                                : (index + 1) * sampling_rate]
            binaural_convolved = np.array([
                fftconvolve(
                    source_sound,
                    binaural_rir[:, channel],
                    mode='valid',
                )
                for channel in range(binaural_rir.shape[-1])
            ])
            audiogoal = binaural_convolved[:, :-1]

        return audiogoal