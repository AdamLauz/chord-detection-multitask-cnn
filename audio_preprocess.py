import json
import os
import math
import librosa
import itertools
import numpy as np
from create_chords_dataset import ALLOWED_CHORD_TYPES, NOTES, ACCIDENTALS
import music21 as m21
import scipy as sp
import random


DATASET_PATH = "dataset"
JSON_PATH = "chords_data.json"
FRAME_SIZE = 2048
HOP_SIZE = 512 * 1
EDGE = FRAME_SIZE * 1
SAMPLE_RATE = 22050
TRACK_DURATION = 1  # measured in seconds
SAMPLES_PER_TRACK = (SAMPLE_RATE - 2 * EDGE) * TRACK_DURATION
AUDIO_FILE_FORMAT = "wav"
NOTE_MAPPING = [note + accidental for note, accidental in list(itertools.product(*[NOTES, ACCIDENTALS]))]
MAX_FREQ = m21.pitch.Pitch("C7").frequency
TOP_N_MAGNITUDE_FREQUENCIES = 30
SAMPLE_PERCENTAGE = 0.3


def save_cqt(dataset_path: str, json_path: str, hop_length: int = HOP_SIZE, num_segments: int = 5, verbose: bool = False):
    """
    This function goes over the audio data set and generates CQTs and set of labels for each audio file and finally
    saves all the data into single json file under JSON_PATH

    :param dataset_path: Path to audio dataset
    :param json_path:  Path to json file used to save CQTs
    :param hop_length: Sliding window for FFT. Measured in # of samples
    :param num_segments: Number of segments we want to divide sample tracks into
    :param verbose:
    :return:
    """

    # dictionary to store mapping, labels, and CQTs
    data = {
        "chord_root_mappings": [],
        "chord_type_mappings": [],
        "chord_inversion_mappings": [],
        "chord_root_labels": [],
        "chord_type_labels": [],
        "chord_inversion_labels": [],
        "cqt": [],
        #"ft": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_cqt_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # process all audio files in genre sub-dir
        for f in filenames:
            if random.random() > SAMPLE_PERCENTAGE:
                continue

            f_name_parts = f.split(f".{AUDIO_FILE_FORMAT}")[0].split("_")
            if len(f_name_parts) < 5:
                if verbose:
                    print(f"skipping wrong audio file:: {f}")
                continue


            instrument = f_name_parts[0]
            chord_root = f_name_parts[1]


            if "-" in chord_root: # want to take only # to avoid ambiguity
                continue
            if "#" in chord_root and "B" in chord_root: # B# is enharmonic to C
                continue
            if "#" in chord_root and "E" in chord_root: # E# is enharmonic to F
                continue

            #octave = f_name_parts[2]
            chord_type = f_name_parts[3]
            chord_inversion = f_name_parts[4]

            if verbose:
                print("\nProcessing: {}".format(f"{instrument} {chord_root} {chord_type} {chord_inversion}"))

            # load audio file
            file_path = os.path.join(dirpath, f)
            signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

            # avoid edges as they may contain noise
            signal = signal[EDGE:][:-EDGE]

            # process all segments of audio file
            for d in range(num_segments):

                # calculate start and finish sample for current segment
                start = samples_per_segment * d
                finish = start + samples_per_segment

                # derive spectrum using FT
                # ft = sp.fft.fft(signal[start:finish])
                # ft_magnitude = np.absolute(ft)
                # frequency = np.linspace(0, sample_rate, len(ft_magnitude))
                # ft_magnitude = ft_magnitude[frequency <= MAX_FREQ]
                #ind = np.argpartition(ft_magnitude, -TOP_N_MAGNITUDE_FREQUENCIES)[-TOP_N_MAGNITUDE_FREQUENCIES:]
                #top_frequencies = frequency[ind]

                # extract cqt
                cqt = np.abs(librosa.cqt(signal[start:finish], sr=(sample_rate), hop_length=hop_length, bins_per_octave=24, norm=np.inf, fmin=m21.pitch.Pitch("C0").frequency))
                cqt = librosa.amplitude_to_db(cqt, ref=np.max)
                cqt = cqt.T

                # store only cqt feature with expected number of vectors
                if len(cqt) == num_cqt_vectors_per_segment:
                    data["cqt"].append(cqt.tolist())
                    # data["ft"].append(ft_magnitude.tolist())

                    # save mappings & labels
                    data["chord_root_mappings"].append(chord_root)
                    data["chord_root_labels"].append(NOTE_MAPPING.index(chord_root))

                    data["chord_type_mappings"].append(chord_type)
                    data["chord_type_labels"].append(ALLOWED_CHORD_TYPES.index(chord_type))

                    data["chord_inversion_mappings"].append(chord_inversion)
                    data["chord_inversion_labels"].append(int(chord_inversion))
                    if verbose:
                        print("{}, segment:{}".format(file_path, d + 1))

    # save CQTs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_cqt(DATASET_PATH, JSON_PATH, num_segments=2)
