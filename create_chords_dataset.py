import os

import soundcard as sc
import soundfile as sf
from pathlib import Path
import time
import rtmidi
import music21 as m21
from random import random
from typing import Dict, List
from pedalboard import load_plugin
from pedalboard.io import AudioFile
from mido import Message

"""
This file is used to generate a chords dataset in WAV format.
It's done by generating various chords that can be played by predefined musical instruments (aka "Channels").
Then the chords are sent in a midi channel to the virtual instrument.
For example a VSTI piano is listening to MIDI channel 1.
The sound produced by the VSTI is captured and saved as WAV file.
"""

# list of channels to send midi message to in order to capture the sound
CHANNELS = [
    {"name": "piano", "lowest_note": m21.pitch.Pitch("A0"), "highest_note": m21.pitch.Pitch("C7"), "chords": []},
    {"name": "jazzguitar", "lowest_note": m21.pitch.Pitch("E2"), "highest_note": m21.pitch.Pitch("G5"), "chords": []},
    {"name": "banjo", "lowest_note": m21.pitch.Pitch("G3"), "highest_note": m21.pitch.Pitch("C6"), "chords": []},
    {"name": "mandolin", "lowest_note": m21.pitch.Pitch("G3"), "highest_note": m21.pitch.Pitch("F6"), "chords": []},
    {"name": "violins", "lowest_note": m21.pitch.Pitch("G3"), "highest_note": m21.pitch.Pitch("G6"), "chords": []},
    {"name": "bigband", "lowest_note": m21.pitch.Pitch("D3"), "highest_note": m21.pitch.Pitch("A5"), "chords": []},
    {"name": "nylonguitar", "lowest_note": m21.pitch.Pitch("E2"), "highest_note": m21.pitch.Pitch("C6"), "chords": []},
    {"name": "accordionbasson", "lowest_note": m21.pitch.Pitch("F3"), "highest_note": m21.pitch.Pitch("A6"),
     "chords": []},
    {"name": "accordionviolin", "lowest_note": m21.pitch.Pitch("F3"), "highest_note": m21.pitch.Pitch("A6"),
     "chords": []},
    {"name": "trumpet", "lowest_note": m21.pitch.Pitch("A3"), "highest_note": m21.pitch.Pitch("G5"), "chords": []},
    {"name": "bassguitar", "lowest_note": m21.pitch.Pitch("B0"), "highest_note": m21.pitch.Pitch("A3"), "chords": []},
    {"name": "clarinet", "lowest_note": m21.pitch.Pitch("D3"), "highest_note": m21.pitch.Pitch("D6"), "chords": []},
]

# allowed chord types
ALLOWED_CHORD_TYPES = [
    'minor triad',
    'major triad',
    'diminished triad',
    'augmented triad',
    'diminished seventh chord',
    'half-diminished seventh chord',
    'dominant seventh',
    'minor seventh',
    'major seventh chord',
    'minor-augmented tetrachord'
]

# Globals
OCTAVES = [0, 1, 2, 3, 4, 5, 6]  # range of octaves to be considered when generating the chords
CHORD_INTERVALS = ['m3', 'M3']  # chord interval building blocks
ACCIDENTALS = ['', '#']  # , '-']  # possible accidentals to be considered
NOTES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
SAMPLE_PERCENTAGE = 0.01  # this is the chord sampling rate that is used to decide whether to include a chord or not
SAMPLE_RATE = 22050  # [Hz]. sampling rate.
RECORD_SEC = 5  # [sec]. duration recording audio.
DATASET_PATH = 'dataset'  # the dataset destination folder.
MIDI_OUTPORT = 5  # check which is the output you want to send midi to
MIDI_ON = 16 * 9
MIDI_OFF = 16 * 8
MIDI_ON_VELOCITY = 100
MIDI_OFF_VELOCITY = 0
VST3_PLUGIN_PATH = "C:\Program Files\Common Files\VST3\Kontakt.vst3"
NUM_CHANNELS = 2


def get_chord_inversion(chord: m21.chord.Chord, inversion: int) -> m21.chord.Chord:
    """
    This funtion gets basic form chord and returns its inversion
    :param chord: music 21 chord object
    :param inversion: integer specifying the inversion
    :return: inverted m21 chord object
    """
    chord_inv = m21.chord.Chord([note.nameWithOctave for note in chord.notes])
    chord_inv.inversion(inversion)
    return chord_inv


def can_instrument_play_chord(chord: m21.chord.Chord, instrument: Dict) -> bool:
    """
    This function checks whether a chord can be played with a given instrument
    :param chord: m21 chord object
    :param instrument: Dict with "lower_note" and "higher_note" pitches (m21.pitch.Pitch objects)
    :return: boolean indicating if a certain chord can be played with the given instrument
    """

    def is_chord_in_interval(chord: m21.chord.Chord, lowest: m21.pitch.Pitch, highest: m21.pitch.Pitch) -> bool:
        return lowest <= chord.pitches[0] and highest >= chord.pitches[-1]

    return is_chord_in_interval(chord, instrument["lowest_note"], instrument["highest_note"])


def prepare_chords(channels: List[Dict], octaves: List[int], notes: List[str], accidentals: List[str],
                   chord_intervals: List[str], allowed_chord_types: List[str], verbose=False):
    for channel in channels:
        for octave in octaves:
            for pc in notes:
                for accidental in accidentals:
                    root = m21.note.Note(f'{pc}{accidental}{octave}')
                    if verbose:
                        print(f'{pc}{accidental}{octave}')
                    chord_list = [root]
                    for firt_inverval in chord_intervals:
                        third = root.transpose(firt_inverval)
                        chord_list += [third]
                        for second_interval in chord_intervals:
                            fifth = third.transpose(second_interval)
                            chord_list += [fifth]
                            for seventh_interval in ['skip'] + chord_intervals:
                                if seventh_interval != 'skip':
                                    seventh = fifth.transpose(seventh_interval)
                                    chord_list += [seventh]
                                    chord = m21.chord.Chord(chord_list)
                                    chord_inv_1 = get_chord_inversion(chord, 1)
                                    chord_inv_2 = get_chord_inversion(chord, 2)
                                    chord_inv_3 = get_chord_inversion(chord, 3)

                                    if can_instrument_play_chord(chord,
                                                                 channel) and chord.commonName in allowed_chord_types and random() <= SAMPLE_PERCENTAGE:
                                        channel["chords"] += [chord]
                                    if can_instrument_play_chord(chord_inv_1,
                                                                 channel) and chord_inv_1.commonName in allowed_chord_types and random() <= SAMPLE_PERCENTAGE:
                                        channel["chords"] += [chord_inv_1]
                                    if can_instrument_play_chord(chord_inv_2,
                                                                 channel) and chord_inv_2.commonName in allowed_chord_types and random() <= SAMPLE_PERCENTAGE:
                                        channel["chords"] += [chord_inv_2]
                                    if can_instrument_play_chord(chord_inv_3,
                                                                 channel) and chord_inv_3.commonName in allowed_chord_types and random() <= SAMPLE_PERCENTAGE:
                                        channel["chords"] += [chord_inv_3]

                                    chord_list = chord_list[:-1]
                                else:
                                    chord = m21.chord.Chord(chord_list)

                                    chord_inv_1 = get_chord_inversion(chord, 1)
                                    chord_inv_2 = get_chord_inversion(chord, 2)

                                    if can_instrument_play_chord(chord,
                                                                 channel) and chord.commonName in allowed_chord_types and random() <= SAMPLE_PERCENTAGE:
                                        channel["chords"] += [chord]
                                    if can_instrument_play_chord(chord_inv_1,
                                                                 channel) and chord_inv_1.commonName in allowed_chord_types and random() <= SAMPLE_PERCENTAGE:
                                        channel["chords"] += [chord_inv_1]
                                    if can_instrument_play_chord(chord_inv_2,
                                                                 channel) and chord_inv_2.commonName in allowed_chord_types and random() <= SAMPLE_PERCENTAGE:
                                        channel["chords"] += [chord_inv_2]

                            chord_list = chord_list[:-1]
                        chord_list = chord_list[:-1]
    return channels


def play_chord_and_record(channels: List[Dict], save_dir: str, verbose: bool = False):
    midiout = rtmidi.MidiOut()
    with sc.all_microphones(include_loopback=True)[0].recorder(samplerate=SAMPLE_RATE) as mic:
        with midiout:
            midiout.open_port(MIDI_OUTPORT)

            for channel, instrument in enumerate(channels):
                for chord in instrument["chords"]:
                    if verbose:
                        print(chord.pitchedCommonName)

                    # send midi on
                    midiout.send_message([MIDI_ON + channel, chord.pitches[0].midi, MIDI_ON_VELOCITY])
                    midiout.send_message([MIDI_ON + channel, chord.pitches[1].midi, MIDI_ON_VELOCITY])
                    midiout.send_message([MIDI_ON + channel, chord.pitches[2].midi, MIDI_ON_VELOCITY])
                    if chord.seventh is not None:
                        midiout.send_message([MIDI_ON + channel, chord.pitches[3].midi, MIDI_ON_VELOCITY])

                    # record audio with loopback from speaker.
                    data = mic.record(numframes=SAMPLE_RATE * RECORD_SEC)

                    # send midi off
                    midiout.send_message([MIDI_OFF + channel, chord.pitches[0].midi, MIDI_OFF_VELOCITY])
                    midiout.send_message([MIDI_OFF + channel, chord.pitches[1].midi, MIDI_OFF_VELOCITY])
                    midiout.send_message([MIDI_OFF + channel, chord.pitches[2].midi, MIDI_OFF_VELOCITY])
                    if chord.seventh is not None:
                        midiout.send_message([MIDI_OFF + channel, chord.pitches[3].midi, MIDI_OFF_VELOCITY])
                    time.sleep(1.5)

                    instrument_name = instrument["name"]
                    filename = f"{instrument_name}_{chord.root().name}_{chord.root().octave}_{chord.commonName}_{chord.inversion()}.wav"
                    save_file_path = Path(save_dir, filename)
                    if verbose:
                        print(f"writing {save_file_path} ...")

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    sf.write(file=save_file_path, data=data[:, 0], samplerate=SAMPLE_RATE)


def play_chord_and_record_pedalboard(channels: List[Dict], save_dir: str, verbose: bool = False):
    """
    Use https://github.com/spotify/pedalboard to render midi to a sound from a vst (kontakt)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plugin = load_plugin(VST3_PLUGIN_PATH)

    for channel, instrument in enumerate(channels):
        for chord in instrument["chords"]:
            if verbose:
                print(chord.pitchedCommonName)

            midi_message = [Message("note_on", channel=channel, note=chord.pitches[0].midi, velocity=MIDI_ON_VELOCITY),
                            Message("note_on", channel=channel, note=chord.pitches[1].midi, velocity=MIDI_ON_VELOCITY),
                            Message("note_on", channel=channel, note=chord.pitches[2].midi, velocity=MIDI_ON_VELOCITY),
                            Message("note_off", channel=channel, note=chord.pitches[0].midi, time=RECORD_SEC),
                            Message("note_off", channel=channel, note=chord.pitches[1].midi, time=RECORD_SEC),
                            Message("note_off", channel=channel, note=chord.pitches[2].midi, time=RECORD_SEC)]

            if chord.seventh is not None:
                midi_message += [
                    Message("note_on", channel=channel, note=chord.pitches[3].midi, velocity=MIDI_ON_VELOCITY),
                    Message("note_off", channel=channel, note=chord.pitches[3].midi, time=RECORD_SEC)
                ]

            instrument_name = instrument["name"]
            filename = f"{instrument_name}_{chord.root().name}_{chord.root().octave}_{chord.commonName}_{chord.inversion()}.wav"
            save_file_path = Path(save_dir, filename)

            if verbose:
                print(f"writing {save_file_path} ...")

            with AudioFile(str(save_file_path), "w", SAMPLE_RATE, NUM_CHANNELS) as f:
                f.write(plugin(midi_message,
                               duration=RECORD_SEC,  # seconds
                               sample_rate=SAMPLE_RATE,
                               num_channels=NUM_CHANNELS))


if __name__ == "__main__":
    channels = prepare_chords(CHANNELS, OCTAVES, NOTES, ACCIDENTALS, CHORD_INTERVALS, ALLOWED_CHORD_TYPES)
    # play_chord_and_record(channels, DATASET_PATH)
    play_chord_and_record_pedalboard(channels, DATASET_PATH)
