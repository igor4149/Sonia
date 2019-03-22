import os, random
import pretty_midi
import numpy as np
import keras
import tensorflow as tf
import numpy as np


def parse_midi(path):
    midi = None
    midi = pretty_midi.PrettyMIDI(path)
    midi.remove_invalid_notes()
    return midi

def get_percent_mono(pm_instrument_roll):
    mask = pm_instrument_roll.T > 0
    notes = np.sum(mask, axis=1)
    n = np.count_nonzero(notes)
    single = np.count_nonzero(notes == 1)
    if single > 0:
        return float(single) / n
    elif single == 0 and n > 0:
        return 0.0
    else:
        return 0.0

def filter_mono(pm_instruments, percent_mono=0.99):
    return [i for i in pm_instruments if get_percent_mono(i.get_piano_roll()) >= percent_mono]

def mono_frames_split(midi, frame_size):
    X, y = [], []
    for m in midi:
        if m is not None:
            instruments = filter_mono(m.instruments, 1.0)
            for i in instruments:
                if len(i.notes) > frame_size:
                    frames = encode_sliding_frames(i, frame_size)
                    for fr in frames:
                        X.append(fr[0])
                        y.append(fr[1])
    return (np.asarray(X), np.asarray(y))

def encode_sliding_frames(pm_instrument, frame_size):
    roll = np.copy(pm_instrument.get_piano_roll(fs=5).T)
    summed = np.sum(roll, axis=1)
    mask = (summed > 0).astype(float)
    roll = roll[np.argmax(mask):]
    roll = (roll > 0).astype(float)
    rests = np.sum(roll, axis=1)
    rests = (rests != 1).astype(float)
    roll = np.insert(roll, 0, rests, axis=1)
    frames = []
    for i in range(0, roll.shape[0] - frame_size - 1):
        frames.append((roll[i:i + frame_size], roll[i + frame_size + 1]))
    return frames

def get_data_generator(midi_paths, 
                       frame_size=30, 
                       batch_size=64,
                       max_files_in_ram=200):
    load_index = 0
    while True:
        load_files = midi_paths[load_index:load_index + max_files_in_ram]
        load_index = (load_index + max_files_in_ram) % len(midi_paths)
        parsed = map(parse_midi, load_files)
        data = mono_frames_split(parsed, frame_size)
        batch_index = 0
        while batch_index + batch_size < len(data[0]):
            res = (data[0][batch_index: batch_index + batch_size], 
                   data[1][batch_index: batch_index + batch_size])
            yield res
            batch_index = batch_index + batch_size
