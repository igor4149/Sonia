import os, random, glob
import pretty_midi
import misc
import train
from keras.models import model_from_json
import numpy as np
from midi2audio import FluidSynth

def output_to_midi(frames, instrument_name):
    midi = pretty_midi.PrettyMIDI()
    program = pretty_midi.instrument_name_to_program(instrument_name)
    instrument = pretty_midi.Instrument(program=program)
    cur_note = None
    cur_note_start = None
    clock = 0
    cur_count = 0
    for frame in frames:
        note_num = np.argmax(frame) - 1
        if note_num != cur_note:
            if cur_note == note_num:
                cur_count += 1
            else:
                cur_count = 0
            if cur_note is not None and cur_note >= 0:            
                note = pretty_midi.Note(velocity=127, pitch=int(cur_note), start=cur_note_start, end=clock)
                instrument.notes.append(note)
            cur_note = note_num
            cur_note_start = clock
            clock += 0.2
        else:
            clock += 0.01
    midi.instruments.append(instrument)
    return midi

def generate(model, seeds, frame_size, length, samples_num, instrument_name):
    result = []
    for i in range(samples_num):
        seed = seeds[random.randint(0, len(seeds) - 1)]
        generated = []
        buf = np.copy(seed).tolist()
        while len(generated) < length:
            arr = np.expand_dims(np.asarray(buf), 0)
            pred = model.predict(arr)
            index = np.random.choice(range(0, seed.shape[1]), p=pred[0])
            pred = np.zeros(seed.shape[1])
            pred[index] = 1
            generated.append(pred)
            buf.pop(0)
            buf.append(pred)
        result.append(output_to_midi(generated, instrument_name))
    return result

def load_model_from_checkpoint(model_dir='./model'):
    with open(os.path.join(model_dir, 'model.json'), 'r') as f:
        model = model_from_json(f.read())
    epoch = 0
    newest_checkpoint = max(glob.iglob(model_dir + '/*.hdf5'), key=os.path.getctime)
    if newest_checkpoint: 
       epoch = int(newest_checkpoint[-22:-19])
       model.load_weights(newest_checkpoint)
    return model, epoch

def main():
    randomseed = int(input("Please enter the random seed. If 0 is entered, no seed will be used\n"))
    if randomseed != 0:
        np.random.seed(randomseed)
        random.seed(randomseed)
    prime_files = [os.path.join('./datasets/music', m) for m in os.listdir('./datasets/music') if '.mid' in m]
    random.shuffle(prime_files)
    if not (os.path.isdir('./result')):
        os.makedirs('./result')
    model, epoch = load_model_from_checkpoint()
    frame_size = model.layers[0].get_input_shape_at(0)[1]
    seed_generator = misc.get_data_generator(prime_files)
    X, y = next(seed_generator)
    generated = generate(model, X, frame_size, length=200, samples_num=10, instrument_name='Electric Piano 1')
    for i, midi in enumerate(generated):
        file = os.path.join('./result', '{}.mid'.format(i + 1))
        midi.write(file.format(i + 1))
        fs = FluidSynth('./result/soundfont/FluidR3_GM.sf2')
        fs.midi_to_audio(file.format(i + 1), os.path.join('./result', '{}.wav'.format(i + 1)))
if __name__ == '__main__':
    main()


