import misc
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import time, os, random
import tensorflow as tf
OUTPUT_SIZE = 129

def get_model(num_layers):
    epoch = 0
    model = Sequential()
    for layer_index in range(num_layers):
        kwargs = dict() 
        kwargs['units'] = 100
        if layer_index == 0:
            kwargs['input_shape'] = (30, OUTPUT_SIZE)
            if num_layers == 1:
                kwargs['return_sequences'] = False
            else:
                kwargs['return_sequences'] = True
            model.add(LSTM(**kwargs))
        else:
            if not layer_index == num_layers - 1:
                kwargs['return_sequences'] = True
                model.add(LSTM(**kwargs))
            else:
                kwargs['return_sequences'] = False
                model.add(LSTM(**kwargs))
        model.add(Dropout(0.3))
    model.add(Dense(OUTPUT_SIZE))
    model.add(Activation('softmax'))
    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model, epoch
  
def get_callbacks(experiment_dir='./model', checkpoint_monitor='val_acc'):
    callbacks = []
    filepath = os.path.join(experiment_dir, 
                            'checkpoints', 
                            'checkpoint-epoch_{epoch:03d}-val_acc_{val_acc:.3f}.hdf5')

    callbacks.append(ModelCheckpoint(filepath, 
                                     monitor=checkpoint_monitor, 
                                     verbose=1, 
                                     save_best_only=False, 
                                     mode='max'))

    callbacks.append(ReduceLROnPlateau(monitor='val_loss', 
                                       factor=0.5, 
                                       patience=3, 
                                       verbose=1, 
                                       mode='auto', 
                                       epsilon=0.0001, 
                                       cooldown=0, 
                                       min_lr=0))
    return callbacks

def main():
    config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
    sess = tf.Session(config=config) 
    keras.backend.set_session(sess)
    midi_files = [os.path.join('./datasets/music', path) \
                      for path in os.listdir("./datasets/music") \
                      if '.mid' in path]

    random.shuffle(midi_files)
    val_split = 0.2
    val_split_index = int(float(len(midi_files)) * val_split)
    train_generator = misc.get_data_generator(midi_files[0:val_split_index])
    val_generator = misc.get_data_generator(midi_files[val_split_index:])
    model, epoch = get_model(3)
    if not (os.path.isdir('./model')):
        os.makedirs('./model')
    with open(os.path.join('./model', 'model.json'), 'w') as f:
        f.write(model.to_json())
    frames_in_midi = 500
    start_time = time.time()
    callbacks = get_callbacks()
    model.fit_generator(train_generator,
                        steps_per_epoch=len(midi_files) * frames_in_midi / 64, 
                        epochs=3,
                        validation_data=val_generator, 
                        validation_steps=len(midi_files) * 0.2 * frames_in_midi / 64,
                        verbose=1,
                        callbacks=callbacks,
                        initial_epoch=epoch)
if __name__ == '__main__':
    main()