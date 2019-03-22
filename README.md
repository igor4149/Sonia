# Sonia
neural sound synthesis

# As for now
results/demo - here you can find several examples of generated music-alike things
# How-to
0.  **IMPORTANT:** pip install pretty-midi, pip install midi2audio, sudo apt install fluidsynth. Also make sure you have tf and keras available.
1.  Clone repo then `cd Sonia`
2.  `python3 generation.py` - please be patient, this may take several minutes - that's ok.
3.  Results are now stored in results - 10 .mid files and 10 .wav files for them.
4.  Conversion from midi to wav uses soundfont, stored in results/soundfont - so your .mid and generated wav may sound differently (a bit tho).
5.  If you wanna test training - `python3 train.py` - should work if you have tf and keras.
## TODO
Random seed & argument parsing. requirements.txt (???)
