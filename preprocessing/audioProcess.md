get the audio feature

get the 1/4th beatlen fron csv by 60000/(bpm*4) = 15000/bpm
get first beat timestamp t from .osu 1st timing point
make chunks of audio data starting from t of length +-t/2 from point t and next chunk midpoint will be 2 * t and so on for n * t