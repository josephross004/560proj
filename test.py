import utils

#test 1: see if you can read in a wav file.
wv = utils.SoundData("./Animal-Sound-Dataset/Aslan/aslan_1.wav")

#test 2: see if you can save a vector as a npy file.
wv.saveAsVector("./testout/wav")

#test 3: see if you can save a vector as a spectrogram.
wv.saveAsSpectrogram("./testout/img")