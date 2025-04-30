# Are You Seeing What I’m Hearing? 
## Convolutional Neural Network Model’s Classification of Spectrogram Images
### Sammy Gagou, Joseph Ross, Nadine Hughes, Julia Guzzo, Aneesh Sallarm
## Abstract
This paper explores the results of the study to see how effectively a convolutional neural network (CNN) model can be trained to analyze a spectrogram image of an audio recording and classify the species of animal that made the sound. The dataset was collected from a public GitHub repository [6] and consists of 875 animal sounds and is split into 10 different types (cat, dog, bird, cow, lion, sheep, frog, chicken, donkey, and monkey). Results showed significantly improved processing efficiency and classification accuracy for spectrogram images (75.00\%) versus waveforms (39.77\%), and a percentile-based compression of spectrograms showed another significant decrease in training time for a comparably small decrease in classification accuracy (60.23\%).  

--------------------------

### Usage

#### Data Processing / Creating the Dataset
1. Install all of the necessary packages into your environment:
```
/560proj/$  poetry install
```
2. Run the `prepareData.py` file: to make sure it works correctly, run it once in TEST mode
```
python prepareData.py
```
   then in REAL mode using the `-t` argument
```
python prepareData.py -t
```
common problems:
  - No FFMpeg: you have to have FFMpeg installed and available for use, for converting mp3 data to wav.

#### Using the Model

WIP.
