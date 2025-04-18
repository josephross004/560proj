# Are You Seeing What I’m Hearing? :
## Convolutional Neural Network Model’s Classification of Spectrogram Images
### Sammy Gagou, Joseph Ross, Nadine Hughes, Julia Guzzo, Aneesh Sallarm
## Abstract
Spectrogram classification is useful for a variety of reasons, such as, but not limited to, biodiversity monitoring, animal behavior studies, taxonomic research, and conservation efforts. For example, with more and more species being placed on protected and endangered lists scientists need a way to track the location and population levels of these species accurately. While we can record audio in an attempt to perform this research it is not possible to invest the manpower to analyze enough audio to produce statistically meaningful results. This is where the possibility of AI and model training comes in. We decided to research a proof of concept that a model could be effectively trained to classify Spectrogram images and classify what animal produced it. Different models and techniques will have different levels, so being able to tune/use models for different use cases improves efficiency in pursuing/supporting these efforts. This experiment explores the results of the study to see how effectively a convolutional neural network (CNN) model can be trained to analyze a spectrogram image of an audio recording and classify the species of animal that made the sound. The dataset was collected from a public GitHub repository and consists of 875 animal sounds and is split into 10 different types (cat, dog, bird, cow, lion, sheep, frog, chicken, donkey, and monkey). 

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
