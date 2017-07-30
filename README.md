# Classify Glitches 

### - Pradyoth Hegde, EECS: 495: Machine Learning: Foundations, Algorithms and Applications

You are hired by the National Science Foundation to work on the problem of classifying glitches. Glitches are non-Gaussian disturbances in the gravitational-wave data of the Advanced Laser Interferometer Gravitational-wave Observatory (aLIGO).
Glitches are typically represented as spectrogram, a time-frequency representation where the x-axis represents the duration of the glitch and the y-axis shows the frequency content of the glitch. The colors indicate the “loudness” of the glitch in the aLIGO detector. For this assignment, however, glitches are represented as vectors and they cannot be easily visualized.
Different environmental and instrumental mechanisms will produce glitches of different shape and morphology. Here, we consider four major classes Blip, Whistle, Koi fish and Wandering line. Your task is to build a classification model which could help scientists understand the underlying mechanisms for creating such glitches so that they are removed from the detector.
You are given the file train_data_label.csv. Each row of this matrix represents one data sample, where the last column is an integer representing the corresponding label, i.e.: 1 is Blip, 2 is Whistle, 3 is Koi fish and 4 is Wandering line.
You are also given the file test_data.csv. Each row of this matrix is one data sample, and the corresponding label is missing.


## Using Softmax Gradient Descent in a One versus all method.