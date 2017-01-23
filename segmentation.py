''' Speech detection using entropy-based algorithm described in:
'A robust algorithm for detecting speech segments using an entropic contrast',
Waheed et al.

Steps:
1. Pre-processing stage (add a pre-emph filter + other filter)
2. Choose speech window
3. Determine short time prob. distr. using histogram
4. Compute weighted entropy of speech frame
5. Determine the adaptive threshold for entropy profile
6. Threshold the entropy profile
7. Apply speech segment size constraint
8. Appy intra-segment distance constraint
9. Yay!
'''

#!/usr/bin/python

import numpy as np
import scipy
from scipy.fftpack import rfft, irfft, fftfreq
from scipy.io import wavfile
import matplotlib.pyplot as plt
import python_speech_features as speech

sample = 'samples/sf1_fi1' # To be added feature: make this an argument passed in by the user

input_filename = sample + '.wav'

rate, data = wavfile.read(input_filename)

#################################################
# 			Step 1: Filters
#################################################

# Pre-emphasis:

preemp_data = speech.sigproc.preemphasis(data, 0.97)
signal = preemp_data

################# XM STUFF GOES HERE #######
# Select frequency components 250 Hz < f < 6000 Hz

#################################################
# 			Step 2: Divide into windows
#################################################

# Divide the speech into overlapping frames with 25-50% overlap

# Find duration of sample so we can divide it into 20 ms segments (because inertia of glottis?)

num_samples = len(signal)

duration = num_samples / float(rate)

# Need to define the length of each frame (in samples) and the number of samples after the start of the prev. 
# frame that the next frame should begin

frame_duration_in_seconds = 0.02

num_windows = int(duration / frame_duration_in_seconds)

frame_length = num_samples / float(num_windows)

amount_overlap = 0.5 # 25-50%

frame_step = frame_length * amount_overlap

# plug into python speech features function for dividing into overlapping frames

frames = speech.sigproc.framesig(signal, frame_length, frame_step)

#################################################
# Step 3: Determine short time prob. distr.
#################################################

# Take the FFT of each frame, then use histogram to find prob. distribution

# Construct a histogram with N bins for each frame
# N will be 50-100, choice depending on sensitivity and computational load

# iterate through the frames and create a histogram for each

entropy_profile = np.empty([len(frames)])

for i in range(len(frames)):
	#print i

	# Take FFT
	frame_freq = scipy.fftpack.fft(frames[i])
	#print frame_freq

	# number of bins
	N = 50

	# take histogram

	frame_hist, frame_bin_edges = np.histogram(frame_freq, N, None, True, None, None )


#################################################
# Step 4: Compute the weighted entropy per frame
#################################################

	# Calculate the entropy using 

	# H(X) = -Sum_(k=1)^N p_k log(p_k)

	H = scipy.stats.entropy(frame_hist)
	#print H

	entropy_profile[i] = H


'''
# Use fft to recover frequency from our filtered amplitude

	#k = arange(len(filtered_data))
	#T = len(filtered_data)/rate
	#frq_label = k /T


	complex = scipy.fft(preemp_data)
	d = len(complex)

	shifted = np.fft.fftshift(complex)
	
	fig = plt.plot(abs(shifted[:(d-1)]), 'm')

	# plotting for sanity check
	plt.setp(fig, linewidth=0.25)
	plt.savefig('fftfoo_shift.png')
	plt.clf()
	
	# Normal voice range is 500 Hz to 2 kHz, so apply a hackish "band pass filter" to include only those


	# after filtering, recover our signal

	recovered_signal = scipy.ifft(shifted)'''