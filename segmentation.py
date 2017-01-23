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
import scipy.stats
import matplotlib.pyplot as plt
import python_speech_features as speech

#sample = 'harvard_sentences/OSR_us_000_0010_8k' # To be added feature: make this an argument passed in by the user

sample = 'samples/sm3_fi1'
input_filename = sample + '.wav'
output_filename = sample + '.png'

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

amount_overlap = 0.75 # 25-50%

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
	N = 100

	# take histogram

	frame_hist, frame_bin_edges = np.histogram(frame_freq, N, None, True, None, None)


#################################################
# Step 4: Compute the weighted entropy per frame
#################################################

	# Calculate the entropy using 

	# H(X) = -Sum_(k=1)^N p_k log(p_k)

	H = scipy.stats.entropy(frame_hist)
	#print H

	entropy_profile[i] = H

#################################################
# Step 5: Determine entropy threshold
#################################################

# Use the entropy profile to find an appropriate threshold

# only constraint on mu is that its greater than 0
# this value depends on the noise
# for noise, use a large number
mu = 0.5

# compute threshold
gamma = ((max(entropy_profile) - min(entropy_profile))/2 ) + mu*min(entropy_profile)



#################################################
# Step 6: Apply entropy threshold to data
#################################################

thresholded_entropy_profile = np.empty([len(entropy_profile)])

# apply this threshold to our entropy profile, zeroing out values below the threshold
for i in range(len(entropy_profile)):
	if entropy_profile[i] >= gamma:
		thresholded_entropy_profile[i] = entropy_profile[i]
	else:
		thresholded_entropy_profile[i] = 0
'''
plt.plot(thresholded_entropy_profile, linewidth=0.5, color='m')
plt.savefig('entropy.png')
plt.clf()
'''
smoothed_entropy = np.copy(thresholded_entropy_profile)

for i in range(len(smoothed_entropy)-1):
	if i == 0:
		print 'first element'
	else:
		value = thresholded_entropy_profile[i]
		if value == 0:
			if thresholded_entropy_profile[i-1] != 0. and thresholded_entropy_profile[i+1] != 0.:
				smoothed_entropy[i] = thresholded_entropy_profile[i-1]
		else:
			if (thresholded_entropy_profile[i-1] == 0.) and (thresholded_entropy_profile[i+1] == 0.):
				smoothed_entropy[i] = 0


plt.plot(thresholded_entropy_profile, linewidth=2, color='m')
plt.plot(smoothed_entropy, 'k--', linewidth=1)
plt.savefig('entropy.png')
plt.clf()


# due to artifacts, there may be false positives or negatives
# add more criteria...

#################################################
# Step 7: Apply speech segment size constraint
#################################################

# We make an assumption here that humans do not produce very short duration sounds
# So impose a requirement that the segments have a minimum length, l_i
# l_i = e_i - s_i 
# where s_i is the starting point of the ith frame and e_i is the ending

# l_i should be the shortest phoneme/phone and is a function of sampling freq.

# additionally, we want to merge very short sounds into a single segment

#################################################
# Step 8: Put it all  back together
#################################################

# Build a step function where it is 1 when thresholded entropy is non-zero
'''
speech_present = data.max()
no_speech = 0

# Need to create a way to correspond sample to its window

result = np.empty([len(data),3])

window_counter = 0
current_window_length = frame_length

# remember frame_length is the number of samples in each frame

for i in range(len(data)):
	if i < current_window_length:
		result[i,0] = window_counter
	else:
		window_counter += 1
		result[i,0] = window_counter
		current_window_length = current_window_length + frame_length

	this_windows_entropy = thresholded_entropy_profile[window_counter]
	result[i,2] = this_windows_entropy

	if this_windows_entropy != 0:
		result[i,1] = speech_present
	else:
		result[i,1] = no_speech

final_fig = plt.figure(figsize=(8, 4), dpi=300)
plt.plot(data, linewidth=0.1, color='m')
plt.plot(result[:,1], linewidth=0.5)
#plt.plot(result[:,2], linewidth=0.5)
plt.savefig(output_filename)
plt.clf()
'''