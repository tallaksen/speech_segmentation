''' Speech detection using entropy-based algorithm described in:
'A robust algorithm for detecting speech segments using an entropic contrast',
Waheed et al.

Steps:
1. Pre-processing stage (add a pre-emph filter + other filter)
2. Choose speech window
3. Determine short time prob. distr. using histogram
4. Threshold the entropy profile
5. Determine the adaptive threshold for entropy profile
6. Compute weighted entropy of speech frame
7. Apply speech segment size constraint
8. Appy intra-segment distance constraint
9. Yay!
'''

#!/usr/bin/python

import numpy as np
import scipy
from scipy.fftpack import rfft, irfft, fftfreq
from scipy.io import wavfile
from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta
import matplotlib.pyplot as plt
import python_speech_features as speech

samples = ['samples/sf1_fi1']

for sample in samples:
	input_filename = sample + '.wav'

	rate, data = wavfile.read(input_filename)

#################################################
# 			Step 1: Filters
#################################################

# Pre-emphasis:

	preemp_data = speech.sigproc.preemphasis(data, 0.97)

#	fig = plt.plot(data, 'm', filtered_data, 'b')
#	plt.setp(fig, linewidth=0.25)
#	plt.savefig('foo.png')

# Add a band-pass filter to remove constant and low freq background components,
# and high freq noise and speech harmonics

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

	recovered_signal = scipy.ifft(shifted)

#################################################
# 			Step 1: Divide into windows
#################################################

	# Divide the speech into overlapping frames with 25-50% overlap

	frames = speech.sigproc.framesig(recovered_signal, )
