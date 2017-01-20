''' Speech detecion using entropy-based algorithm described in:
'A robust algorithm for detecting speech segments using an entropic contrast',
Waheed et al.
'''

#!/usr/bin/python

import numpy as np
import scipy
from scipy.io import wavfile
import matplotlib.pyplot as plt
import python_speech_features as speech

samples = ['samples/sf1_fi1']

for sample in samples:
	input_filename = sample + '.wav'

	rate, data = wavfile.read(input_filename)

	# Apply a pre-emphasis filter 

	filtered_data = speech.sigproc.preemphasis(data, 0.97)

	