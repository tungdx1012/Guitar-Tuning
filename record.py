import sounddevice as sd
import scipy.io.wavfile
import time

SAMPLE_FREQ = 44100 # Sampling frequency of the recording
SAMPLE_DUR = 2  # Duration of the recoding

print("Grab your guitar!")
time.sleep(1) # Gives you a second to grab your guitar ;)

myRecording = sd.rec(SAMPLE_DUR * SAMPLE_FREQ, samplerate=SAMPLE_FREQ, channels=1,dtype='float64')
print("Recording audio")
sd.wait()

sd.play(myRecording, SAMPLE_FREQ)
print("Playing audio")
sd.wait()

scipy.io.wavfile.write('voice.wav', SAMPLE_FREQ, myRecording)