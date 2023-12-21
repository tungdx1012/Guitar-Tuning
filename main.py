import numpy as np
import scipy.fftpack
import tkinter as tk
import scipy.io.wavfile as wav

# General settings
SAMPLE_FREQ = 44100 # sample frequency in Hz
WINDOW_SIZE = 44100 # window size of the DFT in samples
WINDOW_STEP = 21050 # step size of window
WINDOW_T_LEN = WINDOW_SIZE / SAMPLE_FREQ # length of the window in seconds
SAMPLE_T_LENGTH = 1 / SAMPLE_FREQ # length between two samples in seconds
POWER_THRESH = 1e-6
windowSamples = [0 for _ in range(WINDOW_SIZE)]

# This function finds the closest note for a given pitch
# Returns: note (e.g. A4, G#3, ..), pitch of the tone
CONCERT_PITCH = 440
ALL_NOTES = ["A","A#","B","C","C#","D","D#","E","F","F#","G","G#"]
def find_closest_note(pitch):
  i = int(np.round(np.log2(pitch/CONCERT_PITCH)*12))
  closest_note = ALL_NOTES[i%12] + str(4 + (i + 9) // 12)
  closest_pitch = CONCERT_PITCH*2**(i/12)
  return closest_note, closest_pitch

# Create a new Tkinter window
window = tk.Tk()
window.title("Note Detection")
window.geometry("300x100")  # Set the size of the window

# Create a label with large font size
label = tk.Label(window, text="", font=("Helvetica", 20))
label.pack()

def show_popup(message):
    label.config(text=message)

# Load the audio file
fs, data = wav.read('D#.wav')

# Process the audio data
windowSamples = np.concatenate((windowSamples, data.flatten())) # append new samples
windowSamples = windowSamples[len(data):] # remove old samples

# skip if signal power is too low
signal_power = (np.linalg.norm(windowSamples, ord=2) ** 2) / len(windowSamples)
if signal_power >= POWER_THRESH:
    absFreqSpectrum = abs( scipy.fftpack.fft(windowSamples)[:len(windowSamples)//2] )

    for i in range(int(62/(SAMPLE_FREQ/WINDOW_SIZE))):
        absFreqSpectrum[i] = 0 #suppress mains hum

    maxInd = np.argmax(absFreqSpectrum)
    maxFreq = maxInd * (SAMPLE_FREQ/WINDOW_SIZE)
    closestNote, closestPitch = find_closest_note(maxFreq)

    # Update the label text
    show_popup(f"{closestNote}  {maxFreq:.1f}/{closestPitch:.1f}")

window.mainloop()