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
  closest_note = ALL_NOTES[i%12]
  closest_pitch = CONCERT_PITCH*2**(i/12)
  return closest_note, closest_pitch

# Create a new Tkinter window
window = tk.Tk()
window.title("Note Detection")
window.geometry("500x300")  # Set the size of the window

# Create a label with large font size
label = tk.Label(window, text="", font=("Helvetica", 20))
label.pack()

# Create a list of specific .wav files
audio_files = ['D.wav', 'D#.wav', 'F#.wav', 'G.wav', 'G#.wav']

def show_popup(message):
    label.config(text=message)

def callback(indata, frames, time, status):
  global windowSamples
  if status:
    print(status)
  if indata.any():
    windowSamples = np.concatenate((windowSamples,indata[:, 0])) # append new samples
    windowSamples = windowSamples[len(indata[:, 0]):] # remove old samples
    magnitudeSpec = abs( scipy.fftpack.fft(windowSamples)[:len(windowSamples)//2] )

    for i in range(int(62/(SAMPLE_FREQ/WINDOW_SIZE))):
      magnitudeSpec[i] = 0 #suppress mains hum

    maxInd = np.argmax(magnitudeSpec)
    maxFreq = maxInd * (SAMPLE_FREQ/WINDOW_SIZE)
    closestNote, closestPitch = find_closest_note(maxFreq)

    # Update the label text
    show_popup(f"{closestNote}  {maxFreq:.1f}/{closestPitch:.1f}")
  else:
    show_popup(f"...")

def process_audio_file(audio_file):
  fs, data = wav.read(audio_file)
  # Process the audio data
  for i in range(0, len(data), WINDOW_STEP):
    callback(data[i:i + WINDOW_STEP], WINDOW_STEP, None, None)

# Create a function to display the list of audio files and let the user choose one
def choose_audio_file():
  chosen_audio_file = listbox.get(listbox.curselection())
  process_audio_file(chosen_audio_file)

# Create a Listbox to display the list of audio files
listbox = tk.Listbox(window, height=10, width=30)
for audio_file in audio_files:
    listbox.insert(tk.END, audio_file)
listbox.pack()

# Create a Button to let the user choose an audio file
button = tk.Button(window, text="Choose an audio file...", command=choose_audio_file)
button.pack()

window.mainloop()