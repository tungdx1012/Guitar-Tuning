import numpy as np
import scipy.fftpack
import tkinter as tk
import copy
import scipy.io.wavfile as wav

# General settings that can be changed by the user
SAMPLE_FREQ = 48000 # sample frequency in Hz
WINDOW_SIZE = 48000 # window size of the DFT in samples
WINDOW_STEP = 12000 # step size of window
NUM_HPS = 5 # max number of harmonic product spectrums
POWER_THRESH = 1e-6 # tuning is activated if the signal power exceeds this threshold
CONCERT_PITCH = 440 # defining a1
WHITE_NOISE_THRESH = 0.2 # everything under WHITE_NOISE_THRESH*avg_energy_per_freq is cut off

WINDOW_T_LEN = WINDOW_SIZE / SAMPLE_FREQ # length of the window in seconds
SAMPLE_T_LENGTH = 1 / SAMPLE_FREQ # length between two samples in seconds
DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE # frequency step width of the interpolated DFT
OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

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

HANN_WINDOW = np.hanning(WINDOW_SIZE)
def callback(indata, frames, time, status):
  # define static variables
  if not hasattr(callback, "window_samples"):
    callback.window_samples = [0 for _ in range(WINDOW_SIZE)]
  if not hasattr(callback, "noteBuffer"):
    callback.noteBuffer = ["1","2"]

  if status:
    print(status)
    return
  if indata.any():
    # Reshape indata to have the same number of dimensions as callback.window_samples
    indata = indata.reshape(-1)
    callback.window_samples = np.concatenate((callback.window_samples, indata))  # append new samples
    callback.window_samples = callback.window_samples[len(indata):]  # remove old samples

    # skip if signal power is too low
    signal_power = (np.linalg.norm(callback.window_samples, ord=2) ** 2) / len(callback.window_samples)
    if signal_power < POWER_THRESH:
      return

    # avoid spectral leakage by multiplying the signal with a hann window
    hann_samples = callback.window_samples * HANN_WINDOW
    magnitude_spec = abs(scipy.fftpack.fft(hann_samples)[:len(hann_samples) // 2])

    # supress mains hum, set everything below 62Hz to zero
    for i in range(int(62 / DELTA_FREQ)):
      magnitude_spec[i] = 0

    # calculate average energy per frequency for the octave bands
    # and suppress everything below it
    for j in range(len(OCTAVE_BANDS) - 1):
      ind_start = int(OCTAVE_BANDS[j] / DELTA_FREQ)
      ind_end = int(OCTAVE_BANDS[j + 1] / DELTA_FREQ)
      ind_end = ind_end if len(magnitude_spec) > ind_end else len(magnitude_spec)
      avg_energy_per_freq = (np.linalg.norm(magnitude_spec[ind_start:ind_end], ord=2) ** 2) / (ind_end - ind_start)
      avg_energy_per_freq = avg_energy_per_freq ** 0.5
      for i in range(ind_start, ind_end):
        magnitude_spec[i] = magnitude_spec[i] if magnitude_spec[i] > WHITE_NOISE_THRESH * avg_energy_per_freq else 0

    # interpolate spectrum
    mag_spec_ipol = np.interp(np.arange(0, len(magnitude_spec), 1 / NUM_HPS), np.arange(0, len(magnitude_spec)),magnitude_spec)
    mag_spec_ipol = mag_spec_ipol / np.linalg.norm(mag_spec_ipol, ord=2)  # normalize it

    hps_spec = copy.deepcopy(mag_spec_ipol)

    # calculate the HPS
    for i in range(NUM_HPS):
      tmp_hps_spec = np.multiply(hps_spec[:int(np.ceil(len(mag_spec_ipol) / (i + 1)))], mag_spec_ipol[::(i + 1)])
      if not any(tmp_hps_spec):
        break
      hps_spec = tmp_hps_spec

    max_ind = np.argmax(hps_spec)
    max_freq = max_ind * (SAMPLE_FREQ / WINDOW_SIZE) / NUM_HPS

    closest_note, closest_pitch = find_closest_note(max_freq)
    max_freq = round(max_freq, 1)
    closest_pitch = round(closest_pitch, 1)

    callback.noteBuffer.insert(0, closest_note)  # note that this is a ringbuffer
    callback.noteBuffer.pop()

    # Update the label text
    if callback.noteBuffer.count(callback.noteBuffer[0]) == len(callback.noteBuffer):
      show_popup(f"{closest_note}  {max_freq:.1f}/{closest_pitch:.1f}")
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