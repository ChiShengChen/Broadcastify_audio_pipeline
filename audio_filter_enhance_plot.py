import os
import shutil
import torch
import torchaudio
import librosa
import librosa.display
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import torchaudio.transforms as T

# Audio processing parameters
SAMPLE_RATE = 16000  # Target sample rate
HIGH_PASS_CUTOFF = 300  # High-pass filter cutoff frequency
ENHANCE_FREQ = 1000  # Speech enhancement filter central frequency

from scipy.signal import butter, lfilter

def butter_highpass(cutoff, sample_rate, order=5):
    """Create a Butterworth high-pass filter."""
    nyquist = 0.5 * sample_rate  # Nyquist frequency
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a

def apply_highpass_filter(waveform, sample_rate, cutoff_freq=300, order=5):
    """Apply a Butterworth high-pass filter to an audio waveform."""
    b, a = butter_highpass(cutoff_freq, sample_rate, order)
    filtered_waveform = lfilter(b, a, waveform.numpy())
    return torch.tensor(filtered_waveform)

def butter_bandpass(lowcut, highcut, sample_rate, order=5):
    """Create a Butterworth band-pass filter."""
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band", analog=False)
    return b, a

def speech_enhance_filter(waveform, sample_rate, lowcut=300, highcut=3000, order=5):
    """Applies a Butterworth band-pass filter for speech enhancement."""
    b, a = butter_bandpass(lowcut, highcut, sample_rate, order)
    filtered_waveform = lfilter(b, a, waveform.numpy())
    return torch.tensor(filtered_waveform)


def wiener_filter(waveform):
    """Applies Wiener filtering for noise reduction."""
    waveform_np = waveform.numpy()
    filtered = scipy.signal.wiener(waveform_np, mysize=5, noise=1e-6)
    return torch.tensor(filtered)

# Visualization function
def plot_audio_features(waveform, sample_rate, output_folder, filename_prefix):
    """Generates and saves waveform, spectrogram, mel spectrogram, and FFT plots."""
    os.makedirs(output_folder, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # 1. Waveform
    plt.subplot(4, 1, 1)
    librosa.display.waveshow(waveform.numpy(), sr=sample_rate)
    plt.title("Waveform")
    
    # 2. Spectrogram
    plt.subplot(4, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(waveform.numpy())), ref=np.max)
    librosa.display.specshow(D, sr=sample_rate, x_axis="time", y_axis="log")
    plt.title("Spectrogram")

    # 3. Mel Spectrogram
    plt.subplot(4, 1, 3)
    mel_spec = librosa.feature.melspectrogram(y=waveform.numpy(), sr=sample_rate)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    librosa.display.specshow(mel_spec_db, sr=sample_rate, x_axis="time", y_axis="mel")
    plt.title("Mel Spectrogram")

    # 4. FFT (Fourier Transform)
    plt.subplot(4, 1, 4)
    fft_spectrum = np.abs(np.fft.rfft(waveform.numpy()))
    freqs = np.fft.rfftfreq(len(waveform.numpy()), d=1/sample_rate)
    plt.plot(freqs, fft_spectrum, color="red")
    plt.title("FFT Spectrum")
    plt.xlabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{filename_prefix}.png"))
    plt.close()

# Main processing function
def process_audio_files(input_folder):
    """Processes all .wav files in the specified folder."""
    for file in os.listdir(input_folder):
        if file.endswith(".wav"):
            file_path = os.path.join(input_folder, file)
            file_name, _ = os.path.splitext(file)

            # 1. Create a corresponding folder for the file
            output_folder = os.path.join(input_folder, file_name)
            os.makedirs(output_folder, exist_ok=True)

            # 2. Move the file into its folder
            moved_wav_path = os.path.join(output_folder, file)
            shutil.move(file_path, moved_wav_path)

            # Load the audio file
            waveform, sample_rate = torchaudio.load(moved_wav_path)

            # 3. Generate visualizations for the original audio
            plot_audio_features(waveform[0], sample_rate, output_folder, f"{file_name}_original")

            # 4. Apply Wiener Filter + High-Pass Filter + Speech Enhancement
            wiener_filtered = wiener_filter(waveform[0])
            # highpass_filtered = high_pass_filter(wiener_filtered, sample_rate)
            highpass_filtered = apply_highpass_filter(wiener_filtered, sample_rate)

            enhanced_filtered = speech_enhance_filter(highpass_filtered, sample_rate)

            # 5. Save the filtered & enhanced audio file
            filtered_enhanced_path = os.path.join(output_folder, f"{file_name}_filtered_n_enhanced.wav")
            torchaudio.save(filtered_enhanced_path, enhanced_filtered.unsqueeze(0), sample_rate)

            # 6. Generate visualizations for the filtered & enhanced audio
            plot_audio_features(enhanced_filtered, sample_rate, output_folder, f"{file_name}_filtered_n_enhanced")

            # 7. Apply only Speech Enhancement
            # enhanced_speech = speech_enhance_filter(waveform[0], sample_rate)
            enhanced_speech = speech_enhance_filter(waveform[0], sample_rate, lowcut=300, highcut=3000)

            enhanced_path = os.path.join(output_folder, f"{file_name}_enhanced.wav")
            torchaudio.save(enhanced_path, enhanced_speech.unsqueeze(0), sample_rate)

            # 8. Generate visualizations for the enhanced-only audio
            plot_audio_features(enhanced_speech, sample_rate, output_folder, f"{file_name}_enhanced")

    print("All audio processing completed 0w0!")

# Set the path to the folder containing .wav files
input_folder_path = "/media/meow/One Touch/ems_call/merged_audio_test_exp_data_200_vad0_1"  # Change this to your directory path
process_audio_files(input_folder_path)
