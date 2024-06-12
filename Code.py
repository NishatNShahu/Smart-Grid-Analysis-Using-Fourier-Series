import numpy as np
import matplotlib.pyplot as plt

def generate_energy_data(num_points, sampling_rate, frequency, amplitude, noise_factor):
    time = np.arange(0, num_points) / sampling_rate
    energy_data = amplitude * np.sin(2 * np.pi * frequency * time) + noise_factor * np.random.randn(num_points)
    return time, energy_data

def apply_fourier_transform(time, energy_data, sampling_rate):
    n = len(energy_data)
    frequencies = np.fft.fftfreq(n, d=1/sampling_rate)
    energy_fft = np.fft.fft(energy_data)
    return frequencies, energy_fft

def filter_high_frequency_components(frequencies, energy_fft, threshold):
    energy_fft_filtered = energy_fft.copy()
    energy_fft_filtered[np.abs(frequencies) > threshold] = 0
    return energy_fft_filtered

def optimize_smart_grid(energy_data, sampling_rate, frequency_threshold):
    time = np.arange(0, len(energy_data)) / sampling_rate

    frequencies, energy_fft = apply_fourier_transform(time, energy_data, sampling_rate)
    energy_fft_filtered = filter_high_frequency_components(frequencies, energy_fft, frequency_threshold)

    optimized_energy_data = np.fft.ifft(energy_fft_filtered).real

    return time, energy_data, optimized_energy_data

# Generate synthetic energy consumption data
np.random.seed(42)  # For reproducibility
sampling_rate = 100  # Hz
frequency_threshold = 5  # Hz
num_points = 1000

# Generate synthetic energy data with a fundamental frequency of 0.5 Hz and noise
time, original_energy_data = generate_energy_data(num_points, sampling_rate, frequency=0.5, amplitude=50, noise_factor=10)

# Example usage
time, original_energy_data, optimized_energy_data = optimize_smart_grid(original_energy_data, sampling_rate, frequency_threshold)

# Plotting results
plt.figure(figsize=(12, 6))
plt.plot(time, original_energy_data, label='Original Energy Data')
plt.plot(time, optimized_energy_data, label='Optimized Energy Data')
plt.title('Smart Grid Optimization using Fourier Analysis')
plt.xlabel('Time (s)')
plt.ylabel('Energy Consumption')
plt.legend()
plt.show()
