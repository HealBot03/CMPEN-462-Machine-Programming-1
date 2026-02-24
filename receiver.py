import numpy as np

# 1. setup parameters
fs = 100  # sampling rate
fc = 20   # carrier freq
N = 3000  # total samples

# load incoming signal
rx_signal = np.loadtxt('input.txt')

# create time vector
t = np.arange(N) / fs

# 2. downconversion
# extract i and q streams via carrier wave multiplication
I = rx_signal * np.cos(2 * np.pi * fc * t) # I=rx_signal × cos(2π*fc*​n*T)
Q = rx_signal * np.sin(2 * np.pi * fc * t) # Q=rx_signal × sin(2π*fc*​n*T)

# 3. filter
# fft to frequency domain
I_fft = np.fft.fft(I)
Q_fft = np.fft.fft(Q)

# get frequency axis bins
freqs = np.fft.fftfreq(N, d=1/fs)

# mask outside +/- 5.1 hz
cutoff = 5.1
mask = np.abs(freqs) > cutoff

# zero out high freq static
I_fft[mask] = 0
Q_fft[mask] = 0

# ifft back to time domain and double to fix amplitude
I_filtered = np.fft.ifft(I_fft).real * 2
Q_filtered = np.fft.ifft(Q_fft).real * 2

# 4. downsample
# grab every 10th sample to match 10hz symbol rate
I_down = I_filtered[::10]
Q_down = Q_filtered[::10]

# merge to complex symbols
rx_complex = I_down + 1j * Q_down

# 5. correlate
# load preamble and swap i for j
with open('preamble.txt', 'r') as file:
    raw_text = file.read().replace('i', 'j')
preamble = np.array([complex(c) for c in raw_text.split()])

# slide preamble over signal to find start index
correlation = np.convolve(rx_complex, np.conj(preamble[::-1]), mode='valid')
start_index = np.argmax(np.abs(correlation))

# slice off preamble and dead air
symbols = rx_complex[start_index + 50:]

# 6. demodulate
# map points to 16-qam binary strings
mapping = {
    ( 3 + 3j): '0000', ( 1 + 3j): '0001', (-1 + 3j): '0011', (-3 + 3j): '0010',
    ( 3 + 1j): '0100', ( 1 + 1j): '0101', (-1 + 1j): '0111', (-3 + 1j): '0110',
    ( 3 - 1j): '1100', ( 1 - 1j): '1101', (-1 - 1j): '1111', (-3 - 1j): '1110',
    ( 3 - 3j): '1000', ( 1 - 3j): '1001', (-1 - 3j): '1011', (-3 - 3j): '1010'
}

all_bits = ""
# snap points to closest grid location
for s in symbols:
    closest_point = min(mapping.keys(), key=lambda p: np.abs(s - p))
    all_bits += mapping[closest_point]

# 7. ASCII to text
# group 8 bits to char
decoded_text = ""
for i in range(0, len(all_bits), 8):
    byte = all_bits[i:i+8]
    if len(byte) == 8:
        decoded_text += chr(int(byte, 2))

print("Decoded Text:", decoded_text)