import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import get_window, sawtooth

def generate_audio(filename, duration=2.0, freq=440, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    vibrato = 3.0 * np.sin(2 * np.pi * 5.0 * t)
    audio = 0.5 * sawtooth(2 * np.pi * (freq + vibrato) * t)
    audio += 0.2 * np.sin(2 * np.pi * 800 * t)  
    audio += 0.15 * np.sin(2 * np.pi * 1200 * t) 
    
    envelope = np.ones_like(audio)
    fade_len = int(0.1 * sr)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    
    final_audio = audio * envelope
    
    wav.write(filename, sr, (final_audio * 32767).astype(np.int16))
    return final_audio

def pitch_shift(input_file, output_file, alpha):
    sr, data = wav.read(input_file)
    x = data.astype(float) / 32768.0
    
    N = 2048
    H = 512
    window = get_window('hann', N)
    output = np.zeros(len(x) + N)
    
    num_frames = (len(x) - N) // H
    
    for m in range(num_frames):
        start = m * H
        x_frame = x[start : start + N] * window
        X = np.fft.fft(x_frame)
        
        Y = np.zeros(N, dtype=complex)
        for k in range(N // 2):
            src_k = k / alpha
            k0 = int(src_k)
            k1 = k0 + 1
            delta = src_k - k0
            
            if k1 < N // 2:
                mag = (1 - delta) * np.abs(X[k0]) + delta * np.abs(X[k1])
                phase = np.angle(X[k0]) 
                Y[k] = mag * np.exp(1j * phase)
                
        for k in range(1, N // 2):
            Y[N - k] = np.conj(Y[k])
            
        y_frame = np.fft.ifft(Y).real
        output[start : start + N] += y_frame * window
        
    wav.write(output_file, sr, (output * 32767).astype(np.int16))
    return x, output[:len(x)], sr

base_filename = "base.wav"
exp1_filename = "expansion.wav"
exp2_filename = "compression.wav"
exp3_filename = "invertibility.wav"
exp4_filename = "interpolation.wav"

generate_audio(base_filename, freq=440)
pitch_shift(base_filename, exp1_filename, 2.0)
pitch_shift(base_filename, exp2_filename, 0.5)
pitch_shift(exp1_filename, exp3_filename, 0.5)
pitch_shift(base_filename, exp4_filename, 1.2)

def calc_spectrum(filename):
    sr, data = wav.read(filename)
    x = data.astype(float) / 32768.0
    mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1/sr)
    return freqs, 20 * np.log10(mag + 1e-10)

f_base, m_base = calc_spectrum(base_filename)
f_exp1, m_exp1 = calc_spectrum(exp1_filename)
f_exp2, m_exp2 = calc_spectrum(exp2_filename)
f_exp3, m_exp3 = calc_spectrum(exp3_filename)
f_exp4, m_exp4 = calc_spectrum(exp4_filename)

plot_size = (6, 4) 

plt.figure(figsize=plot_size)
plt.plot(f_base, m_base, label='Original (440Hz)', color='blue', alpha=0.5)
plt.plot(f_exp1, m_exp1, label='Shifted Up (880Hz)', color='red', alpha=0.8)
plt.xlim(0, 2000)
plt.ylabel("Magnitude (dB)")
plt.xlabel("Frequency (Hz)")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("expansion.png", dpi=300)
plt.close() 

plt.figure(figsize=plot_size)
plt.plot(f_base, m_base, label='Original (440Hz)', color='blue', alpha=0.5)
plt.plot(f_exp2, m_exp2, label='Shifted Down (220Hz)', color='green', alpha=0.8)
plt.xlim(0, 2000)
plt.ylabel("Magnitude (dB)")
plt.xlabel("Frequency (Hz)")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("compression.png", dpi=300)
plt.close()

plt.figure(figsize=plot_size)
plt.plot(f_base, m_base, label='Original Input', color='blue', alpha=0.6, linewidth=2)
plt.plot(f_exp3, m_exp3, label='Restored Output', color='purple', alpha=0.6, linestyle='--')
plt.xlim(0, 2000)
plt.ylabel("Magnitude (dB)")
plt.xlabel("Frequency (Hz)")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("invertibility.png", dpi=300)
plt.close()

plt.figure(figsize=plot_size)
plt.plot(f_base, m_base, label='Original (440Hz)', color='blue', alpha=0.5)
plt.plot(f_exp4, m_exp4, label='Shifted 1.2x (~528Hz)', color='orange', alpha=0.8)
plt.xlim(0, 2000)
plt.ylabel("Magnitude (dB)")
plt.xlabel("Frequency (Hz)")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("interpolation.png", dpi=300)
plt.close()