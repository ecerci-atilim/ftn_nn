import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, Dropout, Bidirectional
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.metrics import BinaryAccuracy

# --- 1. CONFIGURATION (OSMAN'S REPO CONSTANTS) ---
# Bu ayarlar "garanti" ayarlardır, senkronizasyon kaymaz.
FS = 10
G_DELAY = 4
tau = 0.7
SNR_dB = 10
N_train = 50000
N_test = 20000

# Osman's hPSF Filter Coefficients
hPSF = np.array([
    0.00376269812419313, 0.00471337435592727, 0.00482166178880308, 0.00394250078618065, 0.00208448894874903,
    -0.000571498551744591, -0.00367916587359419, -0.00676092379084138, -0.00926712018759014, -0.0106541973225301,
    -0.0104721244330486, -0.00844906382948057, -0.00456035879416295, 0.000930166019828655, 0.00746590728241009,
    0.0142352508683435, 0.0202511199226200, 0.0244662284202333, 0.0259124907726173, 0.0238486447341953,
    0.0178975045338370, 0.00815385367042497, -0.00475397518075267, -0.0196614000546128, -0.0349300149982486,
    -0.0485738175029959, -0.0584399856646967, -0.0624276438876449, -0.0587223182124205, -0.0460204126242920,
    -0.0237175706573207, 0.00796258376002411, 0.0479172896810874, 0.0942011356918112, 0.144150842824562,
    0.194589161559187, 0.242089589947497, 0.283276330059327, 0.315129331838160, 0.335263018086804,
    0.342149545262556,
    0.335263018086804, 0.315129331838160, 0.283276330059327, 0.242089589947497, 0.194589161559187,
    0.144150842824562, 0.0942011356918112, 0.0479172896810874, 0.00796258376002411, -0.0237175706573207,
    -0.0460204126242920, -0.0587223182124205, -0.0624276438876449, -0.0584399856646967, -0.0485738175029959,
    -0.0349300149982486, -0.0196614000546128, -0.00475397518075267, 0.00815385367042497, 0.0178975045338370,
    0.0238486447341953, 0.0259124907726173, 0.0244662284202333, 0.0202511199226200, 0.0142352508683435,
    0.00746590728241009, 0.000930166019828655, -0.00456035879416295, -0.00844906382948057, -0.0104721244330486,
    -0.0106541973225301, -0.00926712018759014, -0.00676092379084138, -0.00367916587359419, -0.000571498551744591,
    0.00208448894874903, 0.00394250078618065, 0.00482166178880308, 0.00471337435592727, 0.00376269812419313
])

step = int(tau * FS)
p_loc = 2 * G_DELAY * FS

# --- 2. DATA GENERATION ---
def generate_data(n_syms, snr):
    bits = np.random.randint(0, 2, n_syms)
    symbols = 2 * bits - 1
    s_up = np.zeros(step * n_syms)
    s_up[::step] = symbols
    tx = np.convolve(hPSF, s_up, mode='full') # numpy convolve (safe)
    
    EbN0 = 10**(snr/10)
    n0 = 10**(-snr/10)
    noise = np.sqrt(n0/2) * np.random.randn(len(tx))
    rx_noisy = tx + noise
    
    mf = np.convolve(hPSF, rx_noisy, mode='full')
    mf = mf / np.std(mf) # Normalize
    return mf, bits

print(f"Generating Data (Tau={tau}, SNR={SNR_dB}dB)...")
mf_train, y_train = generate_data(N_train, SNR_dB)
mf_test, y_test = generate_data(N_test, SNR_dB)

# --- 3. FEATURE EXTRACTION (YOUR WINDOW STRATEGY) ---

# Method 1: Symbol-Rate (Original)
# Sequence: 9 steps, 1 feature per step (Just the peak)
L_sym = 4
offsets_sym = np.arange(-L_sym, L_sym+1) * step

# Method 2: Fractional Window (Dense Information)
# Sequence: 9 steps (SAME AS METHOD 1)
# Features: 7 features per step (Waveform around each peak)
# Her bir sembol için +-3 sample (toplam 7) alıyoruz
local_window = np.arange(-3, 4) 

def extract_inputs(mf_signal, n_syms):
    # Method 1 Input: (N, 9, 1) -> 1D sequence
    X1 = np.zeros((n_syms, len(offsets_sym), 1))
    
    # Method 2 Input: (N, 9, 7) -> 7D sequence (High Info)
    X2 = np.zeros((n_syms, len(offsets_sym), len(local_window)))
    
    # Başlangıç noktası hesabı (Repo ile aynı)
    valid_indices = np.arange(n_syms) * step + p_loc
    max_idx = len(mf_signal) - 1
    
    valid_mask = []
    
    for i, center in enumerate(valid_indices):
        # Base symbol locations (9 steps)
        sym_locs = center + offsets_sym
        
        # Check bounds using the widest window
        if np.all(sym_locs + min(local_window) >= 0) and \
           np.all(sym_locs + max(local_window) <= max_idx):
            
            # Fill Method 1 (Just the peaks)
            X1[i, :, 0] = mf_signal[sym_locs]
            
            # Fill Method 2 (Waveform around each peak)
            for t in range(len(offsets_sym)): # Loop over 9 time steps
                # For each time step, grab 7 samples around it
                # Bu "Feature Engineering"dir. GRU her adımda 7 bilgi görür.
                X2[i, t, :] = mf_signal[sym_locs[t] + local_window]
            
            valid_mask.append(i)
            
    return X1[valid_mask], X2[valid_mask], valid_mask

print("Extracting features...")
X1_train, X2_train, idx_train = extract_inputs(mf_train, N_train)
Y_train = y_train[idx_train]

X1_test, X2_test, idx_test = extract_inputs(mf_test, N_test)
Y_test = y_test[idx_test]

print(f"Method 1 Input Shape: {X1_train.shape} (Time=9, Feat=1)")
print(f"Method 2 Input Shape: {X2_train.shape} (Time=9, Feat=7)")

# --- 4. MODEL DEFINITION ---
def build_gru(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    # Bidirectional GRU (ISI Simetriğini yakalar)
    # Return sequences=False (Sadece son kararı veriyoruz)
    model.add(Bidirectional(GRU(32, activation='tanh', recurrent_dropout=0)))
    
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Nadam(0.001), loss='binary_crossentropy', metrics=[BinaryAccuracy()])
    return model

print("\n--- Training Method 1 (Symbol-Rate) ---")
model1 = build_gru((9, 1))
model1.fit(X1_train, Y_train, epochs=15, batch_size=128, verbose=1, validation_split=0.1)

print("\n--- Training Method 2 (Fractional Features) ---")
model2 = build_gru((9, 7)) # Same time steps, MUCH MORE INFO per step!
model2.fit(X2_train, Y_train, epochs=15, batch_size=128, verbose=1, validation_split=0.1)

# --- 5. EVALUATION ---
print("\nEvaluating...")
p1 = model1.predict(X1_test, verbose=0)
p2 = model2.predict(X2_test, verbose=0)

ber1 = np.mean((p1.flatten() > 0.5) != Y_test)
ber2 = np.mean((p2.flatten() > 0.5) != Y_test)

print("="*40)
print(f"RESULTS (tau={tau}, SNR={SNR_dB}dB)")
print(f"Method 1 (Symbol-Rate): BER = {ber1:.2e}")
print(f"Method 2 (Fractional):  BER = {ber2:.2e}")
print("="*40)