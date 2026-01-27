import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense, GRU, Dropout, Bidirectional, Conv1D, TimeDistributed, Flatten, BatchNormalization, Activation, Reshape
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.metrics import BinaryAccuracy

# --- CONFIGURATION ---
FS = 10
G_DELAY = 4
tau = 0.7
SNR_RANGE = range(0, 15, 2) # 0, 2, ..., 14 dB
N_train = 50000
N_test_per_snr = 30000

# Osman's hPSF (Golden Reference)
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

def generate_data(n_syms, snr):
    bits = np.random.randint(0, 2, n_syms)
    symbols = 2 * bits - 1
    s_up = np.zeros(step * n_syms)
    s_up[::step] = symbols
    tx = np.convolve(hPSF, s_up, mode='full')
    EbN0 = 10**(snr/10)
    n0 = 10**(-snr/10)
    noise = np.sqrt(n0/2) * np.random.randn(len(tx))
    rx_noisy = tx + noise
    mf = np.convolve(hPSF, rx_noisy, mode='full')
    mf = mf / np.std(mf)
    return mf, bits

# --- DATA & MODEL PREP ---
L_sym = 4
offsets_sym = np.arange(-L_sym, L_sym+1) * step
local_window = np.arange(-3, 4) 

def extract_inputs(mf_signal, n_syms):
    X1 = np.zeros((n_syms, len(offsets_sym), 1))
    X2 = np.zeros((n_syms, len(offsets_sym), len(local_window)))
    
    valid_indices = np.arange(n_syms) * step + p_loc
    max_idx = len(mf_signal) - 1
    valid_mask = []
    
    for i, center in enumerate(valid_indices):
        sym_locs = center + offsets_sym
        if np.all(sym_locs + min(local_window) >= 0) and \
           np.all(sym_locs + max(local_window) <= max_idx):
            X1[i, :, 0] = mf_signal[sym_locs]
            for t in range(len(offsets_sym)):
                X2[i, t, :] = mf_signal[sym_locs[t] + local_window]
            valid_mask.append(i)
    return X1[valid_mask], X2[valid_mask], valid_mask

# --- MODELS ---
def build_baseline_gru():
    model = Sequential()
    model.add(Input(shape=(9, 1)))
    model.add(Bidirectional(GRU(32, activation='tanh', recurrent_dropout=0)))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Nadam(0.001), loss='binary_crossentropy', metrics=[BinaryAccuracy()])
    return model

def build_proposed_crnn():
    inp = Input(shape=(9, 7))
    x = Reshape((9, 7, 1))(inp)
    x = TimeDistributed(Conv1D(filters=32, kernel_size=3, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(Flatten())(x) 
    x = Bidirectional(GRU(64, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    model.compile(optimizer=Nadam(0.001), loss='binary_crossentropy', metrics=[BinaryAccuracy()])
    return model

# --- TRAINING (Train once at 10dB) ---
print(f"Training models at {10} dB...")
mf_train, y_train = generate_data(N_train, 10)
X1_tr, X2_tr, idx_tr = extract_inputs(mf_train, N_train)
Y_tr = y_train[idx_tr]

model_base = build_baseline_gru()
model_prop = build_proposed_crnn()

print("Training Baseline (Symbol-Rate)...")
model_base.fit(X1_tr, Y_tr, epochs=15, batch_size=128, verbose=0)
print("Training Proposed (Hybrid CRNN)...")
model_prop.fit(X2_tr, Y_tr, epochs=20, batch_size=128, verbose=0)

# --- TESTING LOOP ---
BER_base = []
BER_prop = []

print("\nStarting SNR Sweep...")
print("SNR | Baseline | Proposed | Improvement")
print("----|----------|----------|------------")

for snr in SNR_RANGE:
    mf_test, y_test = generate_data(N_test_per_snr, snr)
    X1_te, X2_te, idx_te = extract_inputs(mf_test, N_test_per_snr)
    Y_te = y_test[idx_te]
    
    p1 = model_base.predict(X1_te, verbose=0)
    b1 = np.mean((p1.flatten() > 0.5) != Y_te)
    
    p2 = model_prop.predict(X2_te, verbose=0)
    b2 = np.mean((p2.flatten() > 0.5) != Y_te)
    
    imp = 0
    if b1 > 0: imp = (1 - b2/b1)*100
    
    BER_base.append(b1)
    BER_prop.append(b2)
    
    print(f" {snr:2d} | {b1:.2e} | {b2:.2e} | {imp:5.1f}%")

# --- PLOTTING ---
plt.figure(figsize=(10, 6))
plt.semilogy(SNR_RANGE, BER_base, 'r--o', linewidth=2, label='Baseline (Symbol-Rate Bi-GRU)')
plt.semilogy(SNR_RANGE, BER_prop, 'b-s', linewidth=2, label='Proposed (Fractional CRNN)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.title(f'FTN Detection Performance (Tau={tau})')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.legend()
plt.savefig('final_results_waterfall.png')
print("\nPlot saved as 'final_results_waterfall.png'")