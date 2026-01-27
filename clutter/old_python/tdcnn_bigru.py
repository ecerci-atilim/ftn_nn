import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense, GRU, Dropout, Bidirectional, Conv1D, TimeDistributed, Flatten, BatchNormalization, Activation, Reshape
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# --- 1. CONFIGURATION ---
FS = 10
tau = 0.7
SNR_RANGE = [0, 2, 4, 6, 8, 10, 12, 14] 
N_train = 100000 
N_test_per_snr = 50000

# N=8 (17 Steps) - Osman's Setting
L_sym = 8  
step = int(tau * FS)

# --- 2. DATA GENERATION ---
def rcosdesign_numpy(beta, span, sps):
    t = np.arange(-span*sps, span*sps + 1) / sps
    h = np.zeros(len(t))
    for i, val in enumerate(t):
        if val == 0:
            h[i] = 1.0 + beta * (4/np.pi - 1)
        elif abs(val) == 1 / (4 * beta):
            h[i] = (beta / np.sqrt(2)) * ((1 + 2/np.pi) * np.sin(np.pi/(4*beta)) + (1 - 2/np.pi) * np.cos(np.pi/(4*beta)))
        else:
            h[i] = (np.sin(np.pi * val * (1 - beta)) + 4 * beta * val * np.cos(np.pi * val * (1 + beta))) / \
                   (np.pi * val * (1 - (4 * beta * val) ** 2))
    return h / np.linalg.norm(h)

hPSF = rcosdesign_numpy(beta=0.35, span=16, sps=FS) 
p_loc = len(hPSF) // 2 

def generate_data(n_syms, snr):
    bits = np.random.randint(0, 2, n_syms)
    symbols = 2 * bits - 1
    s_up = np.zeros(step * n_syms)
    s_up[::step] = symbols
    tx = np.convolve(s_up, hPSF, mode='full')
    EbN0 = 10**(snr/10)
    n0 = 10**(-snr/10)
    noise = np.sqrt(n0/2) * np.random.randn(len(tx))
    rx_noisy = tx + noise
    mf = np.convolve(rx_noisy, hPSF, mode='full')
    mf = mf / np.std(mf)
    return mf, bits

print(f"Generating Training Data...")
# Training at mixed SNR range to help generalization
# Or stick to high SNR (12dB) to learn pure patterns
mf_train, y_train = generate_data(N_train, 12) 

# --- 3. FEATURE EXTRACTION ---
offsets_sym = np.arange(-L_sym, L_sym+1) * step
local_window = np.arange(-3, 4) # 7 fractional samples

def extract_inputs(mf_signal, n_syms):
    X = np.zeros((n_syms, len(offsets_sym), len(local_window)))
    total_delay = 2 * (len(hPSF) // 2)
    valid_indices = np.arange(n_syms) * step + total_delay
    max_idx = len(mf_signal) - 1
    valid_mask = []
    
    for i, center in enumerate(valid_indices):
        sym_locs = center + offsets_sym
        if np.all(sym_locs + min(local_window) >= 0) and \
           np.all(sym_locs + max(local_window) <= max_idx):
            for t in range(len(offsets_sym)):
                X[i, t, :] = mf_signal[sym_locs[t] + local_window]
            valid_mask.append(i)
    return X[valid_mask], valid_mask

print("Extracting features...")
X_tr, idx_tr = extract_inputs(mf_train, N_train)
Y_tr = y_train[idx_tr]

# --- 4. "PERFORMANCE" CRNN MODEL ---
def build_perf_crnn(input_shape):
    inp = Input(shape=input_shape)
    x = Reshape((input_shape[0], input_shape[1], 1))(inp)
    
    # CNN: Digger Filters, Swish Activation
    x = TimeDistributed(Conv1D(filters=64, kernel_size=3, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('swish'))(x) # Swish is better for signal
    x = TimeDistributed(Flatten())(x)
    
    # Bi-GRU: Deeper Memory
    x = Bidirectional(GRU(128, return_sequences=True))(x) # Sequence to Sequence first
    x = Dropout(0.3)(x)
    x = Bidirectional(GRU(64, return_sequences=False))(x) # Then summarize
    x = Dropout(0.3)(x)
    
    # Dense Refinement
    x = Dense(64, activation='swish')(x)
    x = BatchNormalization()(x)
    
    out = Dense(1, activation='sigmoid')(x)
    
    model = Model(inp, out)
    model.compile(optimizer=Nadam(0.001), loss='binary_crossentropy', metrics=[BinaryAccuracy()])
    return model

model = build_perf_crnn((len(offsets_sym), len(local_window)))
model.summary()

# --- LR SCHEDULER (The Secret Sauce) ---
# Error floor'a takılınca learning rate'i düşürür
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("\n--- Training Model ---")
history = model.fit(X_tr, Y_tr, 
                    epochs=25, 
                    batch_size=256, 
                    verbose=1, 
                    validation_split=0.15,
                    callbacks=[lr_scheduler, early_stop])

# --- 5. WATERFALL TEST ---
BER_res = []
print("\nStarting SNR Sweep...")
print("SNR | BER")
print("----|----")

for snr in SNR_RANGE:
    mf_te, y_te = generate_data(N_test_per_snr, snr)
    X_te, idx_te = extract_inputs(mf_te, N_test_per_snr)
    Y_te_true = y_te[idx_te]
    
    pred = model.predict(X_te, verbose=0)
    ber = np.mean((pred.flatten() > 0.5) != Y_te_true)
    BER_res.append(ber)
    print(f" {snr:2d} | {ber:.2e}")

# Save Plot
plt.figure(figsize=(10, 6))
plt.semilogy(SNR_RANGE, BER_res, 'b-s', linewidth=2, label=f'High-Perf CRNN (N={L_sym})')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.legend()
plt.title(f'FTN Detection Performance (Targeting 1e-5)')
plt.savefig('final_perf_ber.png')