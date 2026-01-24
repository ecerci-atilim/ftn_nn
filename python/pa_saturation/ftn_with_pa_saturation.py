"""
FTN with PA Saturation - Python Implementation

Demonstrates the advantage of Fractionally Spaced Equalization (FSE) over
Symbol-Rate Sampling in the presence of PA nonlinearity.

Key Points:
    1. PA saturation creates nonlinearity BEFORE the channel
    2. Channel remains linear (AWGN)
    3. Fractional sampling at receiver captures nonlinear distortion better
    4. Neural network equalizer can exploit fractional samples

System Model:
    Bits -> BPSK -> FTN (SRRC, tau<1) -> PA Saturation -> AWGN ->
    -> Matched Filter -> Sampling (Symbol-rate vs Fractional) -> Detection

Author: Emre Cerci
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pa_models import apply_pa_model, get_pa_params_from_ibo

# ============================================================================
# CONFIGURATION
# ============================================================================

# FTN Parameters
TAU = 0.7               # FTN compression factor
BETA = 0.35             # SRRC roll-off
SPS = 10                # Samples per symbol
SPAN = 6                # Pulse span in symbols

# PA Parameters
PA_MODEL = 'rapp'       # Options: 'rapp', 'saleh', 'soft_limiter'
IBO_DB = 3              # Input Back-Off in dB

# Simulation Parameters
SNR_DB_RANGE = np.arange(0, 15, 2)  # SNR range
N_TRAIN = 100000        # Training symbols
N_TEST = 30000          # Test symbols per SNR point
N_WINDOW = 8            # Window size (2*N + 1 samples)

# Fractional Sampling
L_FRAC = 2              # Fractional factor (T/L spacing)

# Neural Network
EPOCHS = 15
BATCH_SIZE = 256
LEARNING_RATE = 0.001

print("=" * 60)
print("FTN with PA Saturation - Python Implementation")
print("=" * 60)
print(f"tau={TAU}, beta={BETA}, sps={SPS}")
print(f"PA Model: {PA_MODEL}, IBO={IBO_DB} dB")
print(f"Fractional factor: L={L_FRAC} (T/{L_FRAC} spacing)")
print("=" * 60)
print()

# ============================================================================
# SRRC PULSE GENERATION
# ============================================================================

def generate_srrc_pulse(beta, span, sps):
    """Generate Square Root Raised Cosine pulse"""
    t_vec = np.arange(-span * sps, span * sps + 1)
    t_norm = t_vec / sps

    h = np.zeros_like(t_norm, dtype=float)

    for i, t in enumerate(t_norm):
        if t == 0:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(t) == 1 / (4 * beta) and beta != 0:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi / (4*beta)) +
                (1 - 2/np.pi) * np.cos(np.pi / (4*beta))
            )
        else:
            num = np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))
            den = np.pi * t * (1 - (4 * beta * t) ** 2)
            if den != 0:
                h[i] = num / den

    # Normalize to unit energy
    h = h / np.sqrt(np.sum(h ** 2))
    return h


h_srrc = generate_srrc_pulse(BETA, SPAN, SPS)
delay = SPAN * SPS

# ============================================================================
# PA CONFIGURATION
# ============================================================================

pa_params = get_pa_params_from_ibo(PA_MODEL, IBO_DB)

# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_ftn_with_pa(n_symbols, snr_db, tau, sps, h_srrc, delay, pa_model, pa_params):
    """
    Generate FTN data with PA saturation

    Returns:
        rx_mf   : Matched filter output
        bits    : Transmitted bits
        symbols : Transmitted symbols
    """
    # Generate random bits
    bits = np.random.randint(0, 2, n_symbols)
    symbols = 2 * bits - 1  # BPSK

    # FTN: Compress symbol rate
    step = int(np.round(tau * sps))

    # Upsample
    tx_up = np.zeros(n_symbols * step, dtype=complex)
    tx_up[::step] = symbols

    # Pulse shaping (SRRC)
    tx_shaped = np.convolve(tx_up, h_srrc, mode='full')

    # ===== PA SATURATION (NONLINEARITY) =====
    tx_sig = apply_pa_model(tx_shaped, pa_model, pa_params)

    # AWGN Channel (LINEAR)
    EbN0 = 10 ** (snr_db / 10)
    noise_power = 1 / (2 * EbN0)  # For BPSK
    noise = np.sqrt(noise_power) * (np.random.randn(len(tx_sig)) + 1j * np.random.randn(len(tx_sig)))
    rx_noisy = tx_sig + noise

    # Matched Filter
    rx_mf = np.convolve(rx_noisy, h_srrc, mode='full')
    rx_mf = rx_mf / np.std(rx_mf)  # Normalize

    return rx_mf, bits, symbols


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_symbol_rate_samples(rx_mf, n_symbols, tau, sps, delay, N_window):
    """Extract symbol-rate samples (T-spaced)"""
    step = int(np.round(tau * sps))
    total_delay = 2 * delay

    X = []
    valid_indices = []

    for k in range(n_symbols):
        center_idx = total_delay + k * step

        # Extract window around symbol time
        window = []
        valid = True

        for offset in range(-N_window, N_window + 1):
            idx = center_idx + offset * step
            if 0 <= idx < len(rx_mf):
                window.append(np.real(rx_mf[idx]))
            else:
                valid = False
                break

        if valid:
            X.append(window)
            valid_indices.append(k)

    return np.array(X, dtype=np.float32), np.array(valid_indices)


def extract_fractional_samples(rx_mf, n_symbols, tau, sps, delay, N_window, L_frac):
    """Extract fractional samples (T/L-spaced)"""
    step = int(np.round(tau * sps))
    frac_step = step // L_frac  # Fractional spacing
    total_delay = 2 * delay

    X = []
    valid_indices = []

    for k in range(n_symbols):
        center_idx = total_delay + k * step

        # Extract fractional window
        window = []
        valid = True

        for offset in range(-N_window * L_frac, (N_window + 1) * L_frac):
            idx = center_idx + offset * frac_step
            if 0 <= idx < len(rx_mf):
                window.append(np.real(rx_mf[idx]))
            else:
                valid = False
                break

        if valid:
            X.append(window)
            valid_indices.append(k)

    return np.array(X, dtype=np.float32), np.array(valid_indices)


# ============================================================================
# NEURAL NETWORK EQUALIZERS
# ============================================================================

class SymbolRateNN(nn.Module):
    """Neural network equalizer for symbol-rate samples"""

    def __init__(self, input_size):
        super(SymbolRateNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class FractionalNN(nn.Module):
    """Neural network equalizer for fractional samples"""

    def __init__(self, input_size):
        super(FractionalNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, epochs, lr, device):
    """Train neural network equalizer"""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")


# ============================================================================
# BER TESTING
# ============================================================================

def test_ber(model, X_test, Y_test, device):
    """Calculate BER"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, device=device)
        Y_pred = model(X_tensor).cpu().numpy()

    bits_hat = (Y_pred > 0.5).astype(int)
    ber = np.mean(bits_hat != Y_test)
    return ber


# ============================================================================
# MAIN SIMULATION
# ============================================================================

# Generate training data
print("Generating training data...")
rx_mf_train, bits_train, _ = generate_ftn_with_pa(
    N_TRAIN, 10, TAU, SPS, h_srrc, delay, PA_MODEL, pa_params
)

# Extract features
print("Extracting features...")
X_sym_train, idx_sym = extract_symbol_rate_samples(rx_mf_train, N_TRAIN, TAU, SPS, delay, N_WINDOW)
Y_sym_train = bits_train[idx_sym].astype(np.float32)

X_frac_train, idx_frac = extract_fractional_samples(rx_mf_train, N_TRAIN, TAU, SPS, delay, N_WINDOW, L_FRAC)
Y_frac_train = bits_train[idx_frac].astype(np.float32)

print(f"Symbol-rate training samples: {len(X_sym_train)}, shape: {X_sym_train.shape}")
print(f"Fractional training samples: {len(X_frac_train)}, shape: {X_frac_train.shape}")

# Create DataLoaders
train_dataset_sym = TensorDataset(
    torch.tensor(X_sym_train, dtype=torch.float32),
    torch.tensor(Y_sym_train, dtype=torch.float32)
)
train_loader_sym = DataLoader(train_dataset_sym, batch_size=BATCH_SIZE, shuffle=True)

train_dataset_frac = TensorDataset(
    torch.tensor(X_frac_train, dtype=torch.float32),
    torch.tensor(Y_frac_train, dtype=torch.float32)
)
train_loader_frac = DataLoader(train_dataset_frac, batch_size=BATCH_SIZE, shuffle=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Create models
model_sym = SymbolRateNN(X_sym_train.shape[1]).to(device)
model_frac = FractionalNN(X_frac_train.shape[1]).to(device)

# Train models
print("Training Symbol-Rate NN...")
train_model(model_sym, train_loader_sym, EPOCHS, LEARNING_RATE, device)

print("\nTraining Fractional NN...")
train_model(model_frac, train_loader_frac, EPOCHS, LEARNING_RATE, device)

# BER Testing
print("\n" + "=" * 60)
print("BER Testing")
print("=" * 60)
print(f"{'SNR(dB)':<8} | {'Symbol-Rate BER':<15} | {'Fractional BER':<15} | {'Gain(dB)':<10}")
print("-" * 60)

BER_sym_rate = []
BER_frac = []

for snr_db in SNR_DB_RANGE:
    # Generate test data
    rx_mf_test, bits_test, _ = generate_ftn_with_pa(
        N_TEST, snr_db, TAU, SPS, h_srrc, delay, PA_MODEL, pa_params
    )

    # Symbol-rate testing
    X_sym_test, idx_sym = extract_symbol_rate_samples(rx_mf_test, N_TEST, TAU, SPS, delay, N_WINDOW)
    Y_sym_test = bits_test[idx_sym]
    ber_sym = test_ber(model_sym, X_sym_test, Y_sym_test, device)
    BER_sym_rate.append(ber_sym)

    # Fractional testing
    X_frac_test, idx_frac = extract_fractional_samples(rx_mf_test, N_TEST, TAU, SPS, delay, N_WINDOW, L_FRAC)
    Y_frac_test = bits_test[idx_frac]
    ber_frac = test_ber(model_frac, X_frac_test, Y_frac_test, device)
    BER_frac.append(ber_frac)

    # Calculate gain
    if ber_frac > 0 and ber_sym > 0:
        gain_db = 10 * np.log10(ber_sym / ber_frac)
    else:
        gain_db = np.nan

    print(f"{snr_db:<8} | {ber_sym:<15.2e} | {ber_frac:<15.2e} | {gain_db:<10.2f}")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: PA Characteristic
r_in = np.linspace(0, 2, 1000)
x_test = r_in + 0j
y_test = apply_pa_model(x_test, PA_MODEL, pa_params)

axes[0, 0].plot(r_in, np.abs(y_test), 'b-', linewidth=2, label='PA Output')
axes[0, 0].plot(r_in, r_in, 'r--', linewidth=1.5, label='Linear')
axes[0, 0].grid(True)
axes[0, 0].set_xlabel('Input Amplitude')
axes[0, 0].set_ylabel('Output Amplitude')
axes[0, 0].set_title(f'PA Characteristic ({PA_MODEL.upper()}, IBO={IBO_DB}dB)')
axes[0, 0].legend()

# Subplot 2: Constellation
tx_demo, _, _, _ = generate_ftn_with_pa(1000, 10, TAU, SPS, h_srrc, delay, PA_MODEL, pa_params)
axes[0, 1].plot(np.real(tx_demo), np.imag(tx_demo), 'b.', markersize=2)
axes[0, 1].grid(True)
axes[0, 1].axis('equal')
axes[0, 1].set_xlabel('In-Phase')
axes[0, 1].set_ylabel('Quadrature')
axes[0, 1].set_title('Transmitted Signal Constellation (After PA)')

# Subplot 3: BER Curves
axes[1, 0].semilogy(SNR_DB_RANGE, BER_sym_rate, 'rs-', linewidth=2, markersize=8, label='Symbol-Rate NN')
axes[1, 0].semilogy(SNR_DB_RANGE, BER_frac, 'bo-', linewidth=2, markersize=8, label=f'Fractional NN (L={L_FRAC})')
axes[1, 0].grid(True)
axes[1, 0].set_xlabel('SNR (dB)')
axes[1, 0].set_ylabel('Bit Error Rate')
axes[1, 0].set_title(f'BER Performance (τ={TAU}, PA={PA_MODEL.upper()})')
axes[1, 0].legend()
axes[1, 0].set_ylim([1e-5, 1])

# Subplot 4: Performance Gain
valid_idx = (np.array(BER_sym_rate) > 0) & (np.array(BER_frac) > 0)
gain_db_arr = 10 * np.log10(np.array(BER_sym_rate)[valid_idx] / np.array(BER_frac)[valid_idx])
axes[1, 1].plot(SNR_DB_RANGE[valid_idx], gain_db_arr, 'g^-', linewidth=2, markersize=8)
axes[1, 1].grid(True)
axes[1, 1].set_xlabel('SNR (dB)')
axes[1, 1].set_ylabel('Performance Gain (dB)')
axes[1, 1].set_title('Fractional Sampling Advantage')

plt.suptitle('FTN with PA Saturation: Fractional vs Symbol-Rate Sampling', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('python_ftn_pa_saturation_results.png', dpi=150)

print("\n" + "=" * 60)
print("Simulation Complete!")
print("Figure saved: python_ftn_pa_saturation_results.png")
print("=" * 60)

plt.show()
