"""
CNN-FK3: Structured Fixed Kernel CNN for FTN Detection
Exact implementation of Tokluoglu et al., IEEE Trans. Commun., 2025

Single file: Model + Training + BER Test

Author: Emre Cerci
Date: January 2026
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, convolve
from scipy.special import erfc
import time
import os

# ============================================================
# MODEL DEFINITION
# ============================================================

class CNN_FK3(nn.Module):
    """
    Structured Fixed Kernel CNN for FTN Detection
    
    Architecture (from paper):
    - N Fixed Kernel Layers (each extracts triplet [y_{k-i}, y_k, y_{k+i}])
    - Concatenation of all filter outputs
    - Dense layer (4 neurons, tanh)
    - Output layer (1 neuron, sigmoid)
    """
    
    def __init__(self, tau):
        super(CNN_FK3, self).__init__()
        
        self.tau = tau
        
        # Table III: N and filter allocation based on tau
        if tau == 0.9:
            self.N = 2
            self.filters_per_layer = [2, 1]
        elif tau == 0.8:
            self.N = 6
            self.filters_per_layer = [4, 2, 2, 1, 1, 1]
        elif tau == 0.7:
            self.N = 8
            self.filters_per_layer = [8, 6, 4, 2, 2, 1, 1, 1]
        else:
            raise ValueError(f"Unsupported tau={tau}. Use 0.7, 0.8, or 0.9")
        
        self.input_size = 2 * self.N + 1
        self.num_fk_layers = self.N
        self.total_filters = sum(self.filters_per_layer)
        
        # Fixed Kernel Layers (Eq. 14, 15)
        # Each layer: Linear(3 -> F_i) + Tanh
        self.fk_layers = nn.ModuleList()
        for i, num_filters in enumerate(self.filters_per_layer):
            self.fk_layers.append(nn.Linear(3, num_filters))
        
        # Dense layer (Eq. 16): total_filters -> 4, tanh
        self.dense = nn.Linear(self.total_filters, 4)
        
        # Output layer (Eq. 20): 4 -> 1, sigmoid
        self.output = nn.Linear(4, 1)
        
        # Activations
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights (Xavier)
        self._init_weights()
        
        print(f"CNN-FK3 Model created for tau={tau}")
        print(f"  N={self.N}, Input size={self.input_size}")
        print(f"  Filters per layer: {self.filters_per_layer}")
        print(f"  Total filters: {self.total_filters}")
        print(f"  Total parameters: {self.count_parameters()}")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        """
        Forward pass
        x: [batch_size, 2N+1] - received symbol window
        """
        batch_size = x.shape[0]
        center = self.N  # 0-indexed center position
        
        # Process each Fixed Kernel layer
        fk_outputs = []
        for i, fk_layer in enumerate(self.fk_layers):
            # Extract triplet [y_{k-i-1}, y_k, y_{k+i+1}] (i is 0-indexed, so use i+1)
            dist = i + 1
            idx_left = center - dist
            idx_right = center + dist
            
            # triplet: [batch, 3]
            triplet = x[:, [idx_left, center, idx_right]]
            
            # Linear + Tanh (Eq. 15)
            out = self.tanh(fk_layer(triplet))  # [batch, F_i]
            fk_outputs.append(out)
        
        # Concatenate all FK outputs
        concat = torch.cat(fk_outputs, dim=1)  # [batch, total_filters]
        
        # Dense layer (Eq. 16)
        h = self.tanh(self.dense(concat))  # [batch, 4]
        
        # Output layer (Eq. 20)
        out = self.sigmoid(self.output(h))  # [batch, 1]
        
        return out.squeeze(1)


# ============================================================
# FTN DATA GENERATION
# ============================================================

def generate_srrc_pulse(beta, span, sps):
    """Generate Square Root Raised Cosine pulse"""
    n = np.arange(-span * sps, span * sps + 1)
    t = n / sps
    
    h = np.zeros_like(t, dtype=float)
    
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(ti) == 1 / (4 * beta) and beta != 0:
            h[i] = (beta / np.sqrt(2)) * ((1 + 2/np.pi) * np.sin(np.pi / (4*beta)) + 
                                           (1 - 2/np.pi) * np.cos(np.pi / (4*beta)))
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            if den != 0:
                h[i] = num / den
            else:
                h[i] = 0
    
    # Normalize to unit energy
    h = h / np.sqrt(np.sum(h ** 2))
    
    return h


def generate_ftn_data(tau, num_samples, snr_values, beta=0.35, sps=10, span=6):
    """
    Generate FTN training/test data
    
    Returns:
        X: [num_samples, 2N+1] - input windows
        Y: [num_samples] - labels (0 or 1)
    """
    # Determine N based on tau
    if tau == 0.9:
        N = 2
    elif tau == 0.8:
        N = 6
    elif tau == 0.7:
        N = 8
    else:
        raise ValueError(f"Unsupported tau={tau}")
    
    input_size = 2 * N + 1
    step = int(round(tau * sps))
    
    # Generate SRRC pulse
    h = generate_srrc_pulse(beta, span, sps)
    delay = span * sps
    
    # Preallocate
    X = np.zeros((num_samples, input_size), dtype=np.float32)
    Y = np.zeros(num_samples, dtype=np.float32)
    
    samples_per_snr = num_samples // len(snr_values)
    batch_size = 10000  # Symbols per batch
    
    sample_idx = 0
    
    for snr_db in snr_values:
        # Noise calculation (Eb/N0)
        eb_n0_lin = 10 ** (snr_db / 10)
        n0 = 1 / eb_n0_lin
        noise_std = np.sqrt(n0 / 2)
        
        num_batches = (samples_per_snr + batch_size - 1) // batch_size
        
        for _ in range(num_batches):
            if sample_idx >= num_samples:
                break
            
            # Generate random bits
            bits = np.random.randint(0, 2, batch_size)
            symbols = 2 * bits - 1  # BPSK: 0->-1, 1->+1
            
            # Pad with -1
            symbols_padded = np.concatenate([
                -np.ones(N),
                symbols,
                -np.ones(N)
            ])
            
            # Upsample
            tx_up = np.zeros(len(symbols_padded) * step)
            tx_up[::step] = symbols_padded
            
            # Pulse shaping
            tx_sig = np.convolve(tx_up, h)
            
            # Add noise
            noise = noise_std * np.random.randn(len(tx_sig))
            rx_sig = tx_sig + noise
            
            # Matched filter
            rx_mf = np.convolve(rx_sig, h)
            
            # Extract windows
            total_delay = 2 * delay
            
            for k in range(batch_size):
                if sample_idx >= num_samples:
                    break
                
                center_idx = total_delay + (N + k) * step
                
                window = np.zeros(input_size, dtype=np.float32)
                for w in range(input_size):
                    sym_offset = w - N
                    sample_pos = center_idx + sym_offset * step
                    if 0 <= sample_pos < len(rx_mf):
                        window[w] = rx_mf[sample_pos]
                
                X[sample_idx] = window
                Y[sample_idx] = bits[k]
                sample_idx += 1
    
    return X[:sample_idx], Y[:sample_idx]


# ============================================================
# TRAINING
# ============================================================

def train_model(model, train_loader, val_loader, config, device):
    """
    Train CNN-FK3 model with Adam optimizer and LR decay
    (Table IV hyperparameters)
    """
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['initial_lr'])
    
    # LR scheduler (decay every 5 epochs)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config['decay_interval_epochs'], 
        gamma=config['decay_rate']
    )
    
    history = {'loss': [], 'val_loss': [], 'lr': []}
    
    print(f"\nStarting training...")
    print(f"Epochs: {config['num_epochs']}, Batch size: {config['batch_size']}")
    print(f"Initial LR: {config['initial_lr']}, Decay rate: {config['decay_rate']}\n")
    
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            # Limit steps per epoch
            if num_batches >= config['steps_per_epoch']:
                break
        
        avg_train_loss = train_loss / num_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for X_val, Y_val in val_loader:
                X_val = X_val.to(device)
                Y_val = Y_val.to(device)
                outputs = model(X_val)
                val_loss += criterion(outputs, Y_val).item()
                val_batches += 1
                if val_batches >= 10:  # Limit validation batches
                    break
        
        avg_val_loss = val_loss / val_batches
        
        # Update LR
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} - "
              f"Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
              f"LR: {current_lr:.6f}, Time: {epoch_time:.1f}s")
        
        history['loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)
    
    return history


# ============================================================
# BER TEST
# ============================================================

def test_ber(model, tau, snr_range, device, beta=0.35, min_errors=100, max_symbols=int(1e7)):
    """
    BER test with Monte Carlo simulation
    """
    model.eval()
    
    # FTN parameters
    if tau == 0.9:
        N = 2
    elif tau == 0.8:
        N = 6
    elif tau == 0.7:
        N = 8
    
    sps = 10
    span = 6
    step = int(round(tau * sps))
    delay = span * sps
    input_size = 2 * N + 1
    frame_size = 5000
    max_frames = max_symbols // frame_size
    
    # SRRC pulse
    h = generate_srrc_pulse(beta, span, sps)
    
    print(f"\n{'='*60}")
    print(f"CNN-FK3 BER Test (tau={tau})")
    print(f"{'='*60}")
    print(f"{'SNR (dB)':<10} | {'BER':<12} | {'Errors':<8} | {'Symbols':<12} | {'Frames':<6}")
    print(f"{'-'*10}-+-{'-'*12}-+-{'-'*8}-+-{'-'*12}-+-{'-'*6}")
    
    ber_results = []
    
    np.random.seed(42)
    
    for snr_db in snr_range:
        eb_n0_lin = 10 ** (snr_db / 10)
        n0 = 1 / eb_n0_lin
        noise_std = np.sqrt(n0 / 2)
        
        errors = 0
        symbols = 0
        frame_count = 0
        
        while errors < min_errors and frame_count < max_frames:
            frame_count += 1
            
            # Generate bits
            bits = np.random.randint(0, 2, frame_size)
            tx_symbols = 2 * bits - 1
            
            # Pad
            symbols_padded = np.concatenate([
                -np.ones(N), tx_symbols, -np.ones(N)
            ])
            
            # Transmit
            tx_up = np.zeros(len(symbols_padded) * step)
            tx_up[::step] = symbols_padded
            tx_sig = np.convolve(tx_up, h)
            
            # Add noise
            noise = noise_std * np.random.randn(len(tx_sig))
            rx_sig = tx_sig + noise
            
            # Matched filter
            rx_mf = np.convolve(rx_sig, h)
            
            # Extract windows
            total_delay = 2 * delay
            X_test = np.zeros((frame_size, input_size), dtype=np.float32)
            
            for k in range(frame_size):
                center_idx = total_delay + (N + k) * step
                for w in range(input_size):
                    sym_offset = w - N
                    sample_pos = center_idx + sym_offset * step
                    if 0 <= sample_pos < len(rx_mf):
                        X_test[k, w] = rx_mf[sample_pos]
            
            # Predict
            with torch.no_grad():
                X_tensor = torch.tensor(X_test, device=device)
                y_pred = model(X_tensor).cpu().numpy()
            
            bits_hat = (y_pred > 0.5).astype(int)
            
            errors += np.sum(bits_hat != bits)
            symbols += frame_size
        
        ber = errors / symbols if symbols > 0 else 0
        ber_results.append(ber)
        
        print(f"{snr_db:^10} | {ber:<12.2e} | {errors:<8} | {symbols:<12} | {frame_count:<6}")
        
        if ber == 0:
            print("  (No errors detected, stopping)")
            break
    
    return np.array(ber_results)


# ============================================================
# MAIN
# ============================================================

def main():
    # Configuration (Table IV)
    config = {
        'tau': 0.7,
        'n_train': 4_000_000,
        'n_test': 100_000,
        'snr_train': [7, 8, 9, 10],
        'batch_size': 1000,
        'num_epochs': 20,
        'steps_per_epoch': 4000,
        'initial_lr': 0.001,
        'decay_interval_epochs': 5,
        'decay_rate': 0.9,
        'beta': 0.35,
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tau = config['tau']
    
    # ============== DATA GENERATION ==============
    print(f"\n{'='*60}")
    print(f"Generating FTN data for tau={tau}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    X_train, Y_train = generate_ftn_data(
        tau, config['n_train'], config['snr_train'], config['beta']
    )
    X_test, Y_test = generate_ftn_data(
        tau, config['n_test'], config['snr_train'], config['beta']
    )
    print(f"Data generation completed in {time.time() - start_time:.1f}s")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # ============== MODEL ==============
    print(f"\n{'='*60}")
    print("Creating model...")
    print(f"{'='*60}")
    
    model = CNN_FK3(tau).to(device)
    
    # ============== TRAINING ==============
    print(f"\n{'='*60}")
    print("Training...")
    print(f"{'='*60}")
    
    history = train_model(model, train_loader, test_loader, config, device)
    
    # Save model
    model_path = f'cnn_fk3_tau{tau}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    # ============== BER TEST ==============
    snr_range = np.arange(0, 15, 2)
    ber_cnn = test_ber(model, tau, snr_range, device, config['beta'])
    
    # Theoretical BPSK
    eb_n0_lin = 10 ** (snr_range / 10)
    ber_bpsk = 0.5 * erfc(np.sqrt(eb_n0_lin))
    
    # ============== PLOT ==============
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training curves
    axes[0].plot(history['loss'], 'b-', linewidth=2, label='Train')
    axes[0].plot(history['val_loss'], 'r--', linewidth=2, label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Training Loss (τ={tau})')
    axes[0].legend()
    axes[0].grid(True)
    
    # BER curve
    axes[1].semilogy(snr_range[:len(ber_cnn)], ber_cnn, 'bo-', linewidth=2, markersize=8, label=f'CNN-FK3 (τ={tau})')
    axes[1].semilogy(snr_range, ber_bpsk, 'k--', linewidth=1.5, label='BPSK (ISI-free)')
    axes[1].set_xlabel('Eb/N0 (dB)')
    axes[1].set_ylabel('BER')
    axes[1].set_title(f'CNN-FK3 BER Performance (τ={tau})')
    axes[1].legend()
    axes[1].grid(True, which='both')
    axes[1].set_ylim([1e-6, 1])
    
    plt.tight_layout()
    plt.savefig(f'cnn_fk3_results_tau{tau}.png', dpi=150)
    plt.show()
    
    # ============== SUMMARY ==============
    print(f"\n{'='*60}")
    print("Results Summary @ 10 dB")
    print(f"{'='*60}")
    idx_10db = np.where(snr_range == 10)[0]
    if len(idx_10db) > 0 and idx_10db[0] < len(ber_cnn):
        print(f"BPSK Theoretical: {ber_bpsk[idx_10db[0]]:.2e}")
        print(f"CNN-FK3:          {ber_cnn[idx_10db[0]]:.2e}")
    
    # Save results
    np.savez(f'ber_results_tau{tau}.npz', 
             snr_range=snr_range[:len(ber_cnn)], 
             ber_cnn=ber_cnn, 
             ber_bpsk=ber_bpsk[:len(ber_cnn)])
    
    print(f"\nResults saved to ber_results_tau{tau}.npz")


if __name__ == '__main__':
    main()