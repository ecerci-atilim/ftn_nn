"""
FTN with Configurable Impairments - Easy-to-Use Simulation

Command-line interface for FTN simulation with toggleable impairments.

Usage:
    # Basic (no impairments)
    python ftn_sim_configurable.py

    # With PA saturation only
    python ftn_sim_configurable.py --pa

    # With PA + IQ imbalance
    python ftn_sim_configurable.py --pa --iq-tx

    # All impairments
    python ftn_sim_configurable.py --all

    # Custom configuration
    python ftn_sim_configurable.py --pa --pa-ibo 5 --iq-tx --iq-amp 1.0

Author: Emre Cerci
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import json
from pathlib import Path

from pa_models import apply_pa_model, get_pa_params_from_ibo
from impairments import ImpairmentChain


# ============================================================================
# CONFIGURATION
# ============================================================================

class SimConfig:
    """Simulation configuration"""

    def __init__(self):
        # FTN Parameters
        self.tau = 0.7
        self.beta = 0.35
        self.sps = 10
        self.span = 6

        # Simulation
        self.snr_range = np.arange(0, 15, 2)
        self.n_train = 50000
        self.n_test = 20000
        self.n_window = 8
        self.l_frac = 2

        # Neural Network
        self.epochs = 15
        self.batch_size = 256
        self.lr = 0.001

        # Impairments (default: all OFF)
        self.impairment_config = ImpairmentChain.get_default_config()

    def enable_all_impairments(self):
        """Enable all impairments with moderate settings"""
        self.impairment_config['pa_saturation']['enabled'] = True
        self.impairment_config['pa_saturation']['IBO_dB'] = 3

        self.impairment_config['iq_imbalance_tx']['enabled'] = True
        self.impairment_config['iq_imbalance_tx']['amp_imbalance_dB'] = 0.5
        self.impairment_config['iq_imbalance_tx']['phase_imbalance_deg'] = 5

        self.impairment_config['phase_noise']['enabled'] = True
        self.impairment_config['phase_noise']['psd_dBc_Hz'] = -80
        self.impairment_config['phase_noise']['fs'] = self.sps * 1e4

        self.impairment_config['dac_quantization']['enabled'] = True
        self.impairment_config['dac_quantization']['n_bits'] = 8

        self.impairment_config['adc_quantization']['enabled'] = True
        self.impairment_config['adc_quantization']['n_bits'] = 8

        self.impairment_config['cfo']['enabled'] = True
        self.impairment_config['cfo']['cfo_hz'] = 50
        self.impairment_config['cfo']['fs'] = self.sps * 1e4

        self.impairment_config['iq_imbalance_rx']['enabled'] = True
        self.impairment_config['iq_imbalance_rx']['amp_imbalance_dB'] = 0.3
        self.impairment_config['iq_imbalance_rx']['phase_imbalance_deg'] = 3

    def save_to_file(self, filename):
        """Save configuration to JSON"""
        config_dict = {
            'ftn': {
                'tau': self.tau,
                'beta': self.beta,
                'sps': self.sps,
                'span': self.span
            },
            'simulation': {
                'n_train': self.n_train,
                'n_test': self.n_test,
                'n_window': self.n_window,
                'l_frac': self.l_frac
            },
            'impairments': self.impairment_config
        }

        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Configuration saved to: {filename}")

    @classmethod
    def load_from_file(cls, filename):
        """Load configuration from JSON"""
        config = cls()

        with open(filename, 'r') as f:
            config_dict = json.load(f)

        # Load FTN params
        if 'ftn' in config_dict:
            for key, val in config_dict['ftn'].items():
                setattr(config, key, val)

        # Load simulation params
        if 'simulation' in config_dict:
            for key, val in config_dict['simulation'].items():
                setattr(config, key, val)

        # Load impairments
        if 'impairments' in config_dict:
            config.impairment_config = config_dict['impairments']

        print(f"Configuration loaded from: {filename}")
        return config


# ============================================================================
# PULSE SHAPING
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

    h = h / np.sqrt(np.sum(h ** 2))
    return h


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_ftn_data(config, n_symbols, snr_db):
    """Generate FTN data with configurable impairments"""

    # SRRC pulse
    h_srrc = generate_srrc_pulse(config.beta, config.span, config.sps)
    delay = config.span * config.sps

    # Generate bits
    bits = np.random.randint(0, 2, n_symbols)
    symbols = 2 * bits - 1  # BPSK

    # FTN upsampling
    step = int(np.round(config.tau * config.sps))
    tx_up = np.zeros(n_symbols * step, dtype=complex)
    tx_up[::step] = symbols

    # Pulse shaping
    tx_shaped = np.convolve(tx_up, h_srrc, mode='full')

    # Apply TX impairments (IQ imbalance, DAC, PA)
    impairment_chain = ImpairmentChain(config.impairment_config)
    tx_impaired = impairment_chain.apply_tx_impairments(tx_shaped)

    # AWGN Channel
    EbN0 = 10 ** (snr_db / 10)
    noise_power = 1 / (2 * EbN0)
    noise = np.sqrt(noise_power) * (np.random.randn(len(tx_impaired)) + 1j * np.random.randn(len(tx_impaired)))
    rx_noisy = tx_impaired + noise

    # Apply RX impairments (CFO, phase noise, IQ, ADC)
    rx_impaired = impairment_chain.apply_rx_impairments(rx_noisy)

    # Matched filter
    rx_mf = np.convolve(rx_impaired, h_srrc, mode='full')
    rx_mf = rx_mf / np.std(rx_mf)

    return rx_mf, bits, delay, step


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_fractional_samples(rx_mf, n_symbols, step, delay, n_window, l_frac):
    """Extract fractional samples"""
    frac_step = step // l_frac
    total_delay = 2 * delay

    X, valid_indices = [], []

    for k in range(n_symbols):
        center_idx = total_delay + k * step
        window = []
        valid = True

        for offset in range(-n_window * l_frac, (n_window + 1) * l_frac):
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
# NEURAL NETWORK
# ============================================================================

class FractionalNN(nn.Module):
    """Optimized neural network equalizer"""

    def __init__(self, input_size, hidden_sizes=[64, 32, 16]):
        super(FractionalNN, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_model(model, train_loader, epochs, lr, device, verbose=True):
    """Train neural network"""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

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
        scheduler.step(avg_loss)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")


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

def run_simulation(config, save_results=True):
    """Run FTN simulation with current configuration"""

    # Print configuration
    print("\n" + "=" * 70)
    print(" FTN SIMULATION WITH CONFIGURABLE IMPAIRMENTS")
    print("=" * 70)
    print(f"\nFTN Parameters: tau={config.tau}, beta={config.beta}, sps={config.sps}")
    print(f"Fractional sampling: L={config.l_frac} (T/{config.l_frac} spacing)")
    print(f"Training: {config.n_train} symbols, Testing: {config.n_test} symbols/SNR")

    impairment_chain = ImpairmentChain(config.impairment_config)
    impairment_chain.print_config()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Generate training data
    print("\n[1/4] Generating training data...")
    rx_mf_train, bits_train, delay, step = generate_ftn_data(config, config.n_train, 10)

    print("[2/4] Extracting features...")
    X_train, idx_train = extract_fractional_samples(
        rx_mf_train, config.n_train, step, delay, config.n_window, config.l_frac
    )
    Y_train = bits_train[idx_train].astype(np.float32)

    print(f"Training samples: {len(X_train)}, Feature dimension: {X_train.shape[1]}")

    # Create DataLoader
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # Create and train model
    print("\n[3/4] Training neural network equalizer...")
    model = FractionalNN(X_train.shape[1]).to(device)
    train_model(model, train_loader, config.epochs, config.lr, device)

    # BER Testing
    print("\n[4/4] Testing BER performance...")
    print("=" * 70)
    print(f"{'SNR (dB)':<10} | {'BER':<15}")
    print("-" * 70)

    BER_results = []

    for snr_db in config.snr_range:
        rx_mf_test, bits_test, delay, step = generate_ftn_data(config, config.n_test, snr_db)
        X_test, idx_test = extract_fractional_samples(
            rx_mf_test, config.n_test, step, delay, config.n_window, config.l_frac
        )
        Y_test = bits_test[idx_test]

        ber = test_ber(model, X_test, Y_test, device)
        BER_results.append(ber)

        print(f"{snr_db:<10} | {ber:<15.2e}")

    print("=" * 70)

    # Save results
    if save_results:
        results = {
            'config': {
                'tau': config.tau,
                'beta': config.beta,
                'l_frac': config.l_frac
            },
            'snr_db': config.snr_range.tolist(),
            'ber': BER_results,
            'impairments': config.impairment_config
        }

        results_file = Path('simulation_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    # Plot
    plot_results(config, BER_results)

    return BER_results


def plot_results(config, ber_results):
    """Plot BER results"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(config.snr_range, ber_results, 'bo-', linewidth=2, markersize=8,
                label=f'Fractional NN (L={config.l_frac})')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate', fontsize=12)
    ax.set_title(f'FTN with Impairments (τ={config.tau})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim([1e-5, 1])

    plt.tight_layout()
    plt.savefig('ber_results.png', dpi=150)
    print("BER plot saved: ber_results.png")
    plt.show()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='FTN Simulation with Configurable Impairments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # No impairments (baseline)
  python ftn_sim_configurable.py

  # PA saturation only
  python ftn_sim_configurable.py --pa --pa-ibo 3

  # PA + TX IQ imbalance
  python ftn_sim_configurable.py --pa --iq-tx --iq-tx-amp 0.5

  # All impairments
  python ftn_sim_configurable.py --all

  # Load from config file
  python ftn_sim_configurable.py --load config.json

  # Save config
  python ftn_sim_configurable.py --pa --iq-tx --save-config my_config.json
        """
    )

    # Configuration file
    parser.add_argument('--load', type=str, help='Load configuration from JSON file')
    parser.add_argument('--save-config', type=str, help='Save configuration to JSON file')

    # Quick presets
    parser.add_argument('--all', action='store_true', help='Enable all impairments')

    # PA Saturation
    parser.add_argument('--pa', action='store_true', help='Enable PA saturation')
    parser.add_argument('--pa-model', type=str, default='rapp', choices=['rapp', 'saleh', 'soft_limiter'])
    parser.add_argument('--pa-ibo', type=float, default=3, help='PA Input Back-Off (dB)')
    parser.add_argument('--pa-memory', action='store_true', help='Enable PA memory effects')

    # IQ Imbalance
    parser.add_argument('--iq-tx', action='store_true', help='Enable TX IQ imbalance')
    parser.add_argument('--iq-tx-amp', type=float, default=0.5, help='TX IQ amplitude imbalance (dB)')
    parser.add_argument('--iq-tx-phase', type=float, default=5, help='TX IQ phase imbalance (degrees)')

    parser.add_argument('--iq-rx', action='store_true', help='Enable RX IQ imbalance')
    parser.add_argument('--iq-rx-amp', type=float, default=0.3, help='RX IQ amplitude imbalance (dB)')
    parser.add_argument('--iq-rx-phase', type=float, default=3, help='RX IQ phase imbalance (degrees)')

    # Phase Noise
    parser.add_argument('--pn', action='store_true', help='Enable phase noise')
    parser.add_argument('--pn-psd', type=float, default=-80, help='Phase noise PSD (dBc/Hz)')

    # Quantization
    parser.add_argument('--dac', action='store_true', help='Enable DAC quantization')
    parser.add_argument('--dac-bits', type=int, default=8, help='DAC resolution (bits)')

    parser.add_argument('--adc', action='store_true', help='Enable ADC quantization')
    parser.add_argument('--adc-bits', type=int, default=8, help='ADC resolution (bits)')

    # CFO
    parser.add_argument('--cfo', action='store_true', help='Enable carrier frequency offset')
    parser.add_argument('--cfo-hz', type=float, default=50, help='CFO in Hz')

    # FTN parameters
    parser.add_argument('--tau', type=float, default=0.7, help='FTN compression factor')
    parser.add_argument('--l-frac', type=int, default=2, help='Fractional sampling factor')

    # Simulation
    parser.add_argument('--n-train', type=int, default=50000, help='Number of training symbols')
    parser.add_argument('--n-test', type=int, default=20000, help='Number of test symbols')
    parser.add_argument('--epochs', type=int, default=15, help='Training epochs')

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()

    # Load or create configuration
    if args.load:
        config = SimConfig.load_from_file(args.load)
    else:
        config = SimConfig()

        # Apply FTN parameters
        config.tau = args.tau
        config.l_frac = args.l_frac
        config.n_train = args.n_train
        config.n_test = args.n_test
        config.epochs = args.epochs

        # Apply impairment settings
        if args.all:
            config.enable_all_impairments()
        else:
            # PA
            if args.pa:
                config.impairment_config['pa_saturation']['enabled'] = True
                config.impairment_config['pa_saturation']['model'] = args.pa_model
                config.impairment_config['pa_saturation']['IBO_dB'] = args.pa_ibo
                config.impairment_config['pa_saturation']['memory_effects'] = args.pa_memory

            # IQ Imbalance
            if args.iq_tx:
                config.impairment_config['iq_imbalance_tx']['enabled'] = True
                config.impairment_config['iq_imbalance_tx']['amp_imbalance_dB'] = args.iq_tx_amp
                config.impairment_config['iq_imbalance_tx']['phase_imbalance_deg'] = args.iq_tx_phase

            if args.iq_rx:
                config.impairment_config['iq_imbalance_rx']['enabled'] = True
                config.impairment_config['iq_imbalance_rx']['amp_imbalance_dB'] = args.iq_rx_amp
                config.impairment_config['iq_imbalance_rx']['phase_imbalance_deg'] = args.iq_rx_phase

            # Phase Noise
            if args.pn:
                config.impairment_config['phase_noise']['enabled'] = True
                config.impairment_config['phase_noise']['psd_dBc_Hz'] = args.pn_psd
                config.impairment_config['phase_noise']['fs'] = config.sps * 1e4

            # Quantization
            if args.dac:
                config.impairment_config['dac_quantization']['enabled'] = True
                config.impairment_config['dac_quantization']['n_bits'] = args.dac_bits

            if args.adc:
                config.impairment_config['adc_quantization']['enabled'] = True
                config.impairment_config['adc_quantization']['n_bits'] = args.adc_bits

            # CFO
            if args.cfo:
                config.impairment_config['cfo']['enabled'] = True
                config.impairment_config['cfo']['cfo_hz'] = args.cfo_hz
                config.impairment_config['cfo']['fs'] = config.sps * 1e4

    # Save configuration if requested
    if args.save_config:
        config.save_to_file(args.save_config)

    # Run simulation
    run_simulation(config)


if __name__ == '__main__':
    main()
