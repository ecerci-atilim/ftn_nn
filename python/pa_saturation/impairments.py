"""
Nonlinear Impairments Module

Comprehensive set of hardware impairments for realistic FTN simulation:
1. PA Saturation (memoryless and with memory)
2. IQ Imbalance
3. Phase Noise
4. ADC/DAC Quantization
5. Carrier Frequency Offset (CFO)

Each impairment can be toggled on/off independently.

Author: Emre Cerci
Date: January 2026
"""

import numpy as np
from pa_models import apply_pa_model


class ImpairmentChain:
    """
    Configurable chain of hardware impairments

    Usage:
        config = {
            'pa_saturation': {'enabled': True, 'model': 'rapp', 'IBO_dB': 3},
            'iq_imbalance': {'enabled': True, 'amp_imb': 0.1, 'phase_imb': 5},
            'phase_noise': {'enabled': True, 'psd_dBc_Hz': -80},
            'quantization': {'enabled': True, 'n_bits': 8},
            'cfo': {'enabled': True, 'cfo_hz': 100, 'fs': 10e6}
        }
        chain = ImpairmentChain(config)
        tx_impaired = chain.apply_tx_impairments(tx_signal)
        rx_impaired = chain.apply_rx_impairments(rx_signal)
    """

    def __init__(self, config=None):
        """Initialize with configuration dictionary"""
        self.config = config if config is not None else self.get_default_config()

    @staticmethod
    def get_default_config():
        """Return default configuration with all impairments disabled"""
        return {
            'pa_saturation': {
                'enabled': False,
                'model': 'rapp',
                'IBO_dB': 3,
                'memory_effects': False,
                'memory_depth': 3
            },
            'iq_imbalance_tx': {
                'enabled': False,
                'amp_imbalance_dB': 0.5,  # Amplitude imbalance in dB
                'phase_imbalance_deg': 5   # Phase imbalance in degrees
            },
            'phase_noise': {
                'enabled': False,
                'psd_dBc_Hz': -80,  # Phase noise PSD in dBc/Hz
                'fs': 1e6            # Sampling frequency
            },
            'dac_quantization': {
                'enabled': False,
                'n_bits': 8,
                'full_scale': 1.0
            },
            'adc_quantization': {
                'enabled': False,
                'n_bits': 8,
                'full_scale': 1.0
            },
            'cfo': {
                'enabled': False,
                'cfo_hz': 100,
                'fs': 1e6
            },
            'iq_imbalance_rx': {
                'enabled': False,
                'amp_imbalance_dB': 0.3,
                'phase_imbalance_deg': 3
            }
        }

    def apply_tx_impairments(self, signal):
        """
        Apply transmitter-side impairments in order:
        1. IQ imbalance (TX)
        2. DAC quantization
        3. PA saturation
        """
        s = np.copy(signal)

        # 1. TX IQ Imbalance (before PA)
        if self.config['iq_imbalance_tx']['enabled']:
            s = self._apply_iq_imbalance(s, self.config['iq_imbalance_tx'])

        # 2. DAC Quantization
        if self.config['dac_quantization']['enabled']:
            s = self._apply_quantization(s, self.config['dac_quantization'])

        # 3. PA Saturation
        if self.config['pa_saturation']['enabled']:
            s = self._apply_pa_saturation(s)

        return s

    def apply_rx_impairments(self, signal):
        """
        Apply receiver-side impairments:
        1. CFO
        2. Phase noise
        3. RX IQ imbalance
        4. ADC quantization
        """
        s = np.copy(signal)

        # 1. Carrier Frequency Offset
        if self.config['cfo']['enabled']:
            s = self._apply_cfo(s)

        # 2. Phase Noise
        if self.config['phase_noise']['enabled']:
            s = self._apply_phase_noise(s)

        # 3. RX IQ Imbalance
        if self.config['iq_imbalance_rx']['enabled']:
            s = self._apply_iq_imbalance(s, self.config['iq_imbalance_rx'])

        # 4. ADC Quantization
        if self.config['adc_quantization']['enabled']:
            s = self._apply_quantization(s, self.config['adc_quantization'])

        return s

    def _apply_pa_saturation(self, signal):
        """Apply PA saturation (memoryless or with memory)"""
        cfg = self.config['pa_saturation']

        from pa_models import get_pa_params_from_ibo
        pa_params = get_pa_params_from_ibo(cfg['model'], cfg['IBO_dB'])

        if cfg.get('memory_effects', False):
            # Simple memory polynomial model
            return self._apply_memory_pa(signal, pa_params, cfg)
        else:
            # Memoryless PA
            return apply_pa_model(signal, cfg['model'], pa_params)

    def _apply_memory_pa(self, signal, pa_params, cfg):
        """
        PA with memory effects (simplified Volterra/Memory Polynomial)
        y[n] = PA(x[n]) + sum_k alpha_k * PA(x[n-k])
        """
        depth = cfg.get('memory_depth', 3)

        # Memoryless component
        y = apply_pa_model(signal, cfg['model'], pa_params)

        # Memory components (simplified - use delayed versions)
        memory_coeffs = np.array([0.05, 0.03, 0.01])[:depth]

        for k in range(1, depth + 1):
            if k < len(memory_coeffs):
                delayed = np.roll(signal, k)
                delayed[:k] = 0
                y += memory_coeffs[k-1] * apply_pa_model(delayed, cfg['model'], pa_params)

        return y

    def _apply_iq_imbalance(self, signal, cfg):
        """
        Apply IQ imbalance

        Model: y = (1 + α)·I + j·(1 - α)·Q·exp(jφ)
        where α is amplitude imbalance, φ is phase imbalance
        """
        amp_imb_dB = cfg['amp_imbalance_dB']
        phase_imb_deg = cfg['phase_imbalance_deg']

        # Convert to linear
        alpha = (10**(amp_imb_dB/20) - 1) / 2
        phi = np.deg2rad(phase_imb_deg)

        I = np.real(signal)
        Q = np.imag(signal)

        # Apply imbalance
        I_imb = (1 + alpha) * I
        Q_imb = (1 - alpha) * Q

        # Phase rotation
        y = I_imb + 1j * Q_imb * np.exp(1j * phi)

        return y

    def _apply_phase_noise(self, signal):
        """
        Apply phase noise (Wiener process model)

        φ[n] = φ[n-1] + Δφ[n], where Δφ[n] ~ N(0, σ²)
        σ² = 2π · f_3dB / fs
        """
        cfg = self.config['phase_noise']
        psd_dBc_Hz = cfg['psd_dBc_Hz']
        fs = cfg.get('fs', 1e6)

        # Convert PSD to variance
        # For simplicity, use approximate relationship
        # f_3dB ≈ 10^(PSD/10) (very simplified!)
        f_3dB = 10 ** (psd_dBc_Hz / 10)
        sigma_phi = np.sqrt(2 * np.pi * f_3dB / fs)

        # Generate phase noise (Wiener process)
        phase_noise = np.cumsum(np.random.randn(len(signal)) * sigma_phi)

        # Apply to signal
        return signal * np.exp(1j * phase_noise)

    def _apply_quantization(self, signal, cfg):
        """
        Apply ADC/DAC quantization

        Uniform quantization with n_bits resolution
        """
        n_bits = cfg['n_bits']
        full_scale = cfg['full_scale']

        # Quantization levels
        n_levels = 2 ** n_bits
        step_size = 2 * full_scale / n_levels

        # Quantize I and Q separately
        I = np.real(signal)
        Q = np.imag(signal)

        I_quant = np.round(I / step_size) * step_size
        Q_quant = np.round(Q / step_size) * step_size

        # Clip to range
        I_quant = np.clip(I_quant, -full_scale, full_scale)
        Q_quant = np.clip(Q_quant, -full_scale, full_scale)

        return I_quant + 1j * Q_quant

    def _apply_cfo(self, signal):
        """
        Apply Carrier Frequency Offset

        y[n] = x[n] · exp(j·2π·Δf·n/fs)
        """
        cfg = self.config['cfo']
        cfo_hz = cfg['cfo_hz']
        fs = cfg.get('fs', 1e6)

        n = np.arange(len(signal))
        phase_shift = np.exp(1j * 2 * np.pi * cfo_hz * n / fs)

        return signal * phase_shift

    def print_config(self):
        """Print current configuration"""
        print("=" * 60)
        print("Impairment Configuration")
        print("=" * 60)

        print("\n[Transmitter Impairments]")
        if self.config['iq_imbalance_tx']['enabled']:
            cfg = self.config['iq_imbalance_tx']
            print(f"  ✓ TX IQ Imbalance: amp={cfg['amp_imbalance_dB']}dB, "
                  f"phase={cfg['phase_imbalance_deg']}°")
        else:
            print("  ✗ TX IQ Imbalance: OFF")

        if self.config['dac_quantization']['enabled']:
            cfg = self.config['dac_quantization']
            print(f"  ✓ DAC Quantization: {cfg['n_bits']} bits")
        else:
            print("  ✗ DAC Quantization: OFF")

        if self.config['pa_saturation']['enabled']:
            cfg = self.config['pa_saturation']
            mem_str = " (with memory)" if cfg.get('memory_effects', False) else ""
            print(f"  ✓ PA Saturation: {cfg['model']}, IBO={cfg['IBO_dB']}dB{mem_str}")
        else:
            print("  ✗ PA Saturation: OFF")

        print("\n[Receiver Impairments]")
        if self.config['cfo']['enabled']:
            cfg = self.config['cfo']
            print(f"  ✓ CFO: {cfg['cfo_hz']} Hz")
        else:
            print("  ✗ CFO: OFF")

        if self.config['phase_noise']['enabled']:
            cfg = self.config['phase_noise']
            print(f"  ✓ Phase Noise: {cfg['psd_dBc_Hz']} dBc/Hz")
        else:
            print("  ✗ Phase Noise: OFF")

        if self.config['iq_imbalance_rx']['enabled']:
            cfg = self.config['iq_imbalance_rx']
            print(f"  ✓ RX IQ Imbalance: amp={cfg['amp_imbalance_dB']}dB, "
                  f"phase={cfg['phase_imbalance_deg']}°")
        else:
            print("  ✗ RX IQ Imbalance: OFF")

        if self.config['adc_quantization']['enabled']:
            cfg = self.config['adc_quantization']
            print(f"  ✓ ADC Quantization: {cfg['n_bits']} bits")
        else:
            print("  ✗ ADC Quantization: OFF")

        print("=" * 60)


# Convenience functions for quick impairment application

def apply_iq_imbalance(signal, amp_imb_dB=0.5, phase_imb_deg=5):
    """Standalone IQ imbalance function"""
    config = ImpairmentChain.get_default_config()
    config['iq_imbalance_tx']['enabled'] = True
    config['iq_imbalance_tx']['amp_imbalance_dB'] = amp_imb_dB
    config['iq_imbalance_tx']['phase_imbalance_deg'] = phase_imb_deg
    chain = ImpairmentChain(config)
    return chain._apply_iq_imbalance(signal, config['iq_imbalance_tx'])


def apply_phase_noise(signal, psd_dBc_Hz=-80, fs=1e6):
    """Standalone phase noise function"""
    config = ImpairmentChain.get_default_config()
    config['phase_noise']['enabled'] = True
    config['phase_noise']['psd_dBc_Hz'] = psd_dBc_Hz
    config['phase_noise']['fs'] = fs
    chain = ImpairmentChain(config)
    return chain._apply_phase_noise(signal)


def apply_quantization(signal, n_bits=8, full_scale=1.0):
    """Standalone quantization function"""
    config = ImpairmentChain.get_default_config()
    config['adc_quantization']['enabled'] = True
    config['adc_quantization']['n_bits'] = n_bits
    config['adc_quantization']['full_scale'] = full_scale
    chain = ImpairmentChain(config)
    return chain._apply_quantization(signal, config['adc_quantization'])


def apply_cfo(signal, cfo_hz=100, fs=1e6):
    """Standalone CFO function"""
    config = ImpairmentChain.get_default_config()
    config['cfo']['enabled'] = True
    config['cfo']['cfo_hz'] = cfo_hz
    config['cfo']['fs'] = fs
    chain = ImpairmentChain(config)
    return chain._apply_cfo(signal)


if __name__ == '__main__':
    """Test impairments"""
    import matplotlib.pyplot as plt

    # Generate test signal
    n_samples = 1000
    signal = np.exp(1j * 2 * np.pi * 0.1 * np.arange(n_samples))
    signal += 0.1 * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))

    # Test individual impairments
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].plot(np.real(signal[:100]), np.imag(signal[:100]), '.-')
    axes[0, 0].set_title('Original Signal')
    axes[0, 0].grid(True)
    axes[0, 0].axis('equal')

    # IQ Imbalance
    sig_iq = apply_iq_imbalance(signal, amp_imb_dB=1.0, phase_imb_deg=10)
    axes[0, 1].plot(np.real(sig_iq[:100]), np.imag(sig_iq[:100]), '.-')
    axes[0, 1].set_title('IQ Imbalance')
    axes[0, 1].grid(True)
    axes[0, 1].axis('equal')

    # Phase Noise
    sig_pn = apply_phase_noise(signal, psd_dBc_Hz=-70)
    axes[0, 2].plot(np.real(sig_pn[:100]), np.imag(sig_pn[:100]), '.-')
    axes[0, 2].set_title('Phase Noise')
    axes[0, 2].grid(True)
    axes[0, 2].axis('equal')

    # Quantization
    sig_quant = apply_quantization(signal, n_bits=4)
    axes[1, 0].plot(np.real(sig_quant[:100]), np.imag(sig_quant[:100]), '.-')
    axes[1, 0].set_title('Quantization (4-bit)')
    axes[1, 0].grid(True)
    axes[1, 0].axis('equal')

    # CFO
    sig_cfo = apply_cfo(signal, cfo_hz=5000, fs=100e3)
    axes[1, 1].plot(np.real(sig_cfo[:100]), np.imag(sig_cfo[:100]), '.-')
    axes[1, 1].set_title('CFO (5 kHz)')
    axes[1, 1].grid(True)
    axes[1, 1].axis('equal')

    # All combined
    config = ImpairmentChain.get_default_config()
    config['iq_imbalance_tx']['enabled'] = True
    config['pa_saturation']['enabled'] = True
    config['pa_saturation']['IBO_dB'] = 3
    config['phase_noise']['enabled'] = True
    config['quantization']['enabled'] = True

    chain = ImpairmentChain(config)
    chain.print_config()

    sig_combined = chain.apply_tx_impairments(signal)
    sig_combined = chain.apply_rx_impairments(sig_combined)

    axes[1, 2].plot(np.real(sig_combined[:100]), np.imag(sig_combined[:100]), '.-')
    axes[1, 2].set_title('All Impairments')
    axes[1, 2].grid(True)
    axes[1, 2].axis('equal')

    plt.tight_layout()
    plt.savefig('impairments_test.png', dpi=150)
    print("\nTest plot saved: impairments_test.png")
    plt.show()
