"""
PA_MODELS - Power Amplifier Nonlinearity Models

Implements various PA saturation models for communication systems

Author: Emre Cerci
Date: January 2026
"""

import numpy as np


def rapp_model(x_in, G=1.0, Asat=1.0, p=2.0):
    """
    Rapp Model (Solid State Power Amplifier)

    y(r) = G * r / (1 + (r/Asat)^(2p))^(1/(2p))

    Parameters:
        x_in  : Input signal (complex baseband)
        G     : Small signal gain (default: 1.0)
        Asat  : Saturation amplitude (default: 1.0)
        p     : Smoothness factor (default: 2.0, typical SSPA)

    Returns:
        y_out : Output signal after PA nonlinearity

    Reference:
        Rapp, C. (1991). "Effects of HPA-Nonlinearity on a 4-DPSK/OFDM-Signal
        for a Digital Sound Broadcasting System"
    """
    r = np.abs(x_in)
    phi = np.angle(x_in)

    # Rapp AM/AM conversion
    r_out = G * r / np.power(1 + np.power(r / Asat, 2 * p), 1 / (2 * p))

    # No AM/PM (phase distortion) in basic Rapp model
    y_out = r_out * np.exp(1j * phi)

    return y_out


def saleh_model(x_in, alpha_a=2.0, beta_a=1.0, alpha_p=np.pi/3, beta_p=1.0):
    """
    Saleh Model (Traveling Wave Tube Amplifier)

    AM/AM: A(r) = alpha_a * r / (1 + beta_a * r^2)
    AM/PM: Phi(r) = alpha_p * r^2 / (1 + beta_p * r^2)

    Parameters:
        x_in    : Input signal (complex baseband)
        alpha_a : AM/AM parameter (default: 2.0)
        beta_a  : AM/AM parameter (default: 1.0)
        alpha_p : AM/PM parameter (default: π/3)
        beta_p  : AM/PM parameter (default: 1.0)

    Returns:
        y_out : Output signal after PA nonlinearity

    Reference:
        Saleh, A. A. (1981). "Frequency-independent and frequency-dependent
        nonlinear models of TWT amplifiers", IEEE Trans. Commun.
    """
    r = np.abs(x_in)
    phi = np.angle(x_in)

    # Saleh AM/AM conversion
    r_out = alpha_a * r / (1 + beta_a * r**2)

    # Saleh AM/PM conversion (phase distortion)
    phi_out = phi + alpha_p * r**2 / (1 + beta_p * r**2)

    y_out = r_out * np.exp(1j * phi_out)

    return y_out


def soft_limiter(x_in, A_lin=0.7, A_sat=1.0, compress=0.1):
    """
    Soft Limiter (Simple Clipping with Continuous Transition)

    y(r) = r                                        if r <= A_lin
         = A_lin + (r - A_lin) * compress           if A_lin < r < A_sat
         = A_lin + (A_sat - A_lin) * compress       if r >= A_sat  (continuous saturation)

    Parameters:
        x_in     : Input signal (complex baseband)
        A_lin    : Linear region threshold (default: 0.7)
        A_sat    : Saturation level / input threshold for hard saturation (default: 1.0)
        compress : Compression factor in transition (default: 0.1)

    Returns:
        y_out : Output signal after PA nonlinearity

    Note:
        The saturation output level is computed to ensure continuity at r = A_sat:
        A_out_max = A_lin + (A_sat - A_lin) * compress
    """
    r = np.abs(x_in)
    phi = np.angle(x_in)

    r_out = np.zeros_like(r)

    # Compute the continuous saturation level (ensures no discontinuity)
    A_out_sat = A_lin + (A_sat - A_lin) * compress

    # Linear region
    linear_idx = r <= A_lin
    r_out[linear_idx] = r[linear_idx]

    # Transition region (soft compression)
    transition_idx = (r > A_lin) & (r < A_sat)
    r_out[transition_idx] = A_lin + (r[transition_idx] - A_lin) * compress

    # Saturation region (continuous with transition region)
    sat_idx = r >= A_sat
    r_out[sat_idx] = A_out_sat

    y_out = r_out * np.exp(1j * phi)

    return y_out


def apply_pa_model(x_in, model_type='rapp', params=None):
    """
    Apply PA model to input signal

    Parameters:
        x_in       : Input signal (complex baseband)
        model_type : PA model type ('rapp', 'saleh', 'soft_limiter')
        params     : Dictionary of model parameters (optional)

    Returns:
        y_out : Output signal after PA nonlinearity
    """
    if params is None:
        params = {}

    model_type = model_type.lower()

    if model_type == 'rapp':
        G = params.get('G', 1.0)
        Asat = params.get('Asat', 1.0)
        p = params.get('p', 2.0)
        return rapp_model(x_in, G, Asat, p)

    elif model_type == 'saleh':
        alpha_a = params.get('alpha_a', 2.0)
        beta_a = params.get('beta_a', 1.0)
        alpha_p = params.get('alpha_p', np.pi/3)
        beta_p = params.get('beta_p', 1.0)
        return saleh_model(x_in, alpha_a, beta_a, alpha_p, beta_p)

    elif model_type == 'soft_limiter':
        A_lin = params.get('A_lin', 0.7)
        A_sat = params.get('A_sat', 1.0)
        compress = params.get('compress', 0.1)
        return soft_limiter(x_in, A_lin, A_sat, compress)

    else:
        raise ValueError(f"Unknown PA model type: {model_type}. "
                        "Use 'rapp', 'saleh', or 'soft_limiter'.")


def get_pa_params_from_ibo(model_type='rapp', IBO_dB=3):
    """
    Get PA parameters based on Input Back-Off (IBO)

    Parameters:
        model_type : PA model type
        IBO_dB     : Input Back-Off in dB

    Returns:
        params : Dictionary of PA parameters
    """
    IBO_lin = 10 ** (IBO_dB / 10)

    if model_type.lower() == 'rapp':
        return {
            'G': 1.0,
            'Asat': np.sqrt(IBO_lin),
            'p': 2.0
        }

    elif model_type.lower() == 'saleh':
        return {
            'alpha_a': 2.0,
            'beta_a': 1.0 / IBO_lin,
            'alpha_p': np.pi / 3,
            'beta_p': 1.0 / IBO_lin
        }

    elif model_type.lower() == 'soft_limiter':
        return {
            'A_lin': np.sqrt(IBO_lin) * 0.7,
            'A_sat': np.sqrt(IBO_lin),
            'compress': 0.1
        }

    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    """Test PA models"""
    import matplotlib.pyplot as plt

    # Test signal
    r_in = np.linspace(0, 2, 1000)
    x_test = r_in + 0j  # Real signal for testing

    # Test different PA models
    IBO_dB = 3

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, model in enumerate(['rapp', 'saleh', 'soft_limiter']):
        params = get_pa_params_from_ibo(model, IBO_dB)
        y_out = apply_pa_model(x_test, model, params)

        axes[idx].plot(r_in, np.abs(y_out), 'b-', linewidth=2, label=f'{model.upper()}')
        axes[idx].plot(r_in, r_in, 'r--', linewidth=1.5, label='Linear')
        axes[idx].grid(True)
        axes[idx].set_xlabel('Input Amplitude')
        axes[idx].set_ylabel('Output Amplitude')
        axes[idx].set_title(f'{model.upper()} Model (IBO={IBO_dB}dB)')
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig('pa_models_comparison.png', dpi=150)
    print("PA models test plot saved: pa_models_comparison.png")
    plt.show()
