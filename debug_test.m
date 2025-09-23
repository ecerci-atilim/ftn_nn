% Quick debugging version to identify the issue
clear, clc, close all

%% Debug Test: Perfect Channel First
fprintf('=== DEBUG TEST: Perfect Channel (No Noise, τ=1.0) ===\n');

% Test parameters
M = 4; mod_type = 'psk'; phase_offset = pi/4;
k = log2(M);
constellation = custom_modulator(M, mod_type, phase_offset);

% Generate test bits and symbols
test_bits = [0 1 0 1]'; % Known pattern
test_symbol_idx = bi2de([0 1; 0 1], 'left-msb') + 1; % Should be indices [1,2] for 4-PSK
expected_symbols = constellation(test_symbol_idx);

fprintf('Test constellation:\n');
for i = 1:M
    fprintf('  Symbol %d: %.3f + j%.3f\n', i-1, real(constellation(i)), imag(constellation(i)));
end

fprintf('\nBit-to-symbol mapping test:\n');
fprintf('Bits: [%d %d; %d %d]\n', test_bits(1), test_bits(2), test_bits(3), test_bits(4));
fprintf('Expected indices: [%d, %d]\n', test_symbol_idx(1), test_symbol_idx(2));
fprintf('Expected symbols: [%.3f+j%.3f, %.3f+j%.3f]\n', ...
    real(expected_symbols(1)), imag(expected_symbols(1)), ...
    real(expected_symbols(2)), imag(expected_symbols(2)));

% Test the mapping function
mapped_symbols = map_bits_to_symbols(test_bits, k, constellation);
fprintf('Function output: [%.3f+j%.3f, %.3f+j%.3f]\n', ...
    real(mapped_symbols(1)), imag(mapped_symbols(1)), ...
    real(mapped_symbols(2)), imag(mapped_symbols(2)));

% Test reverse mapping (demodulation)
for i = 1:length(mapped_symbols)
    [~, demod_idx] = min(abs(mapped_symbols(i) - constellation));
    demod_bits = de2bi(demod_idx-1, k, 'left-msb')';
    original_bits = test_bits((i-1)*k+1:i*k);
    
    if isequal(original_bits, demod_bits)
        result_str = 'PASS';
    else
        result_str = 'FAIL';
    end
    
    fprintf('Symbol %d: Original bits [%d %d] -> Demod bits [%d %d] -> %s\n', ...
        i, original_bits(1), original_bits(2), demod_bits(1), demod_bits(2), result_str);
end

%% Test with τ=1.0 (no ISI)
fprintf('\n=== TEST: τ=1.0 (No ISI) ===\n');
tau_test = 1.0;
[err_classical, total_bits] = test_classical_detector(mod_type, M, tau_test, phase_offset, 20);
ber_no_isi = err_classical / total_bits;
fprintf('Classical BER with τ=1.0, SNR=20dB: %.6f (should be ~1e-5)\n', ber_no_isi);

%% Test with τ=0.7 (with ISI)  
fprintf('\n=== TEST: τ=0.7 (With ISI) ===\n');
tau_test = 0.7;
[err_classical, total_bits] = test_classical_detector(mod_type, M, tau_test, phase_offset, 20);
ber_with_isi = err_classical / total_bits;
fprintf('Classical BER with τ=0.7, SNR=20dB: %.6f\n', ber_with_isi);

if ber_no_isi < 1e-3
    fprintf('✅ Basic detection works!\n');
    if ber_with_isi > 0.01
        fprintf('⚠️  ISI is causing major performance degradation\n');
        fprintf('💡 Suggestion: Train and test with τ=1.0 first, then gradually reduce\n');
    end
else
    fprintf('❌ Basic detection is broken - check bit mapping functions!\n');
end

%% Helper Functions
function [errors, total_bits] = test_classical_detector(mod_type, M, tau, phase, SNR_dB)
    k = log2(M);
    constellation = custom_modulator(M, mod_type, phase);
    
    % Simple test
    N_test = 1000;
    bits = randi([0 1], N_test * k, 1);
    symbols = map_bits_to_symbols(bits, k, constellation);
    
    % Channel simulation
    sps = 10; beta = 0.3; span = 6;
    h = rcosdesign(beta, span, sps, 'sqrt');
    tx_up = upsample(symbols, round(sps*tau));
    txSignal = conv(tx_up, h);
    pwr = mean(abs(txSignal).^2); txSignal = txSignal / sqrt(pwr);
    
    % Add noise
    snr_lin = 10^(SNR_dB/10); noise_var = 1 / (2*snr_lin);
    noise = sqrt(noise_var/2) * (randn(size(txSignal)) + 1j*randn(size(txSignal)));
    rxSignal = txSignal + noise;
    rxMF = conv(rxSignal, h);
    delay = finddelay(tx_up, rxMF);
    
    % Classical detection
    errors = 0;
    for i = 1:N_test
        loc = round((i-1)*sps*tau) + 1 + delay;
        if loc > 0 && loc <= length(rxMF)
            rx_symbol = rxMF(loc);
            [~, min_idx] = min(abs(rx_symbol - constellation));
            demod_bits = de2bi(min_idx-1, k, 'left-msb')';
            true_bits = bits((i-1)*k+1:i*k);
            errors = errors + sum(demod_bits ~= true_bits);
        end
    end
    total_bits = N_test * k;
end

function constellation = custom_modulator(M, type, phase)
    if strcmpi(type, 'psk'), p=0:M-1; constellation=exp(1j*(2*pi*p/M + phase));
    elseif strcmpi(type, 'qam'), k=log2(M); if mod(k,2)~=0, error('QAM M must be a square number.'); end; n=sqrt(M); vals=-(n-1):2:(n-1); [X,Y]=meshgrid(vals,vals); constellation=X+1j*Y; constellation = constellation * exp(1j*phase);
    else error('Unknown modulation type.'); end
    constellation = constellation(:);
    constellation = constellation / sqrt(mean(abs(constellation).^2));
end

function symbols = map_bits_to_symbols(bits, k, constellation)
    num_bits = length(bits);
    num_symbols = floor(num_bits/k);
    if num_symbols == 0, symbols = []; return; end
    bit_matrix = reshape(bits(1:num_symbols*k), k, num_symbols)';
    indices = bi2de(bit_matrix, 'left-msb') + 1;
    symbols = constellation(indices);
end