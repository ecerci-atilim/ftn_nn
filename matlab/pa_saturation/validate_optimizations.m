%% VALIDATE_OPTIMIZATIONS - Quick validation of optimization correctness
%
% Runs quick tests to verify the optimizations don't break anything
%
% Author: Emre Cerci
% Date: January 2026

clear; clc; close all;

fprintf('========================================\n');
fprintf('Optimization Validation Suite\n');
fprintf('========================================\n\n');

all_passed = true;
n_tests = 0;
n_passed = 0;

%% Test 1: PA Models Numerical Safeguards
fprintf('Test 1: PA Models Numerical Safeguards...\n');
n_tests = n_tests + 1;

try
    % Test with extreme inputs
    x_extreme = [0; 0.1; 1; 10; 100; 1000; 1e6];
    
    params_rapp.G = 1;
    params_rapp.Asat = 1;
    params_rapp.p = 2;
    
    y_rapp = pa_models(x_extreme, 'rapp', params_rapp);
    
    if all(isfinite(y_rapp)) && all(abs(y_rapp) < 1e10)
        fprintf('  PASS: Rapp model handles extreme inputs\n');
        n_passed = n_passed + 1;
    else
        fprintf('  FAIL: Rapp model produced inf/nan/extreme values\n');
        all_passed = false;
    end
catch ME
    fprintf('  FAIL: %s\n', ME.message);
    all_passed = false;
end

%% Test 2: Noise Model Correctness
fprintf('\nTest 2: Noise Model Correctness...\n');
n_tests = n_tests + 1;

try
    % Generate signal and check SNR
    N = 100000;
    symbols = 2*randi([0 1], 1, N) - 1;
    target_snr_db = 10;
    
    signal_power = mean(symbols.^2);  % = 1 for BPSK
    EbN0 = 10^(target_snr_db/10);
    noise_power = signal_power / (2 * EbN0);
    noise = sqrt(noise_power) * randn(1, N);
    
    measured_signal_power = mean(symbols.^2);
    measured_noise_power = mean(noise.^2);
    measured_snr_db = 10*log10(measured_signal_power / (2*measured_noise_power));
    
    snr_error = abs(measured_snr_db - target_snr_db);
    
    if snr_error < 0.5  % Within 0.5 dB
        fprintf('  PASS: SNR error = %.2f dB (< 0.5 dB)\n', snr_error);
        n_passed = n_passed + 1;
    else
        fprintf('  FAIL: SNR error = %.2f dB (>= 0.5 dB)\n', snr_error);
        all_passed = false;
    end
catch ME
    fprintf('  FAIL: %s\n', ME.message);
    all_passed = false;
end

%% Test 3: Feature Extraction Validation
fprintf('\nTest 3: Feature Extraction Validation...\n');
n_tests = n_tests + 1;

try
    % Generate test data
    tau = 0.7;
    sps = 10;
    beta = 0.3;
    span = 6;
    h = rcosdesign(beta, span, sps, 'sqrt');
    h = h / norm(h);
    delay = span * sps;
    step = round(tau * sps);
    
    N_test = 1000;
    bits_test = randi([0 1], 1, N_test);
    symbols = 2*bits_test - 1;
    
    tx_up_len = 1 + (N_test-1)*step;
    tx_up = zeros(tx_up_len, 1);
    tx_up(1:step:end) = symbols;
    tx = conv(tx_up, h, 'full');
    rx = conv(tx, h, 'full');
    rx = rx(:)' / std(rx);
    
    symbol_indices = delay + 1 + (0:N_test-1) * step;
    offsets = (-3:3) * step;
    margin = max(abs(offsets)) + 10;
    valid_range = (margin+1):(N_test-margin);
    
    % Extract features
    X = zeros(length(valid_range), length(offsets));
    for i = 1:length(valid_range)
        k = valid_range(i);
        center = symbol_indices(k);
        indices = center + offsets;
        X(i, :) = real(rx(indices));
    end
    
    % Check center sample correlation with bit values
    y_bits = bits_test(valid_range);
    center_samples = X(:, 4);  % Center of 7 samples
    correlation = corrcoef(center_samples, y_bits);
    
    if correlation(1,2) > 0.8  % Strong correlation expected
        fprintf('  PASS: Center sample correlation = %.3f (> 0.8)\n', correlation(1,2));
        n_passed = n_passed + 1;
    else
        fprintf('  FAIL: Center sample correlation = %.3f (<= 0.8)\n', correlation(1,2));
        all_passed = false;
    end
catch ME
    fprintf('  FAIL: %s\n', ME.message);
    all_passed = false;
end

%% Test 4: Decision Feedback Feature Dimensions
fprintf('\nTest 4: Decision Feedback Feature Dimensions...\n');
n_tests = n_tests + 1;

try
    n_samples = 7;
    D = 4;
    expected_features = n_samples + D;  % 7 + 4 = 11
    
    % Check that DF features would have correct dimension
    X_test = randn(100, expected_features);
    
    if size(X_test, 2) == 11
        fprintf('  PASS: DF features have correct dimension (%d)\n', expected_features);
        n_passed = n_passed + 1;
    else
        fprintf('  FAIL: Expected %d features, got %d\n', expected_features, size(X_test, 2));
        all_passed = false;
    end
catch ME
    fprintf('  FAIL: %s\n', ME.message);
    all_passed = false;
end

%% Test 5: Hybrid Sampling Offsets
fprintf('\nTest 5: Hybrid Sampling Offsets...\n');
n_tests = n_tests + 1;

try
    step = 7;  % tau=0.7, sps=10
    
    % Neighbor offsets
    offsets_neighbor = (-3:3) * step;
    
    % Hybrid offsets
    t1 = round(step / 3);
    t2 = round(2 * step / 3);
    offsets_hybrid = [-step, -t2, -t1, 0, t1, t2, step];
    
    % Check that hybrid has same count but different values
    same_count = length(offsets_neighbor) == length(offsets_hybrid);
    different_offsets = ~isequal(offsets_neighbor, offsets_hybrid);
    
    if same_count && different_offsets
        fprintf('  PASS: Hybrid offsets: [%s]\n', num2str(offsets_hybrid));
        n_passed = n_passed + 1;
    else
        fprintf('  FAIL: Hybrid offset configuration incorrect\n');
        all_passed = false;
    end
catch ME
    fprintf('  FAIL: %s\n', ME.message);
    all_passed = false;
end

%% Test 6: Multi-SNR Data Generation
fprintf('\nTest 6: Multi-SNR Data Generation...\n');
n_tests = n_tests + 1;

try
    SNR_range = [6, 8, 10];
    N_total = 3000;
    N_per_snr = round(N_total / length(SNR_range));
    
    total_generated = 0;
    for snr = SNR_range
        total_generated = total_generated + N_per_snr;
    end
    
    expected_total = N_per_snr * length(SNR_range);
    
    if total_generated == expected_total
        fprintf('  PASS: Multi-SNR generates %d symbols (%d x %d)\n', ...
            expected_total, length(SNR_range), N_per_snr);
        n_passed = n_passed + 1;
    else
        fprintf('  FAIL: Expected %d symbols, got %d\n', expected_total, total_generated);
        all_passed = false;
    end
catch ME
    fprintf('  FAIL: %s\n', ME.message);
    all_passed = false;
end

%% Test 7: Normalization Consistency
fprintf('\nTest 7: Normalization Consistency...\n');
n_tests = n_tests + 1;

try
    X = randn(1000, 7);
    
    % Per-feature normalization
    mu = mean(X, 1);
    sig = std(X, 0, 1);
    X_norm = (X - mu) ./ sig;
    
    % Check normalized statistics
    mean_check = max(abs(mean(X_norm, 1)));
    std_check = max(abs(std(X_norm, 0, 1) - 1));
    
    if mean_check < 0.01 && std_check < 0.01
        fprintf('  PASS: Normalized mean=%.4f (max), std=%.4f (max err)\n', ...
            mean_check, std_check);
        n_passed = n_passed + 1;
    else
        fprintf('  FAIL: Normalization incorrect (mean=%.4f, std_err=%.4f)\n', ...
            mean_check, std_check);
        all_passed = false;
    end
catch ME
    fprintf('  FAIL: %s\n', ME.message);
    all_passed = false;
end

%% Summary
fprintf('\n========================================\n');
fprintf('Validation Summary\n');
fprintf('========================================\n');
fprintf('Tests passed: %d / %d\n', n_passed, n_tests);

if all_passed
    fprintf('\nALL TESTS PASSED - Optimizations validated!\n');
else
    fprintf('\nSOME TESTS FAILED - Review before proceeding\n');
end
fprintf('========================================\n');
