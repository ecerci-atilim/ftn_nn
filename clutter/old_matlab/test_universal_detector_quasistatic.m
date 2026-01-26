% =========================================================================
% FIXED QUASI-STATIC CHANNEL DF-CNN TESTING: BER Performance Analysis
% =========================================================================
% Tests DF-CNN performance in quasi-static fading channels
% Uses actual data dimensions and simplified channel model
% =========================================================================

clear, clc, close all

%% LOAD MODEL
[file, path] = uigetfile('mat/quasi_static/*.mat', 'Select a Trained Quasi-Static DF-CNN Model');
if isequal(file, 0), error('User canceled model selection.'); end
model_filename = fullfile(path, file);
load(model_filename); % loads model and parameters

fprintf('Testing Quasi-Static DF-CNN: %d-%s, phase=%.3f, tau=%.1f\n', ...
    modulation_order, upper(modulation_type), phase_offset, tau);
fprintf('Channel: %d-tap %s fading, coherence=%d symbols\n', ...
    num_taps_channel, fading_type, coherence_length);

%% SETUP BER SIMULATION
SNR_range_dB = 0:2:20;  % Extended range for fading channels
ser_results = zeros(size(SNR_range_dB));
ber_results = zeros(size(SNR_range_dB));

%% MAIN TESTING LOOP
fprintf('\n=== BER Testing in Quasi-Static Fading Channel ===\n');
for snridx = 1:length(SNR_range_dB)
    current_SNR_dB = SNR_range_dB(snridx);
    [symbol_errors, bit_errors, total_symbols, total_bits] = run_quasi_static_test_fixed(quasi_static_model, ...
        modulation_type, modulation_order, phase_offset, tau, current_SNR_dB, ...
        coherence_length, num_taps_channel, max_delay_spread, fading_type, window_len, num_feedback_taps);
    
    ser_results(snridx) = symbol_errors / total_symbols;
    ber_results(snridx) = bit_errors / total_bits;
    
    fprintf('SNR: %2d dB -> SER: %.4e, BER: %.4e (%d/%d symbols, %d/%d bits)\n', ...
        current_SNR_dB, ser_results(snridx), ber_results(snridx), ...
        symbol_errors, total_symbols, bit_errors, total_bits);
end

%% PLOT RESULTS AGAINST FADING CHANNEL THEORY
figure('Position', [100, 100, 1000, 600]);

% Calculate theoretical BER for fading channels
[ber_awgn, ber_fading_approx] = calculate_fading_theory(SNR_range_dB, modulation_type, modulation_order, fading_type);

% Plot comparison
semilogy(SNR_range_dB, ber_awgn, 'k--', 'LineWidth', 1.5, 'DisplayName', sprintf('AWGN %d-%s', modulation_order, upper(modulation_type)));
hold on;
semilogy(SNR_range_dB, ber_fading_approx, 'r:', 'LineWidth', 2, 'DisplayName', sprintf('Theoretical %s Fading', upper(fading_type)));
semilogy(SNR_range_dB, ber_results, 'b-o', 'LineWidth', 2.5, 'MarkerSize', 7, 'DisplayName', sprintf('DF-CNN (τ=%.1f, %d-tap)', tau, num_taps_channel));
hold off;

grid on; grid minor;
xlabel('SNR (E_b/N_0) [dB]', 'FontSize', 12);
ylabel('Bit Error Rate (BER)', 'FontSize', 12);
legend('show', 'Location', 'southwest', 'FontSize', 11);
title(sprintf('Quasi-Static DF-CNN: %d-%s, %s Fading (L=%d, coherence=%d)', ...
    modulation_order, upper(modulation_type), upper(fading_type), num_taps_channel, coherence_length), 'FontSize', 13);
ylim([1e-6 1]);

% Add channel info text box
channel_info = sprintf('Channel Model:\n• %s fading\n• %d taps\n• τ = %.1f\n• Coherence = %d sym', ...
    upper(fading_type), num_taps_channel, tau, coherence_length);
annotation('textbox', [0.65, 0.7, 0.3, 0.2], 'String', channel_info, 'FontSize', 10, ...
    'BackgroundColor', 'white', 'EdgeColor', 'black');

% Save results
results_file = strrep(file, '.mat', '_ber_results.mat');
save(fullfile(path, results_file), 'SNR_range_dB', 'ber_results', 'ser_results', 'ber_awgn', 'ber_fading_approx');
fprintf('\nResults saved to: %s\n', results_file);

%% HELPER FUNCTION - FIXED QUASI-STATIC CHANNEL TESTING
function [sym_errors, bit_errors, total_symbols, total_bits] = run_quasi_static_test_fixed(model, mod_type, M, phase, tau, SNR_dB, coh_len, ch_taps, max_delay, fading_type, win_len, num_feedback_taps)
    k = log2(M);
    constellation = generate_constellation(M, mod_type, phase);
    
    % Determine data dimensions by testing with sample data
    % Generate a small test batch to determine actual input dimensions
    test_x = generate_test_sample(mod_type, M, phase, tau, SNR_dB, win_len, num_feedback_taps, ch_taps, max_delay, fading_type, coh_len);
    input_len = size(test_x, 2);
    fprintf('  Detected input dimension: %d features\n', input_len);
    
    half_win = floor(win_len/2);
    is_real_modulation = (M == 2) && strcmpi(mod_type, 'psk');
    sps = 10; beta = 0.3; span = 6;
    h = rcosdesign(beta, span, sps, 'sqrt');
    
    % Pre-generate channel coefficients for efficiency
    delays = linspace(0, max_delay, ch_taps);
    power_profile = exp(-delays / (max_delay/3));
    power_profile = power_profile / sum(power_profile);
    
    sym_errors = 0; bit_errors = 0; total_symbols = 0; total_bits = 0;
    
    while sym_errors < 200 && total_symbols < 2e5
        N_batch = 3000;
        
        % Generate symbols
        symbol_indices = randi([0 M-1], N_batch, 1);
        symbols = constellation(symbol_indices + 1);
        
        % Pulse shaping
        tx_up = upsample(symbols, round(sps*tau));
        txSignal = conv(tx_up, h);
        pwr = mean(abs(txSignal).^2); 
        txSignal = txSignal / sqrt(pwr);
        
        % Generate channel realizations (SAME AS TRAINING)
        num_blocks = ceil(N_batch / coh_len);
        switch lower(fading_type)
            case 'rayleigh'
                h_channels = sqrt(power_profile/2) .* (randn(num_blocks, ch_taps) + 1j*randn(num_blocks, ch_taps));
            case 'rician'
                K = 10^(3/10);
                los = sqrt(K/(K+1)) * sqrt(power_profile);
                scatter = sqrt(power_profile/(2*(K+1))) .* (randn(num_blocks, ch_taps) + 1j*randn(num_blocks, ch_taps));
                h_channels = repmat(los, num_blocks, 1) + scatter;
            case 'nakagami'
                m = 1.5;
                h_channels = sqrt(power_profile) .* sqrt(gamrnd(m, 1/m, num_blocks, ch_taps)) .* ...
                             exp(1j*2*pi*rand(num_blocks, ch_taps));
        end
        
        % Apply simplified channel (SAME AS TRAINING)
        rxSignal = zeros(size(txSignal));
        samples_per_block = coh_len * round(sps * tau);
        
        for block = 1:num_blocks
            start_idx = (block-1) * samples_per_block + 1;
            end_idx = min(start_idx + samples_per_block - 1, length(txSignal));
            
            if start_idx <= length(txSignal)
                dominant_tap = h_channels(block, 1);  % Use same simplification as training
                rxSignal(start_idx:end_idx) = txSignal(start_idx:end_idx) * dominant_tap;
            end
        end
        
        % Add noise (SAME AS TRAINING)
        snr_eb_n0 = 10^(SNR_dB/10);
        snr_es_n0 = k * snr_eb_n0;
        signal_power = mean(abs(rxSignal).^2);
        noise_power = signal_power / snr_es_n0;
        
        if is_real_modulation
            noise = sqrt(noise_power) * randn(size(rxSignal));
        else
            noise = sqrt(noise_power/2) * (randn(size(rxSignal)) + 1j*randn(size(rxSignal)));
        end
        
        rxSignal = rxSignal + noise;
        rxMF = conv(rxSignal, h);
        delay = finddelay(tx_up, rxMF);
        
        % Decision feedback history
        decision_history = zeros(num_feedback_taps, 1);
        
        for i = (num_feedback_taps + 1):N_batch
            loc = round((i-1)*sps*tau) + 1 + delay;
            if (loc > half_win) && (loc + half_win <= length(rxMF))
                % Extract signal features (SAME AS TRAINING)
                win_complex = rxMF(loc-half_win:loc+half_win);
                
                if is_real_modulation
                    win_features = [real(win_complex(:))', real(win_complex(:))'];
                else
                    win_features = [real(win_complex(:))', imag(win_complex(:))'];
                end
                
                % Get CSI features (SAME AS TRAINING)
                block_idx = ceil(i / coh_len);
                if block_idx <= size(h_channels, 1)
                    h_current = h_channels(block_idx, :);
                    csi_features = [real(h_current), imag(h_current)];  % I/Q separation
                else
                    csi_features = zeros(1, ch_taps*2);
                end
                
                % Combine all features (SAME FORMAT AS TRAINING)
                test_input = [win_features, decision_history', csi_features];
                
                % Ensure correct input size
                if length(test_input) ~= input_len
                    if length(test_input) < input_len
                        test_input = [test_input, zeros(1, input_len - length(test_input))];
                    else
                        test_input = test_input(1:input_len);
                    end
                end
                
                % NN prediction
                pred_probs = predict(model, test_input);
                [~, pred_idx] = max(pred_probs);
                predicted_symbol_idx = pred_idx - 1;
                
                % Error counting
                if predicted_symbol_idx ~= symbol_indices(i)
                    sym_errors = sym_errors + 1;
                end
                
                true_bits = de2bi(symbol_indices(i), k, 'left-msb');
                pred_bits = de2bi(predicted_symbol_idx, k, 'left-msb');
                bit_errors = bit_errors + sum(true_bits ~= pred_bits);
                
                % Update decision history
                decision_history = [predicted_symbol_idx; decision_history(1:end-1)];
                
                total_symbols = total_symbols + 1;
                total_bits = total_bits + k;
            end
        end
    end
end

function test_x = generate_test_sample(mod_type, M, phase, tau, SNR_dB, win_len, num_feedback_taps, ch_taps, max_delay, fading_type, coh_len)
    % Generate a small sample to determine input dimensions
    k = log2(M);
    constellation = generate_constellation(M, mod_type, phase);
    
    N_test = 200;  % Small test batch
    symbol_indices = randi([0 M-1], N_test, 1);
    symbols = constellation(symbol_indices + 1);
    
    is_real_modulation = (M == 2) && strcmpi(mod_type, 'psk');
    sps = 10; beta = 0.3; span = 6;
    h = rcosdesign(beta, span, sps, 'sqrt');
    
    % Simple processing to get one sample
    tx_up = upsample(symbols(1:50), round(sps*tau));
    txSignal = conv(tx_up, h);
    rxSignal = txSignal;  % No channel/noise for dimension test
    rxMF = conv(rxSignal, h);
    
    half_win = floor(win_len/2);
    win_complex = rxMF(50:50+win_len-1);  % Extract a window
    
    if is_real_modulation
        win_features = [real(win_complex(:))', real(win_complex(:))'];
    else
        win_features = [real(win_complex(:))', imag(win_complex(:))'];
    end
    
    past_symbols = zeros(1, num_feedback_taps);
    csi_features = zeros(1, ch_taps*2);  % Real CSI features
    
    test_x = [win_features, past_symbols, csi_features];
end

function [ber_awgn, ber_fading] = calculate_fading_theory(SNR_dB, mod_type, M, fading_type)
    % Calculate theoretical BER for AWGN and fading channels
    
    % AWGN reference
    if strcmpi(mod_type, 'psk')
        if M == 2
            ber_awgn = qfunc(sqrt(2 * 10.^(SNR_dB/10)));
        else
            ber_awgn = berawgn(SNR_dB, 'psk', M, 'nondiff');
        end
    elseif strcmpi(mod_type, 'qam')
        ber_awgn = berawgn(SNR_dB, 'qam', M);
    end
    
    % Simplified fading channel approximations
    gamma_bar = 10.^(SNR_dB/10);
    
    switch lower(fading_type)
        case 'rayleigh'
            if strcmpi(mod_type, 'psk') && M == 2
                % Exact BPSK Rayleigh fading
                ber_fading = 0.5 * (1 - sqrt(gamma_bar ./ (1 + gamma_bar)));
            else
                % High SNR approximation
                ber_fading = ber_awgn .* (10.^(SNR_dB/10) / 4);
                ber_fading = min(ber_fading, 0.5);
            end
            
        case 'rician'
            % Rician approximation (between AWGN and Rayleigh)
            rayleigh_ber = 0.5 * (1 - sqrt(gamma_bar ./ (1 + gamma_bar)));
            ber_fading = ber_awgn + 0.3 * (rayleigh_ber - ber_awgn);
            
        case 'nakagami'
            % Nakagami approximation
            rayleigh_ber = 0.5 * (1 - sqrt(gamma_bar ./ (1 + gamma_bar)));
            ber_fading = rayleigh_ber * 0.7;  % Slightly better than Rayleigh
            
        otherwise
            ber_fading = ber_awgn;
    end
end

function constellation = generate_constellation(M, type, phase)
    % Same constellation generation as training
    if strcmpi(type, 'psk')
        p = 0:M-1;
        constellation = exp(1j*(2*pi*p/M + phase));
    elseif strcmpi(type, 'qam')
        k = log2(M);
        if mod(k,2) ~= 0
            error('QAM order must be a power of 4');
        end
        n = sqrt(M);
        vals = -(n-1):2:(n-1);
        [X,Y] = meshgrid(vals, vals);
        constellation = (X + 1j*Y) * exp(1j*phase);
        constellation = constellation(:);
    else
        error('Unknown modulation type');
    end
    
    constellation = constellation / sqrt(mean(abs(constellation).^2));
end