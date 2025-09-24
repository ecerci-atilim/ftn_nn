% =========================================================================
% QUASI-STATIC CHANNEL DF-CNN TESTING: BER Performance Analysis
% =========================================================================
% Tests DF-CNN performance in quasi-static fading channels with multipath
% Compares against theoretical fading channel limits
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
    [symbol_errors, bit_errors, total_symbols, total_bits] = run_quasi_static_test(quasi_static_model, ...
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

%% HELPER FUNCTION - QUASI-STATIC CHANNEL TESTING
function [sym_errors, bit_errors, total_symbols, total_bits] = run_quasi_static_test(model, mod_type, M, phase, tau, SNR_dB, coh_len, ch_taps, max_delay, fading_type, win_len, num_feedback_taps)
    k = log2(M);
    constellation = generate_constellation(M, mod_type, phase);
    
    % Test parameters
    input_len = win_len*2 + num_feedback_taps + ch_taps;  % Include CSI
    half_win = floor(win_len/2);
    
    % Channel parameters
    is_real_modulation = (M == 2) && strcmpi(mod_type, 'psk');
    sps = 10; beta = 0.3; span = 6;
    h = rcosdesign(beta, span, sps, 'sqrt');
    
    sym_errors = 0; bit_errors = 0; total_symbols = 0; total_bits = 0;
    
    while sym_errors < 200 && total_symbols < 5e5  % More errors needed for fading statistics
        N_batch = 5000;  % Smaller batches for channel block management
        
        % Generate symbols
        symbol_indices = randi([0 M-1], N_batch, 1);
        symbols = constellation(symbol_indices + 1);
        
        % Pulse shaping
        tx_up = upsample(symbols, round(sps*tau));
        txSignal = conv(tx_up, h);
        pwr = mean(abs(txSignal).^2); 
        txSignal = txSignal / sqrt(pwr);
        
        % Generate channel realizations for this batch
        num_blocks = ceil(N_batch / coh_len);
        channel_responses = generate_channel_realizations(num_blocks, ch_taps, max_delay, fading_type);
        
        % Apply quasi-static channel
        rxSignal = apply_quasi_static_channel(txSignal, channel_responses, coh_len, sps, tau, SNR_dB, k, is_real_modulation);
        rxMF = conv(rxSignal, h);
        delay = finddelay(tx_up, rxMF);
        
        % Decision feedback history
        decision_history = zeros(num_feedback_taps, 1);
        
        for i = (num_feedback_taps + 1):N_batch
            loc = round((i-1)*sps*tau) + 1 + delay;
            if (loc > half_win) && (loc + half_win <= length(rxMF))
                % Extract signal features
                win_complex = rxMF(loc-half_win:loc+half_win);
                
                if is_real_modulation
                    win_features = [real(win_complex(:))', real(win_complex(:))'];
                else
                    win_features = [real(win_complex(:))', imag(win_complex(:))'];
                end
                
                % Get channel state information for this symbol
                block_idx = ceil(i / coh_len);
                if block_idx <= size(channel_responses, 1)
                    csi = channel_responses(block_idx, :);
                    % Convert complex CSI to real features
                    csi_features = [real(csi), imag(csi)];
                    csi_features = csi_features(1:ch_taps);  % Take first ch_taps elements
                else
                    csi_features = zeros(1, ch_taps);
                end
                
                % Combine all features
                test_input = [win_features, decision_history', csi_features];
                
                % Ensure correct input size
                if length(test_input) ~= input_len
                    % Pad or truncate if necessary
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
                
                % Symbol error
                if predicted_symbol_idx ~= symbol_indices(i)
                    sym_errors = sym_errors + 1;
                end
                
                % Bit errors
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
    
    fprintf('    Completed: %d symbol errors, %d bit errors, %d total symbols\n', sym_errors, bit_errors, total_symbols);
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
    
    % Approximate fading channel BER (simplified expressions)
    switch lower(fading_type)
        case 'rayleigh'
            if strcmpi(mod_type, 'psk') && M == 2
                % Exact BPSK Rayleigh fading formula
                gamma_bar = 10.^(SNR_dB/10);
                ber_fading = 0.5 * (1 - sqrt(gamma_bar ./ (1 + gamma_bar)));
            else
                % High SNR approximation for other modulations
                gamma_bar = 10.^(SNR_dB/10);
                % Approximate diversity gain (slope) difference
                ber_fading = ber_awgn .* (1./(4*gamma_bar));  % Rough approximation
                ber_fading = min(ber_fading, 0.5);  % Cap at 50%
            end
            
        case 'rician'
            % Approximation for Rician fading (between AWGN and Rayleigh)
            K_factor = 10^(3/10);  % 3 dB K-factor
            rayleigh_approx = calculate_fading_theory(SNR_dB, mod_type, M, 'rayleigh');
            ber_fading = ber_awgn + 0.5 * (rayleigh_approx - ber_awgn);
            
        case 'nakagami'
            % Approximation for Nakagami fading
            m = 1.5;  % Nakagami parameter
            rayleigh_approx = calculate_fading_theory(SNR_dB, mod_type, M, 'rayleigh');
            ber_fading = rayleigh_approx ./ m;  % Rough approximation
            
        otherwise
            ber_fading = ber_awgn;  % Default to AWGN if unknown
    end
    
    % Extract only ber_fading for return
    if exist('rayleigh_approx', 'var')
        ber_fading = rayleigh_approx(2,:);  % Get the ber_fading part
    end
end

function channel_responses = generate_channel_realizations(num_blocks, ch_taps, max_delay, fading_type)
    % Same as training script - generate channel realizations
    channel_responses = zeros(num_blocks, ch_taps);
    
    delays = linspace(0, max_delay, ch_taps);
    power_profile = exp(-delays / (max_delay/3));
    power_profile = power_profile / sum(power_profile);
    
    for block = 1:num_blocks
        switch lower(fading_type)
            case 'rayleigh'
                h = sqrt(power_profile/2) .* (randn(1, ch_taps) + 1j*randn(1, ch_taps));
            case 'rician'
                K = 10^(3/10);
                los_component = sqrt(K/(K+1)) * sqrt(power_profile);
                scattered_component = sqrt(power_profile/(2*(K+1))) .* (randn(1, ch_taps) + 1j*randn(1, ch_taps));
                h = los_component + scattered_component;
            case 'nakagami'
                m = 1.5;
                h = sqrt(power_profile) .* sqrt(gamrnd(m, 1/m, 1, ch_taps)) .* exp(1j*2*pi*rand(1, ch_taps));
            otherwise
                error('Unknown fading type: %s', fading_type);
        end
        channel_responses(block, :) = h;
    end
end

function rxSignal = apply_quasi_static_channel(txSignal, channel_responses, coh_len, sps, tau, SNR_dB, k, is_real_modulation)
    % Same as training script - apply channel and noise
    rxSignal = zeros(size(txSignal));
    
    samples_per_block = coh_len * round(sps * tau);
    
    for block = 1:size(channel_responses, 1)
        start_idx = (block-1) * samples_per_block + 1;
        end_idx = min(start_idx + samples_per_block - 1, length(txSignal));
        
        if start_idx <= length(txSignal)
            tx_block = txSignal(start_idx:end_idx);
            h = channel_responses(block, :);
            
            rx_block = conv(tx_block, h, 'same');
            rxSignal(start_idx:end_idx) = rx_block;
        end
    end
    
    % Add noise
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