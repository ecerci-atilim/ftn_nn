clear; clc; close all;

%% Parameters (NN ile aynı)
sps = 10;
span = 6;
beta_rrc = 0.3;
tau_values = [0.6, 0.7, 0.8];
SNR_range = 0:2:14;

% Detector parameters
M_bcjr = 32;        % BCJR states
gbK = 4;            % SSSgbKSE go-back length

% Monte Carlo
N_symbols = 100;
min_errors = 100;
max_bits = 1e6;

% Results storage
ber_bcjr = zeros(length(tau_values), length(SNR_range));
ber_kse = zeros(length(tau_values), length(SNR_range));

%% Main simulation
for t_idx = 1:length(tau_values)
    tau = tau_values(t_idx);
    fprintf('\n=== τ = %.1f ===\n', tau);
    
    % Generate pulse and ISI
    h = rcosdesign(beta_rrc, span, sps, 'sqrt');
    h = h / norm(h);
    step = round(sps * tau);
    
    % ISI sequence (from matched filter output)
    hh = conv(h, h);
    [~, peak_idx] = max(hh);
    g = hh(peak_idx : step : end);
    g = g / g(1);  % normalize so g(1) = 1
    g = g(abs(g) > 0.01);  % truncate small taps
    
    % Spectral factorization for BCJR
    grev = g(end:-1:2);
    chISI = [grev, g];
    [v, ~] = sfact(chISI);
    idx_remove = find(abs(v) < 0.01);
    v(idx_remove) = [];
    M_T = length(v);
    
    % Initialize BCJR decoder
    BCJRdec = M_BCJR_decoder(v);
    
    for s_idx = 1:length(SNR_range)
        SNR_dB = SNR_range(s_idx);
        fprintf('SNR = %d dB: ', SNR_dB);
        
        % Noise variance (NN ile aynı)
        EbN0_linear = 10^(SNR_dB/10);
        noise_var = 1 / (2 * EbN0_linear);
        
        errors_bcjr = 0;
        errors_kse = 0;
        total_bits = 0;
        
        while (errors_bcjr < min_errors || errors_kse < min_errors) && total_bits < max_bits
            % Generate data
            bits = randi([0 1], 1, N_symbols);
            symbols = 2 * bits - 1;
            
            % Transmit (NN ile aynı)
            tx = conv(upsample(symbols, step), h);
            noise = sqrt(noise_var) * randn(size(tx));
            rx = tx + noise;
            rx_mf = conv(rx, h);
            
            % Sample at symbol rate
            delay = span * sps;
            rx_sampled = zeros(1, N_symbols);
            for i = 1:N_symbols
                idx = (i-1) * step + 1 + delay;
                if idx <= length(rx_mf)
                    rx_sampled(i) = real(rx_mf(idx));
                end
            end
            
            %% M-BCJR Detection
            % Prepare input for BCJR (needs padding)
            b_padded = [zeros(1, M_T), symbols, zeros(1, M_T)];
            sf = conv(b_padded, v);
            y_bcjr = sf(M_T+1 : 2*M_T+N_symbols-1);
            % Add noise with same variance
            y_bcjr = y_bcjr + sqrt(noise_var) * randn(size(y_bcjr));
            
            % BCJR decode
            ysoft = BCJRdec.step(y_bcjr', zeros(N_symbols, 1), ones(N_symbols, 1), M_bcjr)';
            bits_bcjr = ysoft > 0;
            errors_bcjr = errors_bcjr + sum(bits_bcjr ~= bits);
            
            %% SSSgbKSE Detection
            mesSE = SSSgbKSE(rx_sampled, g, gbK);
            bits_kse = mesSE > 0;
            errors_kse = errors_kse + sum(bits_kse ~= bits);
            
            total_bits = total_bits + N_symbols;
        end
        
        ber_bcjr(t_idx, s_idx) = errors_bcjr / total_bits;
        ber_kse(t_idx, s_idx) = errors_kse / total_bits;
        
        fprintf('BCJR=%.2e, KSE=%.2e\n', ber_bcjr(t_idx, s_idx), ber_kse(t_idx, s_idx));
    end
end

%% Save results
save('classical_results.mat', 'SNR_range', 'tau_values', 'ber_bcjr', 'ber_kse');

%% Plot
figure('Position', [100 100 900 600]);
colors = {'b', 'r', 'g'};
ber_theory = qfunc(sqrt(2 * 10.^(SNR_range/10)));

semilogy(SNR_range, ber_theory, 'k--', 'LineWidth', 2, 'DisplayName', 'AWGN Theory');
hold on;

for t_idx = 1:length(tau_values)
    semilogy(SNR_range, ber_bcjr(t_idx,:), [colors{t_idx} '-o'], ...
        'LineWidth', 1.5, 'DisplayName', sprintf('M-BCJR (τ=%.1f)', tau_values(t_idx)));
    semilogy(SNR_range, ber_kse(t_idx,:), [colors{t_idx} '--s'], ...
        'LineWidth', 1.5, 'DisplayName', sprintf('SSSgbKSE (τ=%.1f)', tau_values(t_idx)));
end

grid on;
xlabel('E_b/N_0 (dB)');
ylabel('BER');
legend('Location', 'southwest');
title('Classical Detectors Performance');
ylim([1e-6 1]);
