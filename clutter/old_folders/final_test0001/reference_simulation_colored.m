%% Reference Simulation: M-BCJR and SSSgbKSE (Parallelized)
% Generate baseline BER curves for comparison with NN approaches

clear; clc; close all;

%% Parameters
tau_values = [0.5, 0.6, 0.7, 0.8, 0.9];
SNR_range = 0:2:14;

N_block = 10000;
min_errors = 100;
max_symbols = 1e6;

sps = 10;
span = 6;
beta = 0.3;

% SSSgbKSE parameters
gbK = 1;

%% Create output directory
[~,~] = mkdir('results');

%% Generate pulse
h = rcosdesign(beta, span, sps, 'sqrt');
h = h / norm(h);
delay = span * sps;

%% Main simulation
for tau_idx = 1:length(tau_values)
    tau = tau_values(tau_idx);
    step = round(tau * sps);
    
    % Check if already done
    result_file = sprintf('results/reference_tau%.1f.mat', tau);
    if exist(result_file, 'file')
        fprintf('tau = %.1f already done, skipping...\n', tau);
        continue;
    end
    
    fprintf('\n========================================\n');
    fprintf('  Reference: tau = %.1f (step = %d)\n', tau, step);
    fprintf('========================================\n');
    
    % Compute ISI coefficients
    hh = conv(h, h);
    [~, pk] = max(hh);
    g = hh(pk:step:end); 
    g = g / g(1); 
    % Truncate from end, but keep intermediate small values
    last_significant = find(abs(g) > 0.01, 1, 'last');
    g = g(1:last_significant);
    fprintf('ISI taps: %d\n', length(g));
    
    % Spectral factorization for M-BCJR
    grev = g(end:-1:2);
    chISI = [grev, g];
    [v, ~] = sfact(chISI);
    v(abs(v) < 0.01) = [];
    M_T = length(v);
    
    % Adaptive M-BCJR states based on ISI length
    M_bcjr = 2^(length(g)-1);
    fprintf('M-BCJR states: %d, M_T: %d\n', M_bcjr, M_T);
    fprintf('Running with parfor (%d SNR points)...\n\n', length(SNR_range));
    
    % Preallocate
    n_snr = length(SNR_range);
    BER_th = zeros(1, n_snr);
    BER_sss = zeros(1, n_snr);
    BER_bcjr = zeros(1, n_snr);
    
    % Parallel loop over SNR
    parfor snr_idx = 1:n_snr
        SNR_dB = SNR_range(snr_idx);
        EbN0_lin = 10^(SNR_dB/10);
        noise_var = 1 / (2 * EbN0_lin);
        
        % Each worker creates its own decoder
        BCJRdec_local = M_BCJR_decoder(v);
        
        %% Threshold
        total_errors_th = 0;
        total_symbols_th = 0;
        block_idx = 0;
        
        while total_errors_th < min_errors && total_symbols_th < max_symbols
            block_idx = block_idx + 1;
            rng(1000*snr_idx + block_idx);
            
            bits = randi([0 1], N_block, 1);
            symbols = 2*bits - 1;
            
            tx = conv(upsample(symbols, step), h);
            rx_noisy = tx + sqrt(noise_var) * randn(size(tx));
            rx_mf = conv(rx_noisy, h);
            
            rx_sym = zeros(1, N_block);
            for i = 1:N_block
                idx_s = (i-1)*step + 1 + delay;
                if idx_s > 0 && idx_s <= length(rx_mf)
                    rx_sym(i) = rx_mf(idx_s);
                end
            end
            
            bits_hat = (rx_sym > 0)';
            total_errors_th = total_errors_th + sum(bits_hat ~= bits);
            total_symbols_th = total_symbols_th + N_block;
        end
        BER_th(snr_idx) = total_errors_th / total_symbols_th;
        
        %% SSSgbKSE
        total_errors_sss = 0;
        total_symbols_sss = 0;
        block_idx = 0;
        
        while total_errors_sss < min_errors && total_symbols_sss < max_symbols
            block_idx = block_idx + 1;
            rng(2000*snr_idx + block_idx);
            
            bits = randi([0 1], N_block, 1);
            symbols = 2*bits - 1;
            
            tx = conv(upsample(symbols, step), h);
            rx_noisy = tx + sqrt(noise_var) * randn(size(tx));
            rx_mf = conv(rx_noisy, h);
            
            rx_sym = zeros(1, N_block);
            for i = 1:N_block
                idx_s = (i-1)*step + 1 + delay;
                if idx_s > 0 && idx_s <= length(rx_mf)
                    rx_sym(i) = rx_mf(idx_s);
                end
            end
            
            det_sss = SSSgbKSE_local(rx_sym, g, gbK) > 0;
            total_errors_sss = total_errors_sss + sum(det_sss' ~= bits);
            total_symbols_sss = total_symbols_sss + N_block;
        end
        BER_sss(snr_idx) = total_errors_sss / total_symbols_sss;
        
        %% M-BCJR (whitened model - optimal receiver)
        % Uses same bits/noise seed as others for fair comparison
        total_errors_bcjr = 0;
        total_symbols_bcjr = 0;
        block_idx = 0;
        
        while total_errors_bcjr < min_errors && total_symbols_bcjr < max_symbols
            block_idx = block_idx + 1;
            rng(1000*snr_idx + block_idx);  % SAME seed as Threshold
            
            bits = randi([0 1], N_block, 1);
            symbols = 2*bits - 1;
            
            % M-BCJR uses whitened channel (optimal for colored noise)
            b_padded = [zeros(1, M_T), symbols', zeros(1, M_T)];
            sf = conv(b_padded, v);
            y_bcjr = sf(M_T+1 : 2*M_T+N_block-1);
            y_bcjr = y_bcjr + sqrt(noise_var) * randn(size(y_bcjr));
            
            ysoft = BCJRdec_local.step(y_bcjr', zeros(N_block, 1), ones(N_block, 1), M_bcjr)';
            bits_hat = (ysoft > 0)';
            
            total_errors_bcjr = total_errors_bcjr + sum(bits_hat ~= bits);
            total_symbols_bcjr = total_symbols_bcjr + N_block;
        end
        BER_bcjr(snr_idx) = total_errors_bcjr / total_symbols_bcjr;
        
        fprintf('SNR=%2d: Th=%.2e, SSS=%.2e, BCJR=%.2e\n', ...
                SNR_dB, BER_th(snr_idx), BER_sss(snr_idx), BER_bcjr(snr_idx));
    end
    
    %% Save results
    results_ref.threshold.BER = BER_th;
    results_ref.threshold.SNR = SNR_range;
    results_ref.sss.BER = BER_sss;
    results_ref.sss.SNR = SNR_range;
    results_ref.bcjr.BER = BER_bcjr;
    results_ref.bcjr.SNR = SNR_range;
    results_ref.tau = tau;
    results_ref.params.M_bcjr = M_bcjr;
    results_ref.params.gbK = gbK;
    results_ref.params.g = g;
    
    save(result_file, 'results_ref');
    fprintf('\nSaved: %s\n', result_file);
    
    % Print summary
    fprintf('\nSummary for tau = %.1f:\n', tau);
    fprintf('SNR:  '); fprintf('%6d ', SNR_range); fprintf('\n');
    fprintf('Th:   '); fprintf('%.1e ', BER_th); fprintf('\n');
    fprintf('SSS:  '); fprintf('%.1e ', BER_sss); fprintf('\n');
    fprintf('BCJR: '); fprintf('%.1e ', BER_bcjr); fprintf('\n');
end

fprintf('\n========================================\n');
fprintf('  Reference simulation complete!\n');
fprintf('========================================\n');

%% ===================== HELPER FUNCTIONS =====================

function out = SSSgbKSE_local(in, ISI, K)
    N = length(in); 
    L = length(ISI);
    ssd = zeros(1, N); 
    out = zeros(1, N);
    
    for n = 1:N
        if n == 1
            ssd(n) = in(n);
        else
            c = 0;
            for k = 1:min(n-1, L-1)
                if k+1 <= L
                    c = c + sign(ssd(n-k)) * ISI(k+1);
                end
            end
            ssd(n) = in(n) - c;
        end
        
        if n > K
            for i = (n-K):n
                vv = in(i);
                for k = 1:min(n-i, L-1)
                    if k+1 <= L && i+k <= n
                        vv = vv - sign(ssd(i+k)) * ISI(k+1);
                    end
                end
                for k = 1:min(i-1, L-1)
                    if k+1 <= L && i-k >= 1
                        vv = vv - sign(out(i-k)) * ISI(k+1);
                    end
                end
                out(i) = vv;
            end
        end
    end
    out(out == 0) = ssd(out == 0);
end