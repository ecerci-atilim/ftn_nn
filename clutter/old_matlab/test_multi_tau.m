%% FTN Full Comparison - NN vs Classical vs M-BCJR
% τ = 0.6, 0.7, 0.8

clear; clc; close all;

%% Parameters
tau_values = [0.6, 0.7, 0.8];
SNR_range = 0:2:14;
N = 50000;
min_errors = 50;  % For M-BCJR (slow)

sps = 10; span = 6; beta = 0.3;
h = rcosdesign(beta, span, sps, 'sqrt');
h = h / norm(h);
delay = span * sps;

% M-BCJR parameters
M_bcjr = 32;
gbK = 4;

%% Results storage
n_tau = length(tau_values);
n_snr = length(SNR_range);

ber_th = zeros(n_tau, n_snr);
ber_sss = zeros(n_tau, n_snr);
ber_bcjr = zeros(n_tau, n_snr);
ber_nb = zeros(n_tau, n_snr);
ber_win = zeros(n_tau, n_snr);

%% Main Loop
for t = 1:n_tau
    tau = tau_values(t);
    step = round(sps * tau);
    half_win = 15;
    num_fb = 4;
    num_nb = 3;
    
    fprintf('\n========================================\n');
    fprintf('=== τ = %.1f ===\n', tau);
    fprintf('========================================\n');
    
    % ISI coefficients
    hh = conv(h, h);
    [~, pk] = max(hh);
    g = hh(pk:step:end); g = g/g(1); g = g(abs(g)>0.01);
    fprintf('ISI taps: %d\n', length(g));
    
    % Spectral factorization for M-BCJR
    grev = g(end:-1:2);
    chISI = [grev, g];
    [v, ~] = sfact(chISI);
    v(abs(v) < 0.01) = [];
    M_T = length(v);
    BCJRdec = M_BCJR_decoder(v);
    fprintf('M-BCJR states: %d, M_T: %d\n\n', M_bcjr, M_T);
    
    % Load NN model for this tau
    model_file = sprintf('mat/comparison/tau%02d.mat', tau*10);
    if exist(model_file, 'file')
        load(model_file);
        has_nn = true;
        fprintf('Loaded NN: %s\n', model_file);
    else
        has_nn = false;
        fprintf('No NN model for τ=%.1f\n', tau);
    end
    
    %% SNR Loop
    for s = 1:n_snr
        SNR_dB = SNR_range(s);
        EbN0_lin = 10^(SNR_dB/10);
        noise_var = 1 / (2 * EbN0_lin);
        
        % Generate data
        bits = randi([0 1], N, 1);
        symbols = 2*bits - 1;
        
        % Transmit
        tx = conv(upsample(symbols, step), h);
        rx_noisy = tx + sqrt(noise_var) * randn(size(tx));
        rx_mf = conv(rx_noisy, h);
        rx_mf_norm = rx_mf / std(rx_mf);
        
        % Sample at symbol rate
        rx_sym = zeros(1, N);
        for i = 1:N
            idx = (i-1)*step + 1 + delay;
            if idx > 0 && idx <= length(rx_mf)
                rx_sym(i) = rx_mf(idx);
            end
        end
        
        % === THRESHOLD ===
        ber_th(t, s) = sum((rx_sym > 0) ~= bits') / N;
        
        % === SSSgbKSE ===
        det_sss = SSSgbKSE(rx_sym, g, gbK) > 0;
        ber_sss(t, s) = sum(det_sss ~= bits') / N;
        
        % === M-BCJR ===
        % Note: M-BCJR uses whitened channel model
        b_padded = [zeros(1, M_T), symbols', zeros(1, M_T)];
        sf = conv(b_padded, v);
        y_bcjr = sf(M_T+1 : 2*M_T+N-1);
        y_bcjr = y_bcjr + sqrt(noise_var) * randn(size(y_bcjr));
        
        ysoft = BCJRdec.step(y_bcjr', zeros(N, 1), ones(N, 1), M_bcjr)';
        bits_bcjr = ysoft > 0;
        ber_bcjr(t, s) = sum(bits_bcjr ~= bits') / N;
        
        % === NN Models (if available) ===
        if has_nn
            % Neighbors+DF
            err_nb = 0; cnt_nb = 0;
            df = zeros(num_fb, 1);
            for i = (num_nb+num_fb+1):N
                ctr = (i-1)*step + 1 + delay;
                if ctr - num_nb*step < 1 || ctr + num_nb*step > length(rx_mf_norm), continue; end
                
                feat = rx_mf_norm(ctr);
                for k = 1:num_nb, feat = [feat, rx_mf_norm(ctr - k*step)]; end
                for k = 1:num_nb, feat = [feat, rx_mf_norm(ctr + k*step)]; end
                feat = [feat, df'];
                
                [~, pred] = max(predict(net4, feat));
                pred_bit = pred - 1;
                if pred_bit ~= bits(i), err_nb = err_nb + 1; end
                cnt_nb = cnt_nb + 1;
                df = [2*pred_bit-1; df(1:end-1)];
            end
            ber_nb(t, s) = err_nb / cnt_nb;
            
            % Window+DF
            err_win = 0; cnt_win = 0;
            df = zeros(num_fb, 1);
            for i = (num_fb+1):N
                ctr = (i-1)*step + 1 + delay;
                if ctr - half_win < 1 || ctr + half_win > length(rx_mf_norm), continue; end
                
                feat = [rx_mf_norm(ctr-half_win:ctr+half_win)', df'];
                
                [~, pred] = max(predict(net6, feat));
                pred_bit = pred - 1;
                if pred_bit ~= bits(i), err_win = err_win + 1; end
                cnt_win = cnt_win + 1;
                df = [2*pred_bit-1; df(1:end-1)];
            end
            ber_win(t, s) = err_win / cnt_win;
        else
            ber_nb(t, s) = NaN;
            ber_win(t, s) = NaN;
        end
        
        fprintf('SNR=%2d: Th=%.2e, SSS=%.2e, BCJR=%.2e, Nb=%.2e, Win=%.2e\n', ...
            SNR_dB, ber_th(t,s), ber_sss(t,s), ber_bcjr(t,s), ber_nb(t,s), ber_win(t,s));
    end
end

%% Summary Table @ 10 dB
fprintf('\n\n');
fprintf('╔══════════════════════════════════════════════════════════════════╗\n');
fprintf('║                    BER @ 10 dB Summary                           ║\n');
fprintf('╠═══════╦═══════════╦═══════════╦═══════════╦═══════════╦═════════╣\n');
fprintf('║   τ   ║ Threshold ║  SSSgbKSE ║   M-BCJR  ║ Neighb+DF ║ Wind+DF ║\n');
fprintf('╠═══════╬═══════════╬═══════════╬═══════════╬═══════════╬═════════╣\n');

idx10 = find(SNR_range == 10);
for t = 1:n_tau
    fprintf('║  %.1f  ║  %.2e ║  %.2e ║  %.2e ║  %.2e ║ %.2e║\n', ...
        tau_values(t), ber_th(t,idx10), ber_sss(t,idx10), ber_bcjr(t,idx10), ...
        ber_nb(t,idx10), ber_win(t,idx10));
end
fprintf('╚═══════╩═══════════╩═══════════╩═══════════╩═══════════╩═════════╝\n');

%% Plots
figure('Position', [50 50 1400 400]);
colors_tau = {'b', 'r', 'g'};
ber_theory = qfunc(sqrt(2 * 10.^(SNR_range/10)));

for t = 1:n_tau
    subplot(1, 3, t);
    semilogy(SNR_range, ber_theory, 'k--', 'LineWidth', 2, 'DisplayName', 'AWGN');
    hold on;
    semilogy(SNR_range, ber_th(t,:), 'r-^', 'LineWidth', 1.5, 'DisplayName', 'Threshold');
    semilogy(SNR_range, ber_sss(t,:), 'g-v', 'LineWidth', 1.5, 'DisplayName', 'SSSgbKSE');
    semilogy(SNR_range, ber_bcjr(t,:), 'c-d', 'LineWidth', 2, 'DisplayName', 'M-BCJR');
    if ~isnan(ber_nb(t,1))
        semilogy(SNR_range, ber_nb(t,:), 'm-o', 'LineWidth', 2, 'DisplayName', 'Neighbors+DF');
        semilogy(SNR_range, ber_win(t,:), 'b-s', 'LineWidth', 2.5, 'MarkerSize', 9, 'DisplayName', 'Window+DF');
    end
    grid on;
    xlabel('E_b/N_0 (dB)');
    ylabel('BER');
    title(sprintf('τ = %.1f', tau_values(t)));
    legend('Location', 'southwest', 'FontSize', 8);
    ylim([1e-5 1]);
end

sgtitle('FTN Detection Comparison', 'FontSize', 14, 'FontWeight', 'bold');

if ~exist('figures', 'dir'), mkdir('figures'); end
saveas(gcf, 'figures/full_comparison_all_tau.png');

%% Save
save('mat/comparison/full_results.mat', 'tau_values', 'SNR_range', ...
    'ber_th', 'ber_sss', 'ber_bcjr', 'ber_nb', 'ber_win');
fprintf('\nSaved: mat/comparison/full_results.mat\n');

%% Functions
function out = SSSgbKSE(in, ISI, K)
    N = length(in); L = length(ISI);
    ssd = zeros(1,N); out = zeros(1,N);
    for n = 1:N
        if n == 1, ssd(n) = in(n);
        else
            c = 0;
            for k = 1:min(n-1,L-1)
                if k+1<=L, c = c + sign(ssd(n-k))*ISI(k+1); end
            end
            ssd(n) = in(n) - c;
        end
        if n > K
            for i = (n-K):n
                v = in(i);
                for k = 1:min(n-i,L-1)
                    if k+1<=L && i+k<=n, v = v - sign(ssd(i+k))*ISI(k+1); end
                end
                for k = 1:min(i-1,L-1)
                    if k+1<=L && i-k>=1, v = v - sign(out(i-k))*ISI(k+1); end
                end
                out(i) = v;
            end
        end
    end
    out(out==0) = ssd(out==0);
end