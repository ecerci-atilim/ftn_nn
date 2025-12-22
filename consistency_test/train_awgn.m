%% FTN Simple Consistent Test
% Mevcut modelleri kullan, herkes AYNI koşullarda
%
% Emre Çerçi - Aralık 2025

clear; clc; close all;

%% Parameters
tau = 0.7;
SNR_range = 0:2:14;
N = 5000;  % Her SNR için sabit

sps = 10; span = 6;
h = rcosdesign(0.3, span, sps, 'sqrt');
h = h / norm(h);
step = round(sps * tau);
delay = span * sps;
half_win = 15;  % window_len = 31

% ISI
hh = conv(h, h);
[~, pk] = max(hh);
g = hh(pk:step:end); g = g/g(1); g = g(abs(g)>0.001);

fprintf('=== Simple FTN Test (τ=%.1f) ===\n\n', tau);

%% Load NN
load('mat/comparison/tau07_awgn.mat');  % veya mevcut model dosyası

%% Results
n_snr = length(SNR_range);
ber_th = zeros(1, n_snr);
ber_sss = zeros(1, n_snr);
ber_nb = zeros(1, n_snr);
ber_win = zeros(1, n_snr);

%% Main Loop
for s = 1:n_snr
    SNR_dB = SNR_range(s);
    
    % Generate ONE dataset for ALL detectors
    bits = randi([0 1], N, 1);
    symbols = 2*bits - 1;
    
    tx = conv(upsample(symbols, step), h);
    rx_noisy = awgn(tx, SNR_dB, 'measured');
    rx_mf = conv(rx_noisy, h);
    
    % === THRESHOLD ===
    err_th = 0;
    for i = 1:N
        idx = (i-1)*step + 1 + delay;
        if idx > 0 && idx <= length(rx_mf)
            if (rx_mf(idx) > 0) ~= bits(i), err_th = err_th + 1; end
        end
    end
    ber_th(s) = err_th / N;
    
    % === SSSgbKSE ===
    rx_sym = zeros(1, N);
    for i = 1:N
        idx = (i-1)*step + 1 + delay;
        if idx > 0 && idx <= length(rx_mf), rx_sym(i) = rx_mf(idx); end
    end
    det_sss = SSSgbKSE(rx_sym, g, 4) > 0;
    ber_sss(s) = sum(det_sss ~= bits') / N;
    
    % === NN: Neighbors+DF ===
    err_nb = 0;
    df = zeros(num_feedback, 1);
    for i = (num_neighbor+num_feedback+1):N
        ctr = (i-1)*step + 1 + delay;
        if ctr - num_neighbor*step < 1 || ctr + num_neighbor*step > length(rx_mf), continue; end
        
        % Features: center + neighbors + df
        feat = rx_mf(ctr);
        for k = 1:num_neighbor
            feat = [feat, rx_mf(ctr - k*step), rx_mf(ctr + k*step)];
        end
        feat = [feat, df'];
        
        prob = predict(net4, feat);
        [~, pred] = max(prob);
        pred_bit = pred - 1;
        
        if pred_bit ~= bits(i), err_nb = err_nb + 1; end
        df = [2*pred_bit-1; df(1:end-1)];
    end
    ber_nb(s) = err_nb / (N - num_neighbor - num_feedback);
    
    % === NN: Window+DF ===
    err_win = 0;
    df = zeros(num_feedback, 1);
    for i = (num_feedback+1):N
        ctr = (i-1)*step + 1 + delay;
        if ctr - half_win < 1 || ctr + half_win > length(rx_mf), continue; end
        
        % Features: window + df
        feat = rx_mf(ctr-half_win : ctr+half_win)';
        feat = [feat, df'];
        
        prob = predict(net6, feat);
        [~, pred] = max(prob);
        pred_bit = pred - 1;
        
        if pred_bit ~= bits(i), err_win = err_win + 1; end
        df = [2*pred_bit-1; df(1:end-1)];
    end
    ber_win(s) = err_win / (N - num_feedback);
    
    fprintf('SNR=%2d: Th=%.2e, SSS=%.2e, Nb+DF=%.2e, Win+DF=%.2e\n', ...
        SNR_dB, ber_th(s), ber_sss(s), ber_nb(s), ber_win(s));
end

%% Results
fprintf('\n=== @ 10 dB ===\n');
idx = find(SNR_range == 10);
fprintf('Threshold:    %.2e\n', ber_th(idx));
fprintf('SSSgbKSE:     %.2e\n', ber_sss(idx));
fprintf('Neighbors+DF: %.2e\n', ber_nb(idx));
fprintf('Window+DF:    %.2e\n', ber_win(idx));
fprintf('\nWindow+DF vs Neighbors+DF: %.1fx\n', ber_nb(idx)/ber_win(idx));

%% Plot
figure;
semilogy(SNR_range, qfunc(sqrt(2*10.^(SNR_range/10))), 'k--', 'LineWidth', 2, 'DisplayName', 'AWGN');
hold on;
semilogy(SNR_range, ber_th, 'r-^', 'LineWidth', 1.5, 'DisplayName', 'Threshold');
semilogy(SNR_range, ber_sss, 'g-v', 'LineWidth', 1.5, 'DisplayName', 'SSSgbKSE');
semilogy(SNR_range, ber_nb, 'm-o', 'LineWidth', 2, 'DisplayName', 'Neighbors+DF');
semilogy(SNR_range, ber_win, 'b-s', 'LineWidth', 2.5, 'DisplayName', 'Window+DF');
grid on; xlabel('SNR (dB)'); ylabel('BER');
legend('Location', 'southwest');
title(sprintf('FTN Detection (τ=%.1f)', tau));
ylim([1e-5 1]);

%% SSSgbKSE function
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