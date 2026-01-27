%% Reference Simulation: M-BCJR and SSSgbKSE (Fixed with Whitening)
% Comparison with AI-based Receiver on physical channel data

clear; clc; close all;

%% Parameters
tau_values = [0.7]; % Test için problemli olan 0.7'yi seçtim
SNR_range = 10;     % Kıyaslama noktası (10 dB)

N_block = 10000;
min_errors = 100;
max_symbols = 1e6;

sps = 10;
span = 6;
beta = 0.35;

% M-BCJR parameters
M_bcjr = 32; % 32 State = 5 Memory (L=5)

% SSSgbKSE parameters
gbK = 1;

%% Generate pulse
h = rcosdesign(beta, span, sps, 'sqrt');
h = h / norm(h); % Enerji normalizasyonu
delay = span * sps;

%% Main simulation
for tau_idx = 1:length(tau_values)
    tau = tau_values(tau_idx);
    step = round(tau * sps);
    
    fprintf('\n========================================\n');
    fprintf('  Reference: tau = %.1f (step = %d)\n', tau, step);
    fprintf('========================================\n');
    
    % 1. Compute G(z) (Autocorrelation / Ungerboeck Model)
    hh = conv(h, h);
    [~, pk] = max(hh);
    g = hh(pk:step:end); 
    
    % Enerji koruyarak normalizasyon (Önemli!)
    % g(1) (center tap) aslında sinyal enerjisidir.
    % sfact için g'yi olduğu gibi bırakıyoruz, sonra v'yi normalize edeceğiz.
    
    % 2. Compute V(z) (Minimum Phase / Forney Model) via Spectral Factorization
    % G(z) simetriktir, sfact bunu V(z) * V(z^-1) olarak ayırır.
    grev = g(end:-1:2);
    chISI = [grev, g];
    [v, ~] = sfact(chISI);
    
    % 3. Truncation (BCJR Hafızasına Uydurma)
    % BCJR 32 state ise, hafıza 5'tir. Filtre en fazla 5+1=6 tap olmalı.
    memory_len = log2(M_bcjr);
    target_len = memory_len + 1;
    
    if length(v) > target_len
        fprintf('UYARI: Filtre boyu (%d), BCJR hafızasından (%d) büyük.\n', length(v), memory_len);
        fprintf('       Kuyruk kesiliyor (Truncation applied)...\n');
        
        v_full = v;
        v = v(1:target_len); % Kesme işlemi
        
        % Kesme sonrası enerji kaybını dengelemek için tekrar normalize et
        % (Sinyal/Gürültü oranını korumak için)
        v = v / norm(v) * norm(v_full(1:target_len)); 
    end
    
    M_T = length(v) - 1; % Efektif hafıza
    fprintf('Final BCJR Filter (v): '); fprintf('%.3f ', v); fprintf('\n');
    
    % Preallocate
    n_snr = length(SNR_range);
    BER_bcjr = zeros(1, n_snr);
    
    % Parallel loop over SNR
    % parfor snr_idx = 1:n_snr % Debug için 'for' yapabilirsiniz
    for snr_idx = 1:n_snr
        SNR_dB = SNR_range(snr_idx);
        EbN0_lin = 10^(SNR_dB/10);
        noise_var = 1 / (2 * EbN0_lin);
        
        % Create decoder compatible with TRUNCATED filter v
        BCJRdec_local = M_BCJR_decoder(v);
        
        %% M-BCJR (With Whitening Filter)
        total_errors_bcjr = 0;
        total_symbols_bcjr = 0;
        block_idx = 0;
        
        while total_errors_bcjr < min_errors && total_symbols_bcjr < max_symbols
            block_idx = block_idx + 1;
            rng(3000*snr_idx + block_idx);
            
            % --- TX SIDE ---
            bits = randi([0 1], N_block, 1);
            symbols = 2*bits - 1;
            
            % Fiziksel Kanal Simülasyonu
            tx = conv(upsample(symbols, step), h);
            
            % --- CHANNEL ---
            rx_noisy = tx + sqrt(noise_var) * randn(size(tx));
            
            % --- RX SIDE (MATCHED FILTER) ---
            % Bu sinyal RENKLİ GÜRÜLTÜ içerir ve G(z) ile şekillenmiştir
            rx_mf = conv(rx_noisy, h);
            
            % Downsampling (Sembol hızına inme)
            rx_sym = zeros(1, N_block);
            for i = 1:N_block
                idx_s = (i-1)*step + 1 + delay;
                if idx_s > 0 && idx_s <= length(rx_mf)
                    rx_sym(i) = rx_mf(idx_s);
                end
            end
            
            % --- WHITENING FILTER (Kritik Düzeltme) ---
            % Amaç: G(z) + Renkli Gürültü --> V(z) + Beyaz Gürültü
            % İşlem: 1/V*(z^-1) ile filtreleme (Anti-causal)
            % Yöntem: Sinyali ters çevir -> 1/V(z) ile filtrele -> Tekrar ters çevir
            
            % 1. Zamanı tersine çevir
            rx_rev = fliplr(rx_sym);
            
            % 2. Kararlı IIR Filtreleme (1/v)
            % v minimum fazlı olduğu için 1/v kararlıdır.
            % Initial state (zi) yönetimi önemlidir ama blok uzunluğu (10000)
            % büyük olduğu için ihmal edilebilir veya padding yapılabilir.
            w_rev = filter(1, v, [rx_rev zeros(1, 20)]); % Padding eklendi
            
            % 3. Zamanı tekrar düzelt (ve padding'i at)
            rx_white = fliplr(w_rev);
            rx_white = rx_white(21:end); % Padding offset düzeltmesi
            
            % --- BCJR DECODING ---
            % Artık rx_white, V(z) ile şekillenmiş ve Beyaz Gürültülü kabul edilebilir.
            
            % BCJR'a giren gürültü varyansı teorik olarak noise_var ile aynı kalır
            % (Eğer sfact enerji korunumluysa).
            % BCJR_decoder nesneniz, step fonksiyonunda muhtemelen LLR hesabı için
            % noise variance'a ihtiyaç duyar veya 1 kabul eder.
            % Eğer dışarıdan noise parametresi almıyorsa, sinyali ölçeklemek gerekebilir.
            % Genelde: y_input = y / noise_var;
            
            % Not: Sizin BCJRdec_local.step fonksiyonunuzun içi bilinmediği için
            % standart parametrelerle çağırıyorum.
            ysoft = BCJRdec_local.step(rx_white', zeros(N_block, 1), ones(N_block, 1), M_bcjr)';
            
            bits_hat = (ysoft > 0)';
            
            total_errors_bcjr = total_errors_bcjr + sum(bits_hat ~= bits);
            total_symbols_bcjr = total_symbols_bcjr + N_block;
        end
        BER_bcjr(snr_idx) = total_errors_bcjr / total_symbols_bcjr;
        
        fprintf('SNR=%2d: BCJR (Whitened)=%.2e\n', SNR_dB, BER_bcjr(snr_idx));
    end
end