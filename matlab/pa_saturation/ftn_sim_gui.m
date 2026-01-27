function ftn_sim_gui()
% FTN_SIM_GUI - Interactive GUI for FTN Simulation with Impairments
%
% Features:
%   - Configure all 7 impairments via GUI
%   - Set SNR range (start, end, increment)
%   - Run full BER simulation
%   - Real-time progress display
%   - Automatic BER plot generation
%
% Author: Emre Cerci
% Date: January 2026

    % Create figure
    hFig = figure('Name', 'FTN Simulation with Impairments', ...
                  'NumberTitle', 'off', ...
                  'Position', [50 50 1100 750], ...
                  'MenuBar', 'none', ...
                  'Resize', 'on');

    % Initialize data structure
    data = struct();
    data.config = get_default_config();
    data.sim_running = false;
    data.results = struct();

    % Create UI components
    createUI();

    % Make data accessible to callbacks
    guidata(hFig, data);


    %% UI Creation
    function createUI()

        % ===== Title =====
        uicontrol('Style', 'text', ...
                  'String', 'FTN Simulation with Hardware Impairments', ...
                  'FontSize', 14, ...
                  'FontWeight', 'bold', ...
                  'Units', 'normalized', ...
                  'Position', [0.25 0.95 0.5 0.04]);

        % ===== FTN & Simulation Parameters Panel =====
        uipanel('Title', 'FTN & Simulation Parameters', ...
                'FontSize', 10, ...
                'FontWeight', 'bold', ...
                'Units', 'normalized', ...
                'Position', [0.02 0.75 0.45 0.18]);

        % Row 1: Tau and L_frac
        uicontrol('Style', 'text', 'String', 'Tau:', ...
                  'Units', 'normalized', 'Position', [0.04 0.87 0.08 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_tau = uicontrol('Style', 'edit', 'String', '0.7', ...
                                   'Units', 'normalized', 'Position', [0.13 0.87 0.07 0.03]);

        uicontrol('Style', 'text', 'String', 'Fractional L:', ...
                  'Units', 'normalized', 'Position', [0.22 0.87 0.12 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_lfrac = uicontrol('Style', 'edit', 'String', '2', ...
                                     'Units', 'normalized', 'Position', [0.35 0.87 0.07 0.03]);

        % Row 2: N_test and N_window
        uicontrol('Style', 'text', 'String', 'N test:', ...
                  'Units', 'normalized', 'Position', [0.04 0.83 0.08 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_ntest = uicontrol('Style', 'edit', 'String', '10000', ...
                                     'Units', 'normalized', 'Position', [0.13 0.83 0.07 0.03]);

        uicontrol('Style', 'text', 'String', 'N window:', ...
                  'Units', 'normalized', 'Position', [0.22 0.83 0.12 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_nwindow = uicontrol('Style', 'edit', 'String', '8', ...
                                       'Units', 'normalized', 'Position', [0.35 0.83 0.07 0.03]);

        % Row 3: SNR Range
        uicontrol('Style', 'text', 'String', 'SNR Start (dB):', ...
                  'Units', 'normalized', 'Position', [0.04 0.79 0.12 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_snr_start = uicontrol('Style', 'edit', 'String', '0', ...
                                         'Units', 'normalized', 'Position', [0.17 0.79 0.06 0.03]);

        uicontrol('Style', 'text', 'String', 'End:', ...
                  'Units', 'normalized', 'Position', [0.24 0.79 0.05 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_snr_end = uicontrol('Style', 'edit', 'String', '14', ...
                                       'Units', 'normalized', 'Position', [0.29 0.79 0.06 0.03]);

        uicontrol('Style', 'text', 'String', 'Step:', ...
                  'Units', 'normalized', 'Position', [0.36 0.79 0.05 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_snr_step = uicontrol('Style', 'edit', 'String', '2', ...
                                        'Units', 'normalized', 'Position', [0.41 0.79 0.05 0.03]);

        % ===== Transmitter Impairments Panel =====
        uipanel('Title', 'Transmitter Impairments', ...
                'FontSize', 10, ...
                'FontWeight', 'bold', ...
                'Units', 'normalized', ...
                'Position', [0.02 0.38 0.45 0.35]);

        % PA Saturation
        data.chk_pa = uicontrol('Style', 'checkbox', 'String', 'PA Saturation', ...
                                'Units', 'normalized', 'Position', [0.04 0.67 0.15 0.03], ...
                                'Callback', @cb_toggle_pa);

        uicontrol('Style', 'text', 'String', 'Model:', ...
                  'Units', 'normalized', 'Position', [0.06 0.63 0.08 0.03], ...
                  'HorizontalAlignment', 'left');
        data.popup_pa = uicontrol('Style', 'popupmenu', ...
                                  'String', {'rapp', 'saleh', 'soft_limiter'}, ...
                                  'Units', 'normalized', 'Position', [0.15 0.64 0.12 0.03], ...
                                  'Enable', 'off');

        uicontrol('Style', 'text', 'String', 'IBO (dB):', ...
                  'Units', 'normalized', 'Position', [0.06 0.59 0.08 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_ibo = uicontrol('Style', 'edit', 'String', '3', ...
                                   'Units', 'normalized', 'Position', [0.15 0.60 0.06 0.03], ...
                                   'Enable', 'off');

        data.chk_pa_memory = uicontrol('Style', 'checkbox', 'String', 'Memory effects', ...
                                       'Units', 'normalized', 'Position', [0.23 0.60 0.12 0.03], ...
                                       'Enable', 'off');

        % TX IQ Imbalance
        data.chk_iq_tx = uicontrol('Style', 'checkbox', 'String', 'TX IQ Imbalance', ...
                                   'Units', 'normalized', 'Position', [0.04 0.54 0.15 0.03], ...
                                   'Callback', @cb_toggle_iq_tx);

        uicontrol('Style', 'text', 'String', 'Amp (dB):', ...
                  'Units', 'normalized', 'Position', [0.06 0.50 0.08 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_iq_tx_amp = uicontrol('Style', 'edit', 'String', '0.5', ...
                                        'Units', 'normalized', 'Position', [0.15 0.51 0.06 0.03], ...
                                        'Enable', 'off');

        uicontrol('Style', 'text', 'String', 'Phase (deg):', ...
                  'Units', 'normalized', 'Position', [0.22 0.50 0.08 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_iq_tx_phase = uicontrol('Style', 'edit', 'String', '5', ...
                                          'Units', 'normalized', 'Position', [0.31 0.51 0.06 0.03], ...
                                          'Enable', 'off');

        % DAC Quantization
        data.chk_dac = uicontrol('Style', 'checkbox', 'String', 'DAC Quantization', ...
                                 'Units', 'normalized', 'Position', [0.04 0.45 0.15 0.03], ...
                                 'Callback', @cb_toggle_dac);

        uicontrol('Style', 'text', 'String', 'Bits:', ...
                  'Units', 'normalized', 'Position', [0.06 0.41 0.08 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_dac_bits = uicontrol('Style', 'edit', 'String', '8', ...
                                       'Units', 'normalized', 'Position', [0.15 0.42 0.06 0.03], ...
                                       'Enable', 'off');

        % ===== Receiver Impairments Panel =====
        uipanel('Title', 'Receiver Impairments', ...
                'FontSize', 10, ...
                'FontWeight', 'bold', ...
                'Units', 'normalized', ...
                'Position', [0.50 0.38 0.47 0.55]);

        % CFO
        data.chk_cfo = uicontrol('Style', 'checkbox', 'String', 'Carrier Frequency Offset', ...
                                 'Units', 'normalized', 'Position', [0.52 0.87 0.20 0.03], ...
                                 'Callback', @cb_toggle_cfo);

        uicontrol('Style', 'text', 'String', 'CFO (Hz):', ...
                  'Units', 'normalized', 'Position', [0.54 0.83 0.08 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_cfo_hz = uicontrol('Style', 'edit', 'String', '100', ...
                                     'Units', 'normalized', 'Position', [0.63 0.84 0.08 0.03], ...
                                     'Enable', 'off');

        % Phase Noise
        data.chk_pn = uicontrol('Style', 'checkbox', 'String', 'Phase Noise', ...
                                'Units', 'normalized', 'Position', [0.52 0.78 0.20 0.03], ...
                                'Callback', @cb_toggle_pn);

        uicontrol('Style', 'text', 'String', 'PSD (dBc/Hz):', ...
                  'Units', 'normalized', 'Position', [0.54 0.74 0.10 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_pn_psd = uicontrol('Style', 'edit', 'String', '-80', ...
                                     'Units', 'normalized', 'Position', [0.65 0.75 0.08 0.03], ...
                                     'Enable', 'off');

        % RX IQ Imbalance
        data.chk_iq_rx = uicontrol('Style', 'checkbox', 'String', 'RX IQ Imbalance', ...
                                   'Units', 'normalized', 'Position', [0.52 0.69 0.20 0.03], ...
                                   'Callback', @cb_toggle_iq_rx);

        uicontrol('Style', 'text', 'String', 'Amp (dB):', ...
                  'Units', 'normalized', 'Position', [0.54 0.65 0.08 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_iq_rx_amp = uicontrol('Style', 'edit', 'String', '0.3', ...
                                        'Units', 'normalized', 'Position', [0.63 0.66 0.06 0.03], ...
                                        'Enable', 'off');

        uicontrol('Style', 'text', 'String', 'Phase (deg):', ...
                  'Units', 'normalized', 'Position', [0.70 0.65 0.10 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_iq_rx_phase = uicontrol('Style', 'edit', 'String', '3', ...
                                          'Units', 'normalized', 'Position', [0.81 0.66 0.06 0.03], ...
                                          'Enable', 'off');

        % ADC Quantization
        data.chk_adc = uicontrol('Style', 'checkbox', 'String', 'ADC Quantization', ...
                                 'Units', 'normalized', 'Position', [0.52 0.60 0.20 0.03], ...
                                 'Callback', @cb_toggle_adc);

        uicontrol('Style', 'text', 'String', 'Bits:', ...
                  'Units', 'normalized', 'Position', [0.54 0.56 0.08 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_adc_bits = uicontrol('Style', 'edit', 'String', '8', ...
                                       'Units', 'normalized', 'Position', [0.63 0.57 0.06 0.03], ...
                                       'Enable', 'off');

        % ===== Quick Presets =====
        uicontrol('Style', 'text', 'String', 'Quick Presets:', ...
                  'FontWeight', 'bold', ...
                  'Units', 'normalized', 'Position', [0.52 0.50 0.15 0.03], ...
                  'HorizontalAlignment', 'left');

        uicontrol('Style', 'pushbutton', 'String', 'All OFF', ...
                  'Units', 'normalized', 'Position', [0.52 0.46 0.10 0.03], ...
                  'Callback', @cb_preset_off);

        uicontrol('Style', 'pushbutton', 'String', 'PA Only', ...
                  'Units', 'normalized', 'Position', [0.63 0.46 0.10 0.03], ...
                  'Callback', @cb_preset_pa);

        uicontrol('Style', 'pushbutton', 'String', 'All ON', ...
                  'Units', 'normalized', 'Position', [0.74 0.46 0.10 0.03], ...
                  'Callback', @cb_preset_all);

        % ===== Control Buttons =====
        uicontrol('Style', 'pushbutton', ...
                  'String', 'RUN SIMULATION', ...
                  'FontSize', 12, ...
                  'FontWeight', 'bold', ...
                  'ForegroundColor', [0 0.5 0], ...
                  'Units', 'normalized', ...
                  'Position', [0.30 0.29 0.40 0.06], ...
                  'Callback', @cb_run_simulation);

        % ===== Status Display =====
        data.txt_status = uicontrol('Style', 'text', ...
                                    'String', 'Ready to simulate', ...
                                    'FontSize', 10, ...
                                    'FontWeight', 'bold', ...
                                    'ForegroundColor', [0 0.5 0], ...
                                    'Units', 'normalized', ...
                                    'Position', [0.02 0.24 0.96 0.03], ...
                                    'HorizontalAlignment', 'left');

        % ===== Results Display (Table) =====
        uicontrol('Style', 'text', 'String', 'Results:', ...
                  'FontWeight', 'bold', ...
                  'Units', 'normalized', 'Position', [0.02 0.20 0.10 0.03], ...
                  'HorizontalAlignment', 'left');

        data.txt_results = uicontrol('Style', 'listbox', ...
                                     'String', {''}, ...
                                     'FontName', 'FixedWidth', ...
                                     'FontSize', 9, ...
                                     'Units', 'normalized', ...
                                     'Position', [0.02 0.02 0.96 0.17], ...
                                     'HorizontalAlignment', 'left');
    end


    %% Callbacks

    function cb_toggle_pa(~, ~)
        data = guidata(hFig);
        state = get(data.chk_pa, 'Value');
        if state
            set(data.popup_pa, 'Enable', 'on');
            set(data.edit_ibo, 'Enable', 'on');
            set(data.chk_pa_memory, 'Enable', 'on');
        else
            set(data.popup_pa, 'Enable', 'off');
            set(data.edit_ibo, 'Enable', 'off');
            set(data.chk_pa_memory, 'Enable', 'off');
        end
        guidata(hFig, data);
    end

    function cb_toggle_iq_tx(~, ~)
        data = guidata(hFig);
        state = get(data.chk_iq_tx, 'Value');
        en = {'off', 'on'};
        set(data.edit_iq_tx_amp, 'Enable', en{state+1});
        set(data.edit_iq_tx_phase, 'Enable', en{state+1});
        guidata(hFig, data);
    end

    function cb_toggle_dac(~, ~)
        data = guidata(hFig);
        state = get(data.chk_dac, 'Value');
        en = {'off', 'on'};
        set(data.edit_dac_bits, 'Enable', en{state+1});
        guidata(hFig, data);
    end

    function cb_toggle_cfo(~, ~)
        data = guidata(hFig);
        state = get(data.chk_cfo, 'Value');
        en = {'off', 'on'};
        set(data.edit_cfo_hz, 'Enable', en{state+1});
        guidata(hFig, data);
    end

    function cb_toggle_pn(~, ~)
        data = guidata(hFig);
        state = get(data.chk_pn, 'Value');
        en = {'off', 'on'};
        set(data.edit_pn_psd, 'Enable', en{state+1});
        guidata(hFig, data);
    end

    function cb_toggle_iq_rx(~, ~)
        data = guidata(hFig);
        state = get(data.chk_iq_rx, 'Value');
        en = {'off', 'on'};
        set(data.edit_iq_rx_amp, 'Enable', en{state+1});
        set(data.edit_iq_rx_phase, 'Enable', en{state+1});
        guidata(hFig, data);
    end

    function cb_toggle_adc(~, ~)
        data = guidata(hFig);
        state = get(data.chk_adc, 'Value');
        en = {'off', 'on'};
        set(data.edit_adc_bits, 'Enable', en{state+1});
        guidata(hFig, data);
    end

    function cb_preset_off(~, ~)
        data = guidata(hFig);
        set(data.chk_pa, 'Value', 0); cb_toggle_pa();
        set(data.chk_iq_tx, 'Value', 0); cb_toggle_iq_tx();
        set(data.chk_dac, 'Value', 0); cb_toggle_dac();
        set(data.chk_cfo, 'Value', 0); cb_toggle_cfo();
        set(data.chk_pn, 'Value', 0); cb_toggle_pn();
        set(data.chk_iq_rx, 'Value', 0); cb_toggle_iq_rx();
        set(data.chk_adc, 'Value', 0); cb_toggle_adc();
        set(data.txt_status, 'String', 'Preset: All impairments OFF', 'ForegroundColor', [0 0.5 0]);
        guidata(hFig, data);
    end

    function cb_preset_pa(~, ~)
        data = guidata(hFig);
        cb_preset_off();
        set(data.chk_pa, 'Value', 1); cb_toggle_pa();
        set(data.txt_status, 'String', 'Preset: PA saturation only', 'ForegroundColor', [0 0.5 0]);
        guidata(hFig, data);
    end

    function cb_preset_all(~, ~)
        data = guidata(hFig);
        set(data.chk_pa, 'Value', 1); cb_toggle_pa();
        set(data.chk_iq_tx, 'Value', 1); cb_toggle_iq_tx();
        set(data.chk_dac, 'Value', 1); cb_toggle_dac();
        set(data.chk_cfo, 'Value', 1); cb_toggle_cfo();
        set(data.chk_pn, 'Value', 1); cb_toggle_pn();
        set(data.chk_iq_rx, 'Value', 1); cb_toggle_iq_rx();
        set(data.chk_adc, 'Value', 1); cb_toggle_adc();
        set(data.txt_status, 'String', 'Preset: All impairments ON', 'ForegroundColor', [0 0.5 0]);
        guidata(hFig, data);
    end

    function cb_run_simulation(~, ~)
        data = guidata(hFig);

        if data.sim_running
            set(data.txt_status, 'String', 'Simulation already running...', 'ForegroundColor', [0.8 0.5 0]);
            return;
        end

        data.sim_running = true;
        guidata(hFig, data);

        set(data.txt_status, 'String', 'Reading configuration...', 'ForegroundColor', [0 0 0.8]);
        drawnow;

        % Read configuration from GUI
        config = read_config_from_gui();

        % Run simulation
        try
            set(data.txt_status, 'String', 'Running simulation... Please wait...', 'ForegroundColor', [0.8 0 0]);
            drawnow;

            [snr_range, ber_results] = run_ftn_simulation(config, data);

            % Store results
            data.results.snr = snr_range;
            data.results.ber = ber_results;
            guidata(hFig, data);

            % Display results
            display_results(snr_range, ber_results);

            % Plot results
            plot_results(snr_range, ber_results, config);

            set(data.txt_status, 'String', 'Simulation complete! Check Results and Figure.', 'ForegroundColor', [0 0.5 0]);

        catch ME
            set(data.txt_status, 'String', ['Error: ' ME.message], 'ForegroundColor', [0.8 0 0]);
            fprintf('Error: %s\n', ME.message);
            fprintf('Stack:\n');
            for k = 1:length(ME.stack)
                fprintf('  %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
            end
        end

        data.sim_running = false;
        guidata(hFig, data);
    end

    function config = read_config_from_gui()
        data = guidata(hFig);

        % FTN parameters
        config.tau = str2double(get(data.edit_tau, 'String'));
        config.l_frac = str2double(get(data.edit_lfrac, 'String'));
        config.n_test = str2double(get(data.edit_ntest, 'String'));
        config.n_window = str2double(get(data.edit_nwindow, 'String'));

        % SNR range
        snr_start = str2double(get(data.edit_snr_start, 'String'));
        snr_end = str2double(get(data.edit_snr_end, 'String'));
        snr_step = str2double(get(data.edit_snr_step, 'String'));
        config.snr_range = snr_start:snr_step:snr_end;

        % PA
        config.pa_saturation.enabled = logical(get(data.chk_pa, 'Value'));
        if config.pa_saturation.enabled
            models = get(data.popup_pa, 'String');
            idx = get(data.popup_pa, 'Value');
            config.pa_saturation.model = models{idx};
            config.pa_saturation.IBO_dB = str2double(get(data.edit_ibo, 'String'));
            config.pa_saturation.memory_effects = logical(get(data.chk_pa_memory, 'Value'));

            % Calculate PA params
            IBO_lin = 10^(config.pa_saturation.IBO_dB/10);
            config.pa_saturation.G = 1;
            config.pa_saturation.Asat = sqrt(IBO_lin);
            config.pa_saturation.p = 2;
        end

        % TX IQ
        config.iq_imbalance_tx.enabled = logical(get(data.chk_iq_tx, 'Value'));
        config.iq_imbalance_tx.amp_dB = str2double(get(data.edit_iq_tx_amp, 'String'));
        config.iq_imbalance_tx.phase_deg = str2double(get(data.edit_iq_tx_phase, 'String'));

        % DAC
        config.dac_quantization.enabled = logical(get(data.chk_dac, 'Value'));
        config.dac_quantization.n_bits = str2double(get(data.edit_dac_bits, 'String'));
        config.dac_quantization.full_scale = 1.0;

        % CFO
        config.cfo.enabled = logical(get(data.chk_cfo, 'Value'));
        config.cfo.cfo_hz = str2double(get(data.edit_cfo_hz, 'String'));
        config.cfo.fs = 1e5;

        % Phase Noise
        config.phase_noise.enabled = logical(get(data.chk_pn, 'Value'));
        config.phase_noise.psd_dBc_Hz = str2double(get(data.edit_pn_psd, 'String'));
        config.phase_noise.fs = 1e5;

        % RX IQ
        config.iq_imbalance_rx.enabled = logical(get(data.chk_iq_rx, 'Value'));
        config.iq_imbalance_rx.amp_dB = str2double(get(data.edit_iq_rx_amp, 'String'));
        config.iq_imbalance_rx.phase_deg = str2double(get(data.edit_iq_rx_phase, 'String'));

        % ADC
        config.adc_quantization.enabled = logical(get(data.chk_adc, 'Value'));
        config.adc_quantization.n_bits = str2double(get(data.edit_adc_bits, 'String'));
        config.adc_quantization.full_scale = 1.0;
    end

    function [snr_range, ber_results] = run_ftn_simulation(config, data)
        % Run the actual FTN BER simulation

        % Fixed FTN parameters (same as reference implementation)
        beta = 0.3;
        sps = 10;
        span = 6;

        % Generate SRRC pulse using rcosdesign (same as reference)
        h_srrc = rcosdesign(beta, span, sps, 'sqrt');
        h_srrc = h_srrc / norm(h_srrc);
        delay = span * sps;

        % Simulation parameters
        step = round(config.tau * sps);
        snr_range = config.snr_range;
        ber_results = zeros(size(snr_range));

        % Run BER for each SNR
        for idx = 1:length(snr_range)
            snr_db = snr_range(idx);

            % Update status
            set(data.txt_status, 'String', sprintf('Simulating SNR = %d dB (%d/%d)...', ...
                snr_db, idx, length(snr_range)), 'ForegroundColor', [0.8 0 0]);
            drawnow;

            % Generate data with impairments
            [rx_mf, bits] = generate_ftn_data_with_impairments(config.n_test, snr_db, ...
                config.tau, sps, h_srrc, delay, config);

            % Fractional detection
            ber_results(idx) = detect_fractional(rx_mf, bits, step, delay, config.n_window, config.l_frac);
        end
    end

    function [rx_mf, bits] = generate_ftn_data_with_impairments(n_symbols, snr_db, ...
            tau, sps, h_srrc, delay, config)

        % Generate bits and symbols
        bits = randi([0 1], n_symbols, 1);
        symbols = 2*bits - 1;  % BPSK

        % FTN upsampling - OPTIMIZED: Use exact size needed
        step = round(tau * sps);
        tx_up = zeros(1 + (n_symbols-1)*step, 1);
        tx_up(1:step:end) = symbols;

        % Pulse shaping (use 'full' to preserve all energy)
        tx_shaped = conv(tx_up, h_srrc, 'full');

        % Apply TX impairments (inline implementation)
        tx_impaired = tx_shaped;
        
        % TX IQ Imbalance
        if config.iq_imbalance_tx.enabled
            tx_impaired = apply_iq_imbalance(tx_impaired, config.iq_imbalance_tx);
        end
        
        % DAC Quantization
        if config.dac_quantization.enabled
            tx_impaired = apply_quantization(tx_impaired, config.dac_quantization);
        end
        
        % PA Saturation
        if config.pa_saturation.enabled
            tx_impaired = pa_models(tx_impaired, config.pa_saturation.model, config.pa_saturation);
            % FIXED: Force real for BPSK (PA may output complex)
            tx_impaired = real(tx_impaired);
        end

        % FIXED: Correct SNR calculation based on actual signal power
        signal_power = mean(real(tx_impaired).^2);
        EbN0 = 10^(snr_db/10);
        noise_power = signal_power / (2 * EbN0);
        noise = sqrt(noise_power) * randn(size(tx_impaired));  % Real noise for BPSK
        rx_noisy = tx_impaired + noise;

        % Apply RX impairments (inline implementation)
        rx_impaired = rx_noisy;
        
        % CFO
        if config.cfo.enabled
            n = (0:length(rx_impaired)-1)';
            phase_shift = exp(1j * 2 * pi * config.cfo.cfo_hz * n / config.cfo.fs);
            rx_impaired = rx_impaired .* phase_shift;
        end
        
        % Phase Noise
        if config.phase_noise.enabled
            f_offset = 10e3;  % Reference offset frequency
            L_linear = 10^(config.phase_noise.psd_dBc_Hz / 10);
            linewidth_hz = pi * (f_offset^2) * L_linear;
            sigma_phi = sqrt(2 * pi * linewidth_hz / config.phase_noise.fs);
            phase_noise = cumsum(randn(size(rx_impaired)) * sigma_phi);
            rx_impaired = rx_impaired .* exp(1j * phase_noise);
        end
        
        % RX IQ Imbalance
        if config.iq_imbalance_rx.enabled
            rx_impaired = apply_iq_imbalance(rx_impaired, config.iq_imbalance_rx);
        end
        
        % ADC Quantization
        if config.adc_quantization.enabled
            rx_impaired = apply_quantization(rx_impaired, config.adc_quantization);
        end

        % Matched filter
        rx_mf = conv(rx_impaired, h_srrc, 'full');
        rx_mf = rx_mf / std(rx_mf);
    end
    
    function y = apply_iq_imbalance(x, cfg)
        % Apply IQ imbalance: y = (1+α)·I + j·(1-α)·Q·exp(jφ)
        alpha = (10^(cfg.amp_dB/20) - 1) / 2;
        phi = deg2rad(cfg.phase_deg);
        I = real(x);
        Q = imag(x);
        I_imb = (1 + alpha) * I;
        Q_imb = (1 - alpha) * Q;
        y = I_imb + 1j * Q_imb * exp(1j * phi);
    end
    
    function y = apply_quantization(x, cfg)
        % Apply ADC/DAC quantization
        n_bits = cfg.n_bits;
        full_scale = cfg.full_scale;
        n_levels = 2^n_bits;
        step_size = 2 * full_scale / n_levels;
        I = real(x);
        Q = imag(x);
        I_quant = round(I / step_size) * step_size;
        Q_quant = round(Q / step_size) * step_size;
        I_quant = max(min(I_quant, full_scale), -full_scale);
        Q_quant = max(min(Q_quant, full_scale), -full_scale);
        y = I_quant + 1j * Q_quant;
    end

    function ber = detect_fractional(rx_mf, bits, step, delay, n_window, l_frac)
        % Fractional sampling detection

        frac_step = max(1, round(step / l_frac));
        n_symbols = length(bits);

        bits_hat = zeros(n_symbols, 1);
        valid_count = 0;

        for k = 1:n_symbols
            % Symbol timing (matches reference: delay + 1 + (k-1)*step)
            center_idx = delay + 1 + (k-1) * step;

            % Extract fractional samples
            frac_samples = [];
            for offset = -n_window:n_window
                idx = center_idx + offset * frac_step;
                if idx > 0 && idx <= length(rx_mf)
                    frac_samples = [frac_samples; real(rx_mf(idx))];
                end
            end

            % Decision
            if ~isempty(frac_samples)
                decision_stat = mean(frac_samples);
                bits_hat(k) = decision_stat > 0;
                valid_count = valid_count + 1;
            end
        end

        % Calculate BER
        if valid_count > 0
            ber = sum(bits_hat ~= bits) / n_symbols;
        else
            ber = 0.5;  % Worst case
        end
    end

    function display_results(snr_range, ber_results)
        data = guidata(hFig);

        % Format results as table
        results_str = cell(length(snr_range) + 2, 1);
        results_str{1} = 'SNR (dB) | BER';
        results_str{2} = '---------|---------------';

        for i = 1:length(snr_range)
            results_str{i+2} = sprintf('  %2d     | %.2e', snr_range(i), ber_results(i));
        end

        set(data.txt_results, 'String', results_str);
        guidata(hFig, data);
    end

    function plot_results(snr_range, ber_results, config)
        % Create new figure for BER plot
        figure('Name', 'BER Results', 'NumberTitle', 'off', 'Position', [150 150 800 600]);

        semilogy(snr_range, ber_results, 'bo-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
        grid on;
        xlabel('SNR (dB)', 'FontSize', 12);
        ylabel('Bit Error Rate', 'FontSize', 12);

        % Title with configuration
        title_str = sprintf('FTN BER Performance (\\tau=%.2f, L=%d)', config.tau, config.l_frac);
        if config.pa_saturation.enabled
            title_str = [title_str sprintf('\nPA: %s (IBO=%ddB)', config.pa_saturation.model, config.pa_saturation.IBO_dB)];
        end
        title(title_str, 'FontSize', 12, 'FontWeight', 'bold');

        ylim([1e-5 1]);
        xlim([min(snr_range)-1 max(snr_range)+1]);
    end

end


%% Helper Function
function config = get_default_config()
    config.pa_saturation.enabled = false;
    config.iq_imbalance_tx.enabled = false;
    config.dac_quantization.enabled = false;
    config.cfo.enabled = false;
    config.phase_noise.enabled = false;
    config.iq_imbalance_rx.enabled = false;
    config.adc_quantization.enabled = false;
end
