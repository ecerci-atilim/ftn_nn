function ftn_sim_gui()
% FTN_SIM_GUI - Interactive GUI for FTN Simulation with Impairments
%
% Usage:
%   ftn_sim_gui
%
% Features:
%   - Interactive parameter selection
%   - Toggle impairments on/off
%   - Real-time BER simulation
%   - Automatic plot generation
%
% Author: Emre Cerci
% Date: January 2026

    % Create figure
    hFig = figure('Name', 'FTN Simulation with Impairments', ...
                  'NumberTitle', 'off', ...
                  'Position', [100 100 900 700], ...
                  'MenuBar', 'none', ...
                  'Resize', 'on');

    % Initialize data structure
    data = struct();
    data.config = get_default_config();
    data.sim_running = false;

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
                  'Position', [0.25 0.93 0.5 0.05]);

        % ===== FTN Parameters Panel =====
        uipanel('Title', 'FTN Parameters', ...
                'FontSize', 10, ...
                'FontWeight', 'bold', ...
                'Units', 'normalized', ...
                'Position', [0.02 0.75 0.45 0.16]);

        uicontrol('Style', 'text', 'String', 'Tau (compression):', ...
                  'Units', 'normalized', 'Position', [0.04 0.85 0.15 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_tau = uicontrol('Style', 'edit', 'String', '0.7', ...
                                   'Units', 'normalized', 'Position', [0.20 0.85 0.08 0.03]);

        uicontrol('Style', 'text', 'String', 'Fractional L:', ...
                  'Units', 'normalized', 'Position', [0.04 0.81 0.15 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_lfrac = uicontrol('Style', 'edit', 'String', '2', ...
                                     'Units', 'normalized', 'Position', [0.20 0.81 0.08 0.03]);

        uicontrol('Style', 'text', 'String', 'N train:', ...
                  'Units', 'normalized', 'Position', [0.04 0.77 0.15 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_ntrain = uicontrol('Style', 'edit', 'String', '50000', ...
                                      'Units', 'normalized', 'Position', [0.20 0.77 0.08 0.03]);

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
                'Position', [0.50 0.38 0.47 0.53]);

        % CFO
        data.chk_cfo = uicontrol('Style', 'checkbox', 'String', 'Carrier Frequency Offset', ...
                                 'Units', 'normalized', 'Position', [0.52 0.85 0.20 0.03], ...
                                 'Callback', @cb_toggle_cfo);

        uicontrol('Style', 'text', 'String', 'CFO (Hz):', ...
                  'Units', 'normalized', 'Position', [0.54 0.81 0.08 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_cfo_hz = uicontrol('Style', 'edit', 'String', '100', ...
                                     'Units', 'normalized', 'Position', [0.63 0.82 0.08 0.03], ...
                                     'Enable', 'off');

        % Phase Noise
        data.chk_pn = uicontrol('Style', 'checkbox', 'String', 'Phase Noise', ...
                                'Units', 'normalized', 'Position', [0.52 0.76 0.20 0.03], ...
                                'Callback', @cb_toggle_pn);

        uicontrol('Style', 'text', 'String', 'PSD (dBc/Hz):', ...
                  'Units', 'normalized', 'Position', [0.54 0.72 0.10 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_pn_psd = uicontrol('Style', 'edit', 'String', '-80', ...
                                     'Units', 'normalized', 'Position', [0.65 0.73 0.08 0.03], ...
                                     'Enable', 'off');

        % RX IQ Imbalance
        data.chk_iq_rx = uicontrol('Style', 'checkbox', 'String', 'RX IQ Imbalance', ...
                                   'Units', 'normalized', 'Position', [0.52 0.67 0.20 0.03], ...
                                   'Callback', @cb_toggle_iq_rx);

        uicontrol('Style', 'text', 'String', 'Amp (dB):', ...
                  'Units', 'normalized', 'Position', [0.54 0.63 0.08 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_iq_rx_amp = uicontrol('Style', 'edit', 'String', '0.3', ...
                                        'Units', 'normalized', 'Position', [0.63 0.64 0.06 0.03], ...
                                        'Enable', 'off');

        uicontrol('Style', 'text', 'String', 'Phase (deg):', ...
                  'Units', 'normalized', 'Position', [0.70 0.63 0.10 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_iq_rx_phase = uicontrol('Style', 'edit', 'String', '3', ...
                                          'Units', 'normalized', 'Position', [0.81 0.64 0.06 0.03], ...
                                          'Enable', 'off');

        % ADC Quantization
        data.chk_adc = uicontrol('Style', 'checkbox', 'String', 'ADC Quantization', ...
                                 'Units', 'normalized', 'Position', [0.52 0.58 0.20 0.03], ...
                                 'Callback', @cb_toggle_adc);

        uicontrol('Style', 'text', 'String', 'Bits:', ...
                  'Units', 'normalized', 'Position', [0.54 0.54 0.08 0.03], ...
                  'HorizontalAlignment', 'left');
        data.edit_adc_bits = uicontrol('Style', 'edit', 'String', '8', ...
                                       'Units', 'normalized', 'Position', [0.63 0.55 0.06 0.03], ...
                                       'Enable', 'off');

        % ===== Quick Presets =====
        uicontrol('Style', 'text', 'String', 'Quick Presets:', ...
                  'FontWeight', 'bold', ...
                  'Units', 'normalized', 'Position', [0.52 0.47 0.15 0.03], ...
                  'HorizontalAlignment', 'left');

        uicontrol('Style', 'pushbutton', 'String', 'All OFF', ...
                  'Units', 'normalized', 'Position', [0.52 0.43 0.10 0.03], ...
                  'Callback', @cb_preset_off);

        uicontrol('Style', 'pushbutton', 'String', 'PA Only', ...
                  'Units', 'normalized', 'Position', [0.63 0.43 0.10 0.03], ...
                  'Callback', @cb_preset_pa);

        uicontrol('Style', 'pushbutton', 'String', 'All ON', ...
                  'Units', 'normalized', 'Position', [0.74 0.43 0.10 0.03], ...
                  'Callback', @cb_preset_all);

        % ===== Control Buttons =====
        uicontrol('Style', 'pushbutton', ...
                  'String', 'RUN SIMULATION', ...
                  'FontSize', 12, ...
                  'FontWeight', 'bold', ...
                  'ForegroundColor', [0 0.5 0], ...
                  'Units', 'normalized', ...
                  'Position', [0.35 0.30 0.30 0.06], ...
                  'Callback', @cb_run_simulation);

        % ===== Status Display =====
        data.txt_status = uicontrol('Style', 'text', ...
                                    'String', 'Ready', ...
                                    'FontSize', 10, ...
                                    'Units', 'normalized', ...
                                    'Position', [0.02 0.25 0.96 0.03], ...
                                    'HorizontalAlignment', 'left');

        % ===== Results Display =====
        data.txt_results = uicontrol('Style', 'text', ...
                                     'String', '', ...
                                     'FontSize', 9, ...
                                     'Units', 'normalized', ...
                                     'Position', [0.02 0.02 0.96 0.22], ...
                                     'HorizontalAlignment', 'left', ...
                                     'BackgroundColor', [0.95 0.95 0.95]);
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
        set(data.txt_status, 'String', 'Preset: All impairments OFF');
        guidata(hFig, data);
    end

    function cb_preset_pa(~, ~)
        data = guidata(hFig);
        cb_preset_off();
        set(data.chk_pa, 'Value', 1); cb_toggle_pa();
        set(data.txt_status, 'String', 'Preset: PA saturation only');
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
        set(data.txt_status, 'String', 'Preset: All impairments ON');
        guidata(hFig, data);
    end

    function cb_run_simulation(~, ~)
        data = guidata(hFig);

        if data.sim_running
            set(data.txt_status, 'String', 'Simulation already running...');
            return;
        end

        data.sim_running = true;
        guidata(hFig, data);

        set(data.txt_status, 'String', 'Reading configuration...');
        drawnow;

        % Read configuration from GUI
        config = read_config_from_gui();

        % Run simulation
        try
            set(data.txt_status, 'String', 'Running simulation... Please wait...');
            drawnow;

            run_ftn_simulation(config);

            set(data.txt_status, 'String', 'Simulation complete! Check figure and results.');

        catch ME
            set(data.txt_status, 'String', ['Error: ' ME.message]);
            rethrow(ME);
        end

        data.sim_running = false;
        guidata(hFig, data);
    end

    function config = read_config_from_gui()
        data = guidata(hFig);

        % FTN parameters
        config.tau = str2double(get(data.edit_tau, 'String'));
        config.l_frac = str2double(get(data.edit_lfrac, 'String'));
        config.n_train = str2double(get(data.edit_ntrain, 'String'));

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

    function run_ftn_simulation(config)
        % Actually run the simulation (simplified version)

        % Print configuration
        fprintf('\n========================================\n');
        fprintf('FTN SIMULATION (GUI Mode)\n');
        fprintf('========================================\n');
        fprintf('tau=%.2f, L=%d, N_train=%d\n', config.tau, config.l_frac, config.n_train);

        % This would call the actual simulation code
        % For now, just print that it would run
        fprintf('\nImpairments Configuration:\n');
        if config.pa_saturation.enabled
            fprintf('  PA: %s, IBO=%d dB\n', config.pa_saturation.model, config.pa_saturation.IBO_dB);
        end
        if config.iq_imbalance_tx.enabled
            fprintf('  TX IQ Imbalance: %.1f dB, %.1f deg\n', config.iq_imbalance_tx.amp_dB, config.iq_imbalance_tx.phase_deg);
        end

        fprintf('\nSimulation would run here...\n');
        fprintf('For full simulation, use ftn_sim_configurable.m\n');
        fprintf('========================================\n');

        % Update results display
        data = guidata(hFig);
        results_text = sprintf(['Configuration Summary:\n' ...
                               'tau = %.2f, L = %d\n' ...
                               'PA: %s\n' ...
                               'TX IQ: %s\n' ...
                               'RX IQ: %s\n\n' ...
                               'Use ftn_sim_configurable.m for full simulation'], ...
                               config.tau, config.l_frac, ...
                               mat2str(config.pa_saturation.enabled), ...
                               mat2str(config.iq_imbalance_tx.enabled), ...
                               mat2str(config.iq_imbalance_rx.enabled));

        set(data.txt_results, 'String', results_text);
        guidata(hFig, data);
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
