function ftn_nn_gui()
% FTN_NN_GUI - Interactive GUI for FTN Neural Network Detection with Impairments
%
% This GUI allows you to:
%   1. Configure hardware impairments (PA saturation, IQ imbalance, phase noise, CFO)
%   2. Select detection approaches (Neighbor, Fractional, Hybrid, Structured CNN)
%   3. Train neural networks
%   4. Test and compare BER performance
%
% Author: Emre Cerci
% Date: January 2026

    % Create figure
    fig = uifigure('Name', 'FTN NN Detection with Impairments', ...
                   'Position', [100 100 900 700]);

    % Create tab group
    tabgroup = uitabgroup(fig, 'Position', [10 10 880 680]);
    tab1 = uitab(tabgroup, 'Title', 'Configuration');
    tab2 = uitab(tabgroup, 'Title', 'Training & Testing');
    tab3 = uitab(tabgroup, 'Title', 'Results');

    %% Tab 1: Configuration
    % FTN Parameters Panel
    panel_ftn = uipanel(tab1, 'Title', 'FTN Parameters', ...
                        'Position', [20 500 400 160]);

    uilabel(panel_ftn, 'Position', [10 110 150 22], 'Text', 'Tau (compression):');
    tau_field = uieditfield(panel_ftn, 'numeric', 'Position', [170 110 80 22], ...
                            'Value', 0.7, 'Limits', [0.5 0.99]);

    uilabel(panel_ftn, 'Position', [10 80 150 22], 'Text', 'Roll-off (beta):');
    beta_field = uieditfield(panel_ftn, 'numeric', 'Position', [170 80 80 22], ...
                             'Value', 0.3, 'Limits', [0.1 0.9]);

    uilabel(panel_ftn, 'Position', [10 50 150 22], 'Text', 'Samples/symbol (sps):');
    sps_field = uieditfield(panel_ftn, 'numeric', 'Position', [170 50 80 22], ...
                            'Value', 10, 'Limits', [4 20], 'RoundFractionalValues', 'on');

    uilabel(panel_ftn, 'Position', [10 20 150 22], 'Text', 'Pulse span:');
    span_field = uieditfield(panel_ftn, 'numeric', 'Position', [170 20 80 22], ...
                             'Value', 6, 'Limits', [4 10], 'RoundFractionalValues', 'on');

    % Hardware Impairments Panel
    panel_hw = uipanel(tab1, 'Title', 'Hardware Impairments', ...
                       'Position', [20 220 400 270]);

    % PA Saturation
    pa_check = uicheckbox(panel_hw, 'Position', [10 230 150 22], ...
                          'Text', 'PA Saturation', 'Value', true);
    uilabel(panel_hw, 'Position', [30 200 100 22], 'Text', 'Model:');
    pa_model_drop = uidropdown(panel_hw, 'Position', [140 200 100 22], ...
                               'Items', {'rapp', 'saleh', 'soft_limiter'}, ...
                               'Value', 'rapp');
    uilabel(panel_hw, 'Position', [30 170 100 22], 'Text', 'IBO (dB):');
    pa_ibo_field = uieditfield(panel_hw, 'numeric', 'Position', [140 170 100 22], ...
                               'Value', 3, 'Limits', [0 10]);

    % IQ Imbalance (TX)
    iq_tx_check = uicheckbox(panel_hw, 'Position', [10 130 150 22], ...
                             'Text', 'TX IQ Imbalance', 'Value', false);
    uilabel(panel_hw, 'Position', [30 100 100 22], 'Text', 'Amplitude:');
    iq_tx_amp_field = uieditfield(panel_hw, 'numeric', 'Position', [140 100 100 22], ...
                                  'Value', 0.1, 'Limits', [0 0.5]);
    uilabel(panel_hw, 'Position', [30 70 100 22], 'Text', 'Phase (deg):');
    iq_tx_phase_field = uieditfield(panel_hw, 'numeric', 'Position', [140 70 100 22], ...
                                    'Value', 5, 'Limits', [0 30]);

    % Phase Noise
    pn_check = uicheckbox(panel_hw, 'Position', [10 30 150 22], ...
                          'Text', 'Phase Noise', 'Value', false);
    uilabel(panel_hw, 'Position', [30 5 100 22], 'Text', 'Variance:');
    pn_var_field = uieditfield(panel_hw, 'numeric', 'Position', [140 5 100 22], ...
                               'Value', 0.01, 'Limits', [0.001 0.1]);

    % CFO
    cfo_check = uicheckbox(panel_hw, 'Position', [250 230 130 22], ...
                           'Text', 'CFO', 'Value', false);
    uilabel(panel_hw, 'Position', [270 200 100 22], 'Text', 'Offset (Hz):');
    cfo_hz_field = uieditfield(panel_hw, 'numeric', 'Position', [270 170 100 22], ...
                               'Value', 100, 'Limits', [0 1000]);

    % Simulation Parameters Panel
    panel_sim = uipanel(tab1, 'Title', 'Simulation Parameters', ...
                        'Position', [20 20 400 190]);

    uilabel(panel_sim, 'Position', [10 140 150 22], 'Text', 'Training SNR (dB):');
    snr_train_field = uieditfield(panel_sim, 'numeric', 'Position', [170 140 80 22], ...
                                  'Value', 10, 'Limits', [0 20]);

    uilabel(panel_sim, 'Position', [10 110 150 22], 'Text', 'SNR Start (dB):');
    snr_start_field = uieditfield(panel_sim, 'numeric', 'Position', [170 110 80 22], ...
                                  'Value', 0, 'Limits', [-10 20]);

    uilabel(panel_sim, 'Position', [10 80 150 22], 'Text', 'SNR End (dB):');
    snr_end_field = uieditfield(panel_sim, 'numeric', 'Position', [170 80 80 22], ...
                                'Value', 14, 'Limits', [0 30]);

    uilabel(panel_sim, 'Position', [10 50 150 22], 'Text', 'SNR Step (dB):');
    snr_step_field = uieditfield(panel_sim, 'numeric', 'Position', [170 50 80 22], ...
                                 'Value', 2, 'Limits', [1 5]);

    uilabel(panel_sim, 'Position', [10 20 150 22], 'Text', 'Training symbols:');
    n_train_field = uieditfield(panel_sim, 'numeric', 'Position', [170 20 80 22], ...
                                'Value', 50000, 'Limits', [10000 200000]);

    % Detection Approaches Panel
    panel_det = uipanel(tab1, 'Title', 'Detection Approaches', ...
                        'Position', [440 220 420 440]);

    uilabel(panel_det, 'Position', [10 400 380 22], ...
            'Text', 'Select approaches to train and test:', 'FontWeight', 'bold');

    neighbor_check = uicheckbox(panel_det, 'Position', [20 360 200 22], ...
                                'Text', 'Neighbor (symbol-rate)', 'Value', true);
    uilabel(panel_det, 'Position', [40 330 350 30], ...
            'Text', 'Samples at 7 neighboring symbol instants (-3T to +3T)');

    fractional_check = uicheckbox(panel_det, 'Position', [20 280 200 22], ...
                                  'Text', 'Fractional', 'Value', true);
    uilabel(panel_det, 'Position', [40 250 350 30], ...
            'Text', 'Inter-symbol samples, avoiding exact symbol instants');

    hybrid_check = uicheckbox(panel_det, 'Position', [20 200 200 22], ...
                              'Text', 'Hybrid', 'Value', true);
    uilabel(panel_det, 'Position', [40 170 350 30], ...
            'Text', 'Combines symbol instants with fractional samples');

    structured_check = uicheckbox(panel_det, 'Position', [20 120 200 22], ...
                                  'Text', 'Structured CNN', 'Value', true);
    uilabel(panel_det, 'Position', [40 70 350 50], ...
            'Text', ['7x7 matrix structure: 7 neighbor symbols × ' ...
                     '7 samples. Uses CNN to process ISI spatially.']);

    % Buttons
    select_all_btn = uibutton(panel_det, 'Position', [20 20 180 30], ...
                              'Text', 'Select All', ...
                              'ButtonPushedFcn', @(btn,event) selectAllDetectors(true));

    deselect_all_btn = uibutton(panel_det, 'Position', [220 20 180 30], ...
                                'Text', 'Deselect All', ...
                                'ButtonPushedFcn', @(btn,event) selectAllDetectors(false));

    %% Tab 2: Training & Testing
    % Status and progress
    status_label = uilabel(tab2, 'Position', [20 640 840 30], ...
                          'Text', 'Ready to start simulation', ...
                          'FontSize', 14, 'FontWeight', 'bold');

    progress_gauge = uigauge(tab2, 'linear', 'Position', [20 590 840 40], ...
                            'Limits', [0 100]);

    % Output area
    output_area = uitextarea(tab2, 'Position', [20 100 840 470], ...
                            'Editable', 'off', 'FontName', 'Courier');

    % Control buttons
    start_btn = uibutton(tab2, 'Position', [20 40 200 40], ...
                        'Text', 'Start Simulation', ...
                        'FontSize', 14, 'FontWeight', 'bold', ...
                        'ButtonPushedFcn', @(btn,event) runSimulation());

    stop_btn = uibutton(tab2, 'Position', [240 40 200 40], ...
                       'Text', 'Stop', 'Enable', 'off', ...
                       'FontSize', 14, ...
                       'ButtonPushedFcn', @(btn,event) stopSimulation());

    save_btn = uibutton(tab2, 'Position', [460 40 200 40], ...
                       'Text', 'Save Results', 'Enable', 'off', ...
                       'FontSize', 14, ...
                       'ButtonPushedFcn', @(btn,event) saveResults());

    %% Tab 3: Results
    results_panel = uipanel(tab3, 'Position', [10 10 860 650]);
    results_axes = uiaxes(results_panel, 'Position', [30 30 800 580]);
    title(results_axes, 'BER Performance (Run simulation to see results)');
    grid(results_axes, 'on');

    % Shared data
    data = struct();
    data.stop_flag = false;
    data.results = [];

    %% Helper Functions
    function selectAllDetectors(select_all)
        neighbor_check.Value = select_all;
        fractional_check.Value = select_all;
        hybrid_check.Value = select_all;
        structured_check.Value = select_all;
    end

    function runSimulation()
        % Collect configuration
        config = collectConfig();

        % Check at least one detector is selected
        if ~any([neighbor_check.Value, fractional_check.Value, ...
                 hybrid_check.Value, structured_check.Value])
            uialert(fig, 'Please select at least one detection approach.', 'Error');
            return;
        end

        % Reset stop flag
        data.stop_flag = false;

        % Update UI
        start_btn.Enable = 'off';
        stop_btn.Enable = 'on';
        save_btn.Enable = 'off';
        status_label.Text = 'Initializing...';
        output_area.Value = '';
        progress_gauge.Value = 0;

        % Run simulation
        try
            data.results = runFTNSimulation(config);
            status_label.Text = 'Simulation Complete!';
            save_btn.Enable = 'on';
            plotResults();
        catch ME
            status_label.Text = 'Error occurred';
            uialert(fig, ME.message, 'Simulation Error');
        end

        % Update UI
        start_btn.Enable = 'on';
        stop_btn.Enable = 'off';
    end

    function stopSimulation()
        data.stop_flag = true;
        status_label.Text = 'Stopping...';
    end

    function config = collectConfig()
        % Collect all configuration from GUI
        config = struct();

        % FTN parameters
        config.tau = tau_field.Value;
        config.beta = beta_field.Value;
        config.sps = sps_field.Value;
        config.span = span_field.Value;

        % Hardware impairments
        config.pa_enabled = pa_check.Value;
        config.pa_model = pa_model_drop.Value;
        config.pa_ibo_db = pa_ibo_field.Value;
        config.iq_tx_enabled = iq_tx_check.Value;
        config.iq_tx_amp = iq_tx_amp_field.Value;
        config.iq_tx_phase = iq_tx_phase_field.Value;
        config.phase_noise_enabled = pn_check.Value;
        config.pn_variance = pn_var_field.Value;
        config.cfo_enabled = cfo_check.Value;
        config.cfo_hz = cfo_hz_field.Value;
        config.sample_rate = config.sps * 1000;

        % Simulation parameters
        config.SNR_train = snr_train_field.Value;
        config.SNR_test = snr_start_field.Value:snr_step_field.Value:snr_end_field.Value;
        config.N_train = n_train_field.Value;
        config.N_test = 20000;

        % Detection approaches
        config.use_neighbor = neighbor_check.Value;
        config.use_fractional = fractional_check.Value;
        config.use_hybrid = hybrid_check.Value;
        config.use_structured = structured_check.Value;

        % NN parameters
        config.hidden_sizes = [32, 16];
        config.max_epochs = 30;
        config.mini_batch = 512;
    end

    function results = runFTNSimulation(config)
        % Main simulation function (uses code from ftn_nn_with_impairments.m)

        % Generate pulse
        addOutput('Generating SRRC pulse...');
        h = rcosdesign(config.beta, config.span, config.sps, 'sqrt');
        h = h / norm(h);
        delay = config.span * config.sps;
        step = round(config.tau * config.sps);

        % Compute offsets
        addOutput('Computing sample offsets...');
        offsets = struct();
        offsets.neighbor = (-3:3) * step;
        offsets.fractional = round((-3:3) * (step-1) / 3);
        offsets.fractional(4) = 0;
        t1 = round(step / 3);
        t2 = round(2 * step / 3);
        offsets.hybrid = [-step, -t2, -t1, 0, t1, t2, step];

        % Generate training data
        addOutput(sprintf('Generating training data (%d symbols, SNR=%ddB)...', ...
                          config.N_train, config.SNR_train));
        progress_gauge.Value = 10;
        rng(42);
        bits_train = randi([0 1], 1, config.N_train);
        [rx_train, symbol_indices] = generate_ftn_rx_with_impairments(...
            bits_train, config.tau, config.sps, h, delay, config.SNR_train, config);

        % Train networks
        addOutput('Training neural networks...');
        networks = struct();
        approaches = {};
        total_approaches = sum([config.use_neighbor, config.use_fractional, ...
                               config.use_hybrid, config.use_structured]);
        approach_idx = 0;

        if config.use_neighbor
            approach_idx = approach_idx + 1;
            addOutput(sprintf('  [%d/%d] Training Neighbor detector...', ...
                             approach_idx, total_approaches));
            [X, y] = extract_features(rx_train, bits_train, symbol_indices, offsets.neighbor);
            networks.neighbor = train_nn(X, y, config.hidden_sizes, config.max_epochs, config.mini_batch);
            approaches{end+1} = 'neighbor';
            progress_gauge.Value = 10 + (approach_idx/total_approaches) * 30;
        end

        if config.use_fractional
            approach_idx = approach_idx + 1;
            addOutput(sprintf('  [%d/%d] Training Fractional detector...', ...
                             approach_idx, total_approaches));
            [X, y] = extract_features(rx_train, bits_train, symbol_indices, offsets.fractional);
            networks.fractional = train_nn(X, y, config.hidden_sizes, config.max_epochs, config.mini_batch);
            approaches{end+1} = 'fractional';
            progress_gauge.Value = 10 + (approach_idx/total_approaches) * 30;
        end

        if config.use_hybrid
            approach_idx = approach_idx + 1;
            addOutput(sprintf('  [%d/%d] Training Hybrid detector...', ...
                             approach_idx, total_approaches));
            [X, y] = extract_features(rx_train, bits_train, symbol_indices, offsets.hybrid);
            networks.hybrid = train_nn(X, y, config.hidden_sizes, config.max_epochs, config.mini_batch);
            approaches{end+1} = 'hybrid';
            progress_gauge.Value = 10 + (approach_idx/total_approaches) * 30;
        end

        if config.use_structured
            approach_idx = approach_idx + 1;
            addOutput(sprintf('  [%d/%d] Training Structured CNN...', ...
                             approach_idx, total_approaches));
            [X_struct, y_struct] = extract_structured_features(rx_train, bits_train, ...
                                                                symbol_indices, step);
            networks.structured = train_cnn(X_struct, y_struct, config.max_epochs, config.mini_batch);
            approaches{end+1} = 'structured';
            progress_gauge.Value = 10 + (approach_idx/total_approaches) * 30;
        end

        % Testing
        addOutput(sprintf('Testing over SNR range %d:%d:%d dB...', ...
                          config.SNR_test(1), config.SNR_test(2)-config.SNR_test(1), ...
                          config.SNR_test(end)));

        results = struct();
        for i = 1:length(approaches)
            results.(approaches{i}).BER = zeros(size(config.SNR_test));
            results.(approaches{i}).SNR = config.SNR_test;
        end

        for snr_idx = 1:length(config.SNR_test)
            if data.stop_flag, break; end

            snr_db = config.SNR_test(snr_idx);
            status_label.Text = sprintf('Testing SNR = %d dB...', snr_db);
            progress_gauge.Value = 40 + (snr_idx/length(config.SNR_test)) * 60;

            % Generate test data
            rng(100 + snr_idx);
            bits_test = randi([0 1], 1, config.N_test);
            [rx_test, sym_idx_test] = generate_ftn_rx_with_impairments(...
                bits_test, config.tau, config.sps, h, delay, snr_db, config);

            % Test each approach
            ber_str = sprintf('  SNR=%2ddB: ', snr_db);
            for app_idx = 1:length(approaches)
                app_name = approaches{app_idx};

                if strcmp(app_name, 'structured')
                    [X_test, valid_bits] = extract_structured_features(rx_test, bits_test, ...
                                                                        sym_idx_test, step);
                    bits_hat = detect_cnn(X_test, networks.structured);
                else
                    off = offsets.(app_name);
                    [X_test, valid_bits] = extract_features(rx_test, bits_test, ...
                                                             sym_idx_test, off);
                    bits_hat = detect_fc(X_test, networks.(app_name));
                end

                errors = sum(bits_hat ~= valid_bits);
                ber = errors / length(valid_bits);
                results.(app_name).BER(snr_idx) = ber;
                ber_str = [ber_str sprintf('%s=%.2e ', app_name(1:3), ber)];
            end
            addOutput(ber_str);
        end

        results.config = config;
        results.approaches = approaches;
        addOutput('Simulation complete!');
    end

    function addOutput(text)
        current = output_area.Value;
        if isempty(current)
            output_area.Value = {text};
        else
            output_area.Value = [current; {text}];
        end
        scroll(output_area, 'bottom');
        drawnow;
    end

    function plotResults()
        if isempty(data.results), return; end

        cla(results_axes);
        hold(results_axes, 'on');
        grid(results_axes, 'on');

        approaches = data.results.approaches;
        colors = lines(length(approaches));
        markers = {'o-', 's-', '^-', 'd-'};

        legends = {};
        for i = 1:length(approaches)
            app_name = approaches{i};
            semilogy(results_axes, ...
                     data.results.(app_name).SNR, ...
                     data.results.(app_name).BER, ...
                     markers{mod(i-1, 4)+1}, ...
                     'Color', colors(i,:), ...
                     'LineWidth', 2, ...
                     'MarkerSize', 8);
            legends{i} = capitalize(app_name);
        end

        xlabel(results_axes, 'SNR (dB)', 'FontSize', 12);
        ylabel(results_axes, 'Bit Error Rate', 'FontSize', 12);
        title(results_axes, sprintf('BER Performance (\\tau=%.2f, PA=%s)', ...
                                    data.results.config.tau, ...
                                    data.results.config.pa_model), ...
              'FontSize', 13);
        legend(results_axes, legends, 'Location', 'southwest');
        ylim(results_axes, [1e-4 1]);
        hold(results_axes, 'off');
    end

    function saveResults()
        if isempty(data.results), return; end

        [file, path] = uiputfile('ftn_nn_results.mat', 'Save Results');
        if file ~= 0
            results = data.results;
            save(fullfile(path, file), 'results');
            uialert(fig, 'Results saved successfully!', 'Success', 'Icon', 'success');
        end
    end

    function str = capitalize(s)
        str = [upper(s(1)) s(2:end)];
    end

    % Include all helper functions from ftn_nn_with_impairments.m
    % (These would be the same functions, included here for completeness)

end

% ========================================================================
% HELPER FUNCTIONS (shared with command-line version)
% ========================================================================
% All functions from ftn_nn_with_impairments.m would be included here
% For brevity, assuming they are available in the path or included
