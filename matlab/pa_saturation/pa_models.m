function y_out = pa_models(x_in, model_type, params)
% PA_MODELS - Power Amplifier Nonlinearity Models with Numerical Safeguards
%
% Implements various PA saturation models for communication systems
%
% Inputs:
%   x_in       - Input signal (complex baseband)
%   model_type - Type of PA model: 'rapp', 'saleh', 'soft_limiter'
%   params     - Structure with model parameters
%
% Output:
%   y_out      - Output signal after PA nonlinearity
%
% Numerical safeguards:
%   - Handles zero input (returns zero output)
%   - Clamps very large inputs to prevent overflow
%   - Preserves NaN/Inf positions from input
%
% References:
%   [1] Rapp, C. (1991). "Effects of HPA-Nonlinearity on a 4-DPSK/OFDM-Signal
%       for a Digital Sound Broadcasting System"
%   [2] Saleh, A. A. (1981). "Frequency-independent and frequency-dependent
%       nonlinear models of TWT amplifiers"
%
% Author: Emre Cerci
% Date: January 2026

    % Input validation and safeguards
    if isempty(x_in)
        y_out = x_in;
        return;
    end
    
    % Store original shape for output
    original_shape = size(x_in);
    
    % Track problematic values
    nan_mask = isnan(x_in);
    inf_mask = isinf(x_in);
    
    % Maximum input amplitude to prevent numerical overflow
    MAX_INPUT = 1e6;
    x_clamped = x_in;
    large_mask = abs(x_in) > MAX_INPUT;
    if any(large_mask(:))
        x_clamped(large_mask) = MAX_INPUT * exp(1j * angle(x_in(large_mask)));
    end

    switch lower(model_type)
        case 'rapp'
            % Rapp Model (Solid State Power Amplifier)
            % y(r) = G * r / (1 + (r/Asat)^(2p))^(1/(2p))
            %
            % Parameters:
            %   G     - Small signal gain (default: 1)
            %   Asat  - Saturation amplitude (default: 1)
            %   p     - Smoothness factor (default: 2, typical: 1-5)

            if ~isfield(params, 'G'), params.G = 1; end
            if ~isfield(params, 'Asat'), params.Asat = 1; end
            if ~isfield(params, 'p'), params.p = 2; end
            
            % Ensure positive parameters
            params.G = abs(params.G);
            params.Asat = max(abs(params.Asat), eps);  % Prevent division by zero
            params.p = max(params.p, 0.1);

            r = abs(x_clamped);
            phi = angle(x_clamped);

            % Rapp AM/AM conversion with safeguard
            ratio = r / params.Asat;
            ratio = min(ratio, 1e10);  % Prevent overflow in power
            denom = (1 + ratio.^(2*params.p)).^(1/(2*params.p));
            denom = max(denom, eps);  % Prevent division by zero
            r_out = params.G * r ./ denom;

            % No AM/PM (phase distortion) in basic Rapp model
            y_out = r_out .* exp(1j * phi);

        case 'saleh'
            % Saleh Model (Traveling Wave Tube Amplifier)
            % AM/AM: A(r) = alpha_a * r / (1 + beta_a * r^2)
            % AM/PM: Phi(r) = alpha_p * r^2 / (1 + beta_p * r^2)
            %
            % Parameters:
            %   alpha_a, beta_a - AM/AM parameters
            %   alpha_p, beta_p - AM/PM parameters

            if ~isfield(params, 'alpha_a'), params.alpha_a = 2.0; end
            if ~isfield(params, 'beta_a'), params.beta_a = 1.0; end
            if ~isfield(params, 'alpha_p'), params.alpha_p = pi/3; end
            if ~isfield(params, 'beta_p'), params.beta_p = 1.0; end
            
            % Ensure positive parameters
            params.beta_a = max(params.beta_a, eps);
            params.beta_p = max(params.beta_p, eps);

            r = abs(x_clamped);
            phi = angle(x_clamped);

            % Saleh AM/AM conversion with safeguard
            denom_am = 1 + params.beta_a * r.^2;
            denom_am = max(denom_am, eps);
            r_out = params.alpha_a * r ./ denom_am;

            % Saleh AM/PM conversion (phase distortion) with safeguard
            denom_pm = 1 + params.beta_p * r.^2;
            denom_pm = max(denom_pm, eps);
            phi_out = phi + params.alpha_p * r.^2 ./ denom_pm;

            y_out = r_out .* exp(1j * phi_out);

        case 'soft_limiter'
            % Soft Limiter (Simple Clipping with Continuous Transition)
            % y(r) = r                                      if r <= A_lin
            %      = A_lin + (r - A_lin) * compress         if A_lin < r < A_sat
            %      = A_lin + (A_sat - A_lin) * compress     if r >= A_sat (continuous)
            %
            % Parameters:
            %   A_lin    - Linear region threshold (default: 0.7)
            %   A_sat    - Input threshold for saturation (default: 1.0)
            %   compress - Compression factor in transition (default: 0.1)
            %
            % Note: The saturation output level ensures continuity at r = A_sat:
            %       A_out_max = A_lin + (A_sat - A_lin) * compress

            if ~isfield(params, 'A_lin'), params.A_lin = 0.7; end
            if ~isfield(params, 'A_sat'), params.A_sat = 1.0; end
            if ~isfield(params, 'compress'), params.compress = 0.1; end
            
            % Ensure valid parameters
            params.A_lin = max(params.A_lin, 0);
            params.A_sat = max(params.A_sat, params.A_lin + eps);
            params.compress = max(min(params.compress, 1), 0);

            r = abs(x_clamped);
            phi = angle(x_clamped);

            r_out = zeros(size(r));

            % Compute the continuous saturation output level
            A_out_sat = params.A_lin + (params.A_sat - params.A_lin) * params.compress;

            % Linear region
            linear_idx = r <= params.A_lin;
            r_out(linear_idx) = r(linear_idx);

            % Transition region (soft compression)
            transition_idx = (r > params.A_lin) & (r < params.A_sat);
            r_out(transition_idx) = params.A_lin + ...
                (r(transition_idx) - params.A_lin) * params.compress;

            % Saturation region (continuous with transition region)
            sat_idx = r >= params.A_sat;
            r_out(sat_idx) = A_out_sat;

            y_out = r_out .* exp(1j * phi);

        otherwise
            error('Unknown PA model type: %s. Use ''rapp'', ''saleh'', or ''soft_limiter''.', model_type);
    end
    
    % Restore NaN/Inf positions from original input
    y_out(nan_mask) = NaN;
    y_out(inf_mask) = sign(real(x_in(inf_mask))) * Inf;
    
    % Handle zero inputs (should remain zero)
    zero_mask = (x_in == 0);
    y_out(zero_mask) = 0;
    
    % Ensure output shape matches input
    y_out = reshape(y_out, original_shape);
end
