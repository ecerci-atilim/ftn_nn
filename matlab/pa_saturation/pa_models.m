function y_out = pa_models(x_in, model_type, params)
% PA_MODELS - Power Amplifier Nonlinearity Models
%
% Implements various PA saturation models for communication systems
% with numerical safeguards to prevent NaN/Inf issues.
%
% Inputs:
%   x_in       - Input signal (complex baseband)
%   model_type - Type of PA model: 'rapp', 'saleh', 'soft_limiter', 'none'
%   params     - Structure with model parameters
%
% Output:
%   y_out      - Output signal after PA nonlinearity
%
% References:
%   [1] Rapp, C. (1991). "Effects of HPA-Nonlinearity on a 4-DPSK/OFDM-Signal
%       for a Digital Sound Broadcasting System"
%   [2] Saleh, A. A. (1981). "Frequency-independent and frequency-dependent
%       nonlinear models of TWT amplifiers"
%
% Author: Emre Cerci
% Date: January 2026

    % Handle empty input
    if isempty(x_in)
        y_out = x_in;
        return;
    end

    switch lower(model_type)
        case 'none'
            % Bypass PA model
            y_out = x_in;
            return;
            
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
            
            % Numerical safeguards
            params.Asat = max(params.Asat, eps);  % Prevent division by zero
            params.p = max(params.p, 0.1);        % Minimum smoothness

            r = abs(x_in);
            phi = angle(x_in);
            
            % Prevent overflow: clip very large values
            r = min(r, 100 * params.Asat);

            % Rapp AM/AM conversion with numerical stability
            ratio = r / params.Asat;
            ratio_power = ratio.^(2*params.p);
            ratio_power = min(ratio_power, 1e10);  % Prevent overflow
            r_out = params.G * r ./ (1 + ratio_power).^(1/(2*params.p));

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
            
            % Numerical safeguards
            params.beta_a = max(params.beta_a, eps);
            params.beta_p = max(params.beta_p, eps);

            r = abs(x_in);
            phi = angle(x_in);
            
            % Clip very large values
            r = min(r, 1e6);
            r_sq = r.^2;
            r_sq = min(r_sq, 1e12);  % Prevent overflow

            % Saleh AM/AM conversion
            r_out = params.alpha_a * r ./ (1 + params.beta_a * r_sq);

            % Saleh AM/PM conversion (phase distortion)
            phi_out = phi + params.alpha_p * r_sq ./ (1 + params.beta_p * r_sq);

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

            r = abs(x_in);
            phi = angle(x_in);

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
end
