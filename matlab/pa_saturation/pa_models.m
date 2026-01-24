function y_out = pa_models(x_in, model_type, params)
% PA_MODELS - Power Amplifier Nonlinearity Models
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
% References:
%   [1] Rapp, C. (1991). "Effects of HPA-Nonlinearity on a 4-DPSK/OFDM-Signal
%       for a Digital Sound Broadcasting System"
%   [2] Saleh, A. A. (1981). "Frequency-independent and frequency-dependent
%       nonlinear models of TWT amplifiers"
%
% Author: Emre Cerci
% Date: January 2026

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

            r = abs(x_in);
            phi = angle(x_in);

            % Rapp AM/AM conversion
            r_out = params.G * r ./ (1 + (r / params.Asat).^(2*params.p)).^(1/(2*params.p));

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

            r = abs(x_in);
            phi = angle(x_in);

            % Saleh AM/AM conversion
            r_out = params.alpha_a * r ./ (1 + params.beta_a * r.^2);

            % Saleh AM/PM conversion (phase distortion)
            phi_out = phi + params.alpha_p * r.^2 ./ (1 + params.beta_p * r.^2);

            y_out = r_out .* exp(1j * phi_out);

        case 'soft_limiter'
            % Soft Limiter (Simple Clipping)
            % y(r) = r              if r <= A_lin
            %      = A_lin + (r - A_lin) * compression_factor   if A_lin < r < A_sat
            %      = A_sat           if r >= A_sat
            %
            % Parameters:
            %   A_lin    - Linear region threshold (default: 0.7)
            %   A_sat    - Saturation level (default: 1.0)
            %   compress - Compression factor in transition (default: 0.1)

            if ~isfield(params, 'A_lin'), params.A_lin = 0.7; end
            if ~isfield(params, 'A_sat'), params.A_sat = 1.0; end
            if ~isfield(params, 'compress'), params.compress = 0.1; end

            r = abs(x_in);
            phi = angle(x_in);

            r_out = zeros(size(r));

            % Linear region
            linear_idx = r <= params.A_lin;
            r_out(linear_idx) = r(linear_idx);

            % Transition region (soft compression)
            transition_idx = (r > params.A_lin) & (r < params.A_sat);
            r_out(transition_idx) = params.A_lin + ...
                (r(transition_idx) - params.A_lin) * params.compress;

            % Saturation region
            sat_idx = r >= params.A_sat;
            r_out(sat_idx) = params.A_sat;

            y_out = r_out .* exp(1j * phi);

        otherwise
            error('Unknown PA model type: %s. Use ''rapp'', ''saleh'', or ''soft_limiter''.', model_type);
    end
end
