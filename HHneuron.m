function dudt = HHneuron(t,u,Ia)
% Hodgkin & Huxley Neuron takes input of I and t to output membrane volts
%
% dudt = HHneuron(t,u,Ia)
%
% Explanation of variables:
% ------------------------
% t: time (scalar)
% u: state vector
% Ia: applied current at time t
%
%
% sources:
% (1) http://lcn.epfl.ch/~gerstner/SPNM/node14.html
% (2) Lee, S. G. and S. Kim (1999). "Parameter dependence of stochastic
%   resonance in the stochastic Hodgkin-Huxley neuron." Physical Review E
%   60(1): 826-830.
%
%


% --- empirical parameters ---
% reversal potentials, E, are in mV and the conductances, g, are in mS/cm^2
% taken from Hodgkin & Huxley
E.Na = 115;     g.Na = 120; % sodium
E.K = -12;      g.K = 36;   % potassium
E.L = 10.6;     g.L = 0.3;  % leakage
Cm = 1;                     % membrane capacitance at 1uF/cm^2


% Membrane voltage (u) ODE, and where m, h, and n are the gating variables
dudt(1) = ( Ia - ...                           % |-  applied current
         g.Na*u(2)^3*u(4)*(u(1)-E.Na) - ...    % |   sodium current
         g.K*u(3)^4*(u(1)-E.K) - ...           % |   potassium current
         g.L*(u(1)-E.L) ) ...                  % |   current leakage
         /Cm;                                  % |-  divide by membrane capacitance
dudt(2) = (2.5-0.1*u(1))/(exp(2.5-0.1*u(1))-1)*(1-u(2)) - ...
           4*exp(-u(1)/18)*u(2);                            % gating var m
dudt(3) = (0.1-0.01*u(1))/(exp(1-0.1*u(1))-1)*(1-u(3)) - ...
           0.125*exp(-u(1)/80)*u(3);                        % gating var n
dudt(4) = 0.07*exp(-u(1)/20)*(1-u(4)) - ...
           1/(exp(3-0.1*u(1))+1)*u(4);                      % gating var h
dudt(5) = 0;                                                % placeholder value to keep dims consistent with other models

dudt = dudt';
       





