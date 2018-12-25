function dudt = FNneuron(t,u,Ia)
% FitzHugh-Negumo Neuron Model
%
% dudt = FNneuron(t,u,Ia)
%
% Explanation of variables:
% ------------------------
% t: time (scalar)
% u: state vector
% Ia: applied current at time t
%
%

% FHN neuron parameters
a = 0.139;
eps = 0.008;
g = 2.54;

dudt(1) = u(1)*(u(1)-a)*(1-u(1)) - u(2) + u(3) + Ia;	% membrane voltage (fast)
dudt(2) = eps*(u(1)-g*u(2));                            % membrane conductance (slow)
dudt(3:5) = 0;                                          % placeholders
dudt = dudt';