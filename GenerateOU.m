function pn = GenerateOU(K,dt,mu,sig,theta)
% Generate Ornstein-Uhlenbeck process
%
% pn = GenerateOU(K,dt,mu,sig,theta)
%
% K:        Length of noise vector to generate
% dt:       Timestep
% mu:       Process mean
% sig:      RMS noise amplitude (standard deviation)
% theta:    Noise correlation rate
%
% 
% Z. Danziger June 2014
%


if theta/2 > 1/dt
    warning('timestep too large for given correlation rate')
end

% simple loop to generate noise
pn = zeros(1,K);            % hold noise
S = sqrt(2*theta*sig^2);    % noise standard deviation (accurate)
for k=2:K
    % integrate noise SDE (AR1 model)
    pn(k) = pn(k-1) + theta*(mu-pn(k-1))*dt + S*randn*sqrt(dt);
end
