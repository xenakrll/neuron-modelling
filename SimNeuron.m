function [u, uSpike, ix, SNR] = SimNeuron(NeuronFcn,t,u0,Ia,preNoise,spkThr)
% Simulate Neuron Model
%
% [u, uSpike, ix, SNR] = SimNeuron(NeuronFcn,t,u0,Ia,preNoise,spkThr)
%
% NeuronFcn:    Handle to a function containing the neuron model.
% t:            Time vector (constant dt).
% u0:           Model initial conditions.
% Ia:           External forcing (input signal).
% preNoise:     Pre-determined noise vector.
% spkThr:       Threshold at which membrane potential is considered spike.
% ---
% u:            Integration results.
% uSpike:       Indicator variable for when spikes occur.
% ix:           Spike times and values.
% SNR:          SR measure.
%
%
% Z. Danziger June 2014
% 
%

dt = t(2)-t(1);                     % timestep
K = length(t);                      % vector length
u = zeros(K,length(u0))*nan;        % pre-allocate storage variables
u(1,:) = u0;                        % initial conditions
Iav = zeros(K,1)*nan;               % forcing

for k=2:K
    % update the current plus noise
    Iav(k) = Ia(k) + preNoise(k);
    
    % incremment the solution vector with the new estimate at time t+dt
    u(k,:) = u(k-1,:) + dt*NeuronFcn(t(k),u(k-1,:),Iav(k))';
end


% -- spike times --
% find all spikes
ix = peakdet(u(:,1),spkThr);
uSpike = zeros(K,1);
% mark location of each spike and create a 2ms pulse at each location
% [28] in Yu et al., 2001
for i=1:size(ix,1)
    % pulse width 1ms under and over recorded spike time
    pulse = [ix(i,1)-1/dt ix(i,1)+1/dt];
    % if pulse goes over K or under 0, clip it
    if ix(i,1)-1/dt<1
        pulse(1) = 1;
    elseif ix(i,1)+1/dt>K
        pulse(2) = K;
    end
    uSpike(pulse(1):pulse(2)) = 1;
end


% SNR measure (Collins 1995):
% max normalized correlation value at up to +1ms lag (1/dt)
[c, lags] = xcorr(uSpike,Ia-mean(Ia),1/dt,'biased'); % xcorrelation
[SNR(1), mlag] = max( c(find(lags==0):end) );      % max of positive lags
% power norm at peak xcor
S0 = Ia(1:end-mlag+1)-mean(Ia(1:end-mlag+1));
C0 = mean( S0 .* uSpike(mlag:end)' );
SNR(2) = C0 / ( sqrt(mean(S0.^2))*sqrt(mean((uSpike(mlag:end)-mean(uSpike(mlag:end))).^2)) );




