% Script to execute code related to Danziger & Grill J. Comp. Neuro. 2014
%
% Evaluate cells individually (ctrl+enter) to run code. Note that some
% simulations may take significant time. For simulations taking over 20 min
% to run (especially fig. 3 which may take up to 3 days) simulation data is
% included with the code to avoid having to re-run the analysis. To use the
% included data, set dataFlag variable to 1, to simulate the data, set
% dataFlag to 0.
%
% Parameters have been set to reduced resolution in this code compared to
% that used to generate the manuscript figures to allow timley completion
% of the analysis.
%
%
% Code implemented on version 8.1.0.604 (R2013a)
% 
%
% Z. Danziger June 2014
%


%% Generate Figure 1 (find neuron threshold)
% individual simulation run parameters
t0 = 0;             % start time
tf = 300;          % stop time
dt = 0.025;         % time incrememnt
t = t0:dt:tf;       % vector of simulation times
% list of amps to test
ampVec = 10; %5:1:12;
% hold data
C = zeros(1,length(ampVec));    

hw = waitbar(0,'progress');
for k=1:length(ampVec)
    % input signal
    p.A = ampVec(k);
    p.f = 1/35; p.dc = 0; p.duty = 30; p.i = 80;
    Ia = PhysEUSburst(t,p);
    Ia(1:find(t>=60,1,'first'))=0;  % allow model to reach SS
    % noise
    preNoise = zeros(1,length(t));
    
    % --- integrate model ---
    [u, uSpike, ix, SNR] = SimNeuron(@HHneuron,t,[0 0.05 0.32 0.60 0]',Ia,preNoise,50);
    C(k) = SNR(2);
    waitbar(k/length(ampVec),hw)

end
close(hw)

C(isnan(C))=0;     % define no spikes as C=0;
figure

%plot(ampVec,C,'-ko','markersize',10,'markerfacecolor','k','linewidth',3)
%xlabel('RMS amp.'); ylabel('C_1')

subplot(2,1,1);
plot(t,u)

subplot(2,1,2);
plot(t,Ia);



%% Generate Figure 2 (example traces)

% individual simulation run parameters
t0 = 0;             % start time
tf = 2075;          % stop time
dt = 0.025;         % time incrememnt
t = t0:dt:tf;       % vector of simulation times
K = length(t);

figure

% input signal
p.f = 1/35; p.dc = 0; p.duty = 45; p.i = 163; p.A = 6.5;
Ia = PhysEUSburst(t,p);
Ia(1:find(t>=60,1,'first'))=0;  % allow model to reach SS

% --Examples:
aw=0.15; bt=7.29; bw=1.0526; sigVar=4.62;                       % pulse train parameters
amp = ((bw+aw)/(bt+bw))^(-1/2)*sigVar;
preNoise = VariablePulseTrain2(tf/dt+1,aw/dt,bw/dt,bt/dt)*amp;  % build stochastic pulse train
% --- integrate model ---
[u, uSpike, ix, SNR] = SimNeuron(@HHneuron,t,[0 0.05 0.32 0.60 0]',Ia,preNoise,50);
subplot(3,2,1); plot(t,Ia+preNoise,t,Ia); hold on; plot(t(ix(:,1)),max(preNoise+Ia)+1,'ok'); xlim(t([1 end]))

aw=0.15; bt=7.29; bw=1.3684; sigVar=8.24;                       % pulse train parameters
amp = ((bw+aw)/(bt+bw))^(-1/2)*sigVar;
preNoise = VariablePulseTrain2(tf/dt+1,aw/dt,bw/dt,bt/dt)*amp;  % build stochastic pulse train
% --- integrate model ---
[u, uSpike, ix, SNR] = SimNeuron(@HHneuron,t,[0 0.05 0.32 0.60 0]',Ia,preNoise,50);
subplot(3,2,3); plot(t,Ia+preNoise,t,Ia); hold on; plot(t(ix(:,1)),max(preNoise+Ia)+1,'ok'); xlim(t([1 end]))

aw=0.15; bt=7.29; bw=1.6842; sigVar=11.86;                       % pulse train parameters
amp = ((bw+aw)/(bt+bw))^(-1/2)*sigVar;
preNoise = VariablePulseTrain2(tf/dt+1,aw/dt,bw/dt,bt/dt)*amp;  % build stochastic pulse train
% --- integrate model ---
[u, uSpike, ix, SNR] = SimNeuron(@HHneuron,t,[0 0.05 0.32 0.60 0]',Ia,preNoise,50);
subplot(3,2,5); plot(t,Ia+preNoise,t,Ia); hold on; plot(t(ix(:,1)),max(preNoise+Ia)+1,'ok'); xlim(t([1 end]))

preNoise = GenerateOU(K,dt,0,0.6,0.5);
% --- integrate model ---
[u, uSpike, ix, SNR] = SimNeuron(@HHneuron,t,[0 0.05 0.32 0.60 0]',Ia,preNoise,50);
subplot(3,2,2); plot(t,Ia+preNoise,t,Ia); hold on; plot(t(ix(:,1)),max(preNoise+Ia)+1,'ok'); xlim(t([1 end]))

preNoise = GenerateOU(K,dt,0,1.5,0.5);
% --- integrate model ---
[u, uSpike, ix, SNR] = SimNeuron(@HHneuron,t,[0 0.05 0.32 0.60 0]',Ia,preNoise,50);
subplot(3,2,4); plot(t,Ia+preNoise,t,Ia); hold on; plot(t(ix(:,1)),max(preNoise+Ia)+1,'ok'); xlim(t([1 end]))

preNoise = GenerateOU(K,dt,0,4.5,0.5);
% --- integrate model ---
[u, uSpike, ix, SNR] = SimNeuron(@HHneuron,t,[0 0.05 0.32 0.60 0]',Ia,preNoise,50);
subplot(3,2,6); plot(t,Ia+preNoise,t,Ia); hold on; plot(t(ix(:,1)),max(preNoise+Ia)+1,'ok'); xlim(t([1 end]))



%% Generate Figure 3 (biphasic perturbation SR parameter space)
dataFlag = 0;

% all simulation data will be stored in SNRdata in the following format:
% each row is a simulation result and each column contains the following
%   SNRdata(i,:) = [aw bw bt Arms a C1]
%
% see manuscript (fig 3A) for variable explanations

if dataFlag

load biphaseSNRparamsearch_AllRunsHH_09-Jun-2014
% repetitions of the parameter-space exploration are stacked along the 3rd
% array dimension - average them
SNRdata = mean(SNRdata,3);

% produce parameter ranges
[vaw vbw vbt vPow] = deal(unique(SNRdata(:,1)),unique(SNRdata(:,2)),...
    unique(SNRdata(:,3)),unique(SNRdata(:,4)));
    
else
    
% individual simulation run parameters
t0 = 0;             % start time
tf = 2075;          % stop time
dt = 0.025;         % time incrememnt
t = t0:dt:tf;       % vector of simulation times
K = length(t);

% range of parameter space considered
% (the resolution is reduced compared to the published figure due to the
% time-intensive nature of spanning the full parameter space, and also note
% that this code produces 1 repetition, whereas the published figure is an
% average over many such repetitions)
naw = 1;            % num runs for min pulse width
vaw = 0.15;
nbw = 6;            % num runs for max pulse width
vbw = linspace(0.8,2,nbw);
nbt = 4;            % num runs for max IPI
vbt = linspace(1.5,15,nbt);
nPow = 10;          % num runs for RMS amp.
vPow = linspace(1,16,nPow);

% input signal
p.f = 1/35; p.dc = 0; p.duty = 45; p.i = 163; p.A = 6.5;
Ia = PhysEUSburst(t,p);
Ia(1:find(t>=60,1,'first'))=0;  % allow model to reach SS

% store data here
SNRdata = zeros(1+prod( [naw nbw nbt nPow] ), 6) * nan;
% counter
cc = 1;

% loop over all parameter space
hw = waitbar(0,'progress');
for n1=1:naw
    aw = vaw(n1);
for n2=1:nbw
    bw = vbw(n2);
for n3=1:nbt
    if aw>=bw, break; end   % do not allow bad inputs to pulse generator
    bt = vbt(n3);
for n4=1:nPow
    if bw>=bt, break; end   % constraints on pulses
    Pow = vPow(n4);
    % calculate pulse amplitude to yeild desired power
    amp = ((bw+aw)/(bt+bw))^(-1/2)*Pow;
    % generate noise
    preNoise = VariablePulseTrain2(tf/dt+1,aw/dt,bw/dt,bt/dt)*amp;
    
    % --- integrate model ---
    [u, uSpike, ix, SNR] = SimNeuron(@HHneuron,t,[0 0.05 0.32 0.60 0]',Ia,preNoise,50);
    % save data from this simulation
    SNRdata(cc,:) = [aw bw bt Pow amp SNR(2)];
    cc = cc+1;
    waitbar(cc/length(SNRdata))
end     % n4
end     % n3
end     % n2
end     % n1
close(hw)

SNRdata = SNRdata(1:cc-1,:);    % dump memory placeholders
SNRdata(isnan(SNRdata))=0;      % zero nans, which indicate C1=0

end

% -- Plot Results:
% 3D points colored by SNR [pow x bw x bt]
surfData.x = zeros(length(vPow),length(vbw),length(vbt))*nan;
surfData.y = zeros(length(vPow),length(vbw),length(vbt))*nan;
surfData.z = zeros(length(vPow),length(vbw),length(vbt))*nan;
surfData.c = zeros(length(vPow),length(vbw),length(vbt))*nan;
SNRdata1aw = SNRdata(1:length(SNRdata)/length(vaw),:);  % only use worst aw
% build each surface
for k = 1:length(SNRdata1aw)
    datapt =  SNRdata1aw(k,:);
    inds = [find(datapt(4)==vPow), find(datapt(2)==vbw), ...
        find(datapt(3)==vbt)];
    surfData.c(inds(1),inds(2),inds(3)) = datapt(6);
    surfData.x(inds(1),inds(2),inds(3)) = datapt(4);
    surfData.y(inds(1),inds(2),inds(3)) = datapt(2);
    surfData.z(inds(1),inds(2),inds(3)) = datapt(3);
end
% plot each surface
figure; hold on; hs = 0;
for k=1:length(vbt)
    hs(k) = surf(surfData.x(:,:,k),surfData.y(:,:,k),...
        surfData.z(:,:,k),surfData.c(:,:,k));
end
xlabel('RMS amp.'); ylabel('bw'); zlabel('bt')
set(gca,'dataaspectRatio',[4 1 4])
set(gca,'cameraviewAngle',9.36,'cameraPosition',[76 -16 51])
set(hs,'edgecolor','none')
grid on
ylim([min(vbw) max(vbw)])
xlim([min(vPow) max(vPow)])
colorbar

% histogram of SNR distributions
figure
subplot(2,1,1)
ixaw = SNRdata(:,1)==0.15;
SNRdata_aw = SNRdata(ixaw,:);
mxSNR = max(SNRdata_aw(:,6));
rd=SNRdata_aw(:,6)>=mxSNR*0.85 & SNRdata_aw(:,6)<mxSNR;     % red dots
bd=SNRdata_aw(:,6)<mxSNR*0.85;                              % black dots
gd=SNRdata_aw(:,6)==mxSNR;                                  % green dot
plot(SNRdata_aw(bd,4),SNRdata_aw(bd,6),'.k','markersize',5)
hold on
plot(SNRdata_aw(rd,4),SNRdata_aw(rd,6),'.r','markersize',15)
plot(SNRdata_aw(gd,4),SNRdata_aw(gd,6),'.','color',[0.25 0.8 0.35],'markersize',25)
xlim([min(vPow)-0.5 max(vPow)+0.5])
ylim([0 0.25])
subplot(2,1,2)
hist(SNRdata_aw(:,6))
xlabel('C_1'); ylabel('frequency')




%% Generate Figure 4 (exploration of OU noise correlation rates)

% -- Note that the manuscript figure was the result of averaging over many
% repetitions of this script.

% individual simulation run parameters
t0 = 0;             % start time
tf = 2075;          % stop time
dt = 0.025;         % time incrememnt
t = t0:dt:tf;       % vector of simulation times
K = length(t);

% input signal
p.f = 1/35; p.dc = 0; p.duty = 45; p.i = 163; p.A = 6.5;
Ia = PhysEUSburst(t,p);
Ia(1:find(t>=60,1,'first'))=0;  % allow model to reach SS

% parameter space definitions
% (you may change parameter resolutions for evaluation speeds, steps for
% RMS amp., or vPow, should be >=0.25 especially to detect sharp SR peaks
% at low powers)
vCT = logspace(log10(0.005),log10(5),10);   % correlation rates
nCT = length(vCT);
vPow = 0.2:0.8:10;                            % RMS amps.
nPow = length(vPow);

% store data here
SNRdata = zeros(1+prod([nCT nPow]),3) * nan;
% counter
cc = 1;

% loop over all parameter space
hw = waitbar(0,'progress');
for n1=1:nCT
    CT = vCT(n1);
for n2=1:nPow
    Pow = vPow(n2);
    
    % -- generate OU noise
    preNoise = GenerateOU(K,dt,0,Pow,CT);
    
    % --- integrate model ---
    [u, uSpike, ix, SNR] = SimNeuron(@HHneuron,t,[0 0.05 0.32 0.60 0]',Ia,preNoise,50);
    % save data from this simulation
    SNRdata(cc,:) = [CT Pow SNR(2)];
    cc = cc+1;
    
    waitbar(cc/length(SNRdata))
end     % RMS amp
end     % cor. rate
close(hw)

SNRdata = SNRdata(1:cc-1,:);    % dump memory placeholders
SNRdata(isnan(SNRdata))=0;      % zero nans, which indicate C1=0

% -- plot results
% correlation time vs RMS power
figure;
subplot(5,1,[1 4])
% MATLAB will not display the last bit of data without padding:
SNRpcol = [SNRdata; [ones(length(vPow),1)*(max(vCT)+1) vPow(:) zeros(length(vPow),1)]];
hp = pcolor( reshape(SNRpcol(:,1),length(vPow),[]), ...
             reshape(SNRpcol(:,2),length(vPow),[]), ...
             reshape(SNRpcol(:,3),length(vPow),[]) );
set(hp,'edgecolor','none')
set(gca,'xscale','log')
caxis([0 0.23])
ylabel('RMS'); 
xl = get(gca,'xlim');
colorbar('location','North')

% look at max C1 over each correlation rate value
subplot(5,1,5)
tmp = bsxfun(@eq,SNRdata(:,1),vCT);         % matix of indicies of where each entry matches vCT
[row, col]=find(tmp');                      % indicies into vPow (transpose tmp because find looks down rows first)
A = accumarray(row,SNRdata(:,3),[],@max);   % accumilate maximums
plot(vCT,A); set(gca,'xscale','log')
set(gca,'xlim',xl-[0 1])
ylabel('max[C_1]'); xlabel('Noise Regression Rate, r_c (ms^{-1})'); ylim([0.15 0.23])




%% Generate Figure 5 (RMS amp. vs C1 for varied noise types)

% Figure 6 is generated using the same procedure, simply switching out the
% 'Ia' variable used.

% parameters:
sigVarv = [0:0.5:4 5:2:20];         % noise intensities vector
% parameters used for figure:   sigVarv = [0:0.5:15.5 16:2:26];
numSigs = length(sigVarv);          % how many intensites we consider
numReps = 4;                        % number of repetitions for each intensity (n=20 for figure)
t0 = 0;             % start time
tf = 2075;          % stop time
dt = 0.025;         % time incrememnt (high correlation rates typically require smaller step sizes)
t = t0:dt:tf;       % vector of simulation times
K = length(t);

% input signal
p.f = 1/35; p.dc = 0; p.duty = 45; p.i = 163; p.A = 6.5;
Ia = PhysEUSburst(t,p);         % change Ia (input signal) for Fig. 6
Ia(1:find(t>=60,1,'first'))=0;  % allow model to reach SS

% data storage
SNRstruct = [];

% list of perturbation types
noiseTypes = {'BPHHstocLP','BPHHstocHP','OU','OUlr','OUhr'};
for nn = 1:length(noiseTypes)
    % which noise
    curType = noiseTypes{nn};
    SNRstruct.(curType) = ones(numSigs,numReps)*nan;  % store results
    
for rr = 1:numSigs
    sigVar = sigVarv(rr);   % noise intensity increments
    % --- run the simulation numReps times ---
    for kk=1:numReps
        
        % generate noise
        switch curType
            case 'OU'
                % === OU noise ===
                preNoise = GenerateOU(K,dt,0,sigVar,0.5);
            case 'BPHHstocHP'
                % === stochastic biphsic noise for the HH neuron (high power) ===
                % pulse train parameters
                bt = 3.43; bw = 1.68; aw = 0.15; amp = ((bw+aw)/(bt+bw))^(-1/2)*sigVar;
                preNoise = VariablePulseTrain2(tf/dt+1,aw/dt,bw/dt,bt/dt)*amp;
            case 'BPHHstocLP'
                % === stochastic biphsic noise for the HH neuron (high power) ===
                % pulse train parameters
                bt = 11.14; bw = 1.94; aw = 0.15; amp = ((bw+aw)/(bt+bw))^(-1/2)*sigVar;
                preNoise = VariablePulseTrain2(tf/dt+1,aw/dt,bw/dt,bt/dt)*amp;
            case 'OUlr'
                % === OU noise with low correlation rate ===
                preNoise = GenerateOU(K,dt,0,sigVar,0.008);
            case 'OUhr'
                % === OU noise with high correlation rate ===
                preNoise = GenerateOU(K,dt,0,sigVar,24);
        end
        
        % --- integrate model ---
        [u, uSpike, ix, SNR] = SimNeuron(@HHneuron,t,[0 0.05 0.32 0.60 0]',Ia,preNoise,50);
        % store data
        SNRstruct.(curType)(rr,kk) = SNR(:,2);
    end
    
    % progress report:
    fprintf('Simulation Progress: type - %3.2g, repitition - %3.2g\n',...
        nn/length(noiseTypes),rr/numSigs)
end  
end

% plot results
cmp = hsv(length(noiseTypes));
figure; hold on
for k=1:length(noiseTypes)
    errorbar(sigVarv,nanmean(SNRstruct.(noiseTypes{k}),2), ...
        nanstd(SNRstruct.(noiseTypes{k}),[],2), ...
        'color',cmp(k,:));
end
xlim(sigVarv([1 end])); ylim([0 0.25])
xlabel('RMS amp.'); ylabel('C_1')
legend(noiseTypes)




%% Generate Figure 7 (spectral matching)

% NOTE: to run this cell you must download the Chronux software and place
% it in your MATLAB path. It is freely available at chronux.org as of this
% code's development (June 2014). Chronux is used to generate the power
% spectra.
%
% Figure 8 can be generated using the same procedure for generating spectra
% from the noise time-series.



sigVarv = [0:0.5:4 5:2:20];         % noise intensities vector
% parameters used for figure: sigVarv = [0:0.25:15.5 16:1:29 30:2:50];
numSigs = length(sigVarv);          % how many intensites we consider
numReps = 1;                        % number of repetitions for each intensity (n=50 for figure)
t0 = 0;             % start time
tf = 2075;          % stop time
dt = 0.025;         % time incrememnt
t = t0:dt:tf;       % vector of simulation times
K = length(t);

% input signal
p.f = 1/35; p.dc = 0; p.duty = 45; p.i = 163; p.A = 6.5;
Ia = PhysEUSburst(t,p);
Ia(1:find(t>=60,1,'first'))=0;  % allow model to reach SS

% data storage
SNRstruct = [];

% list of perturbation types
noiseTypes = {'OU','BPHHstoc','SpecMatch','TSmimic',};   
for nn = 1:length(noiseTypes)
    % which noise
    curType = noiseTypes{nn};
    SNRstruct.(curType) = ones(numSigs,numReps)*nan;  % store results
    
for rr = 1:numSigs
    sigVar = sigVarv(rr);   % noise intensity increments
    % --- run the simulation numReps times ---
    for kk=1:numReps
        
        % generate noise
        switch curType
            case 'OU'
                % === OU noise ===
                preNoise = GenerateOU(K,dt,0,sigVar,0.5);
                
                if sigVar==1, PN{1}=preNoise; end % save this time-series when sigVar=1
                
            case 'BPHHstoc'
                % === stochastic biphsic noise for the HH neuron ===
                aw=0.15; bt=7.29; bw=1.37; amp = ((bw+aw)/(bt+bw))^(-1/2)*sigVar;
                preNoise = VariablePulseTrain2(tf/dt+1,aw/dt,bw/dt,bt/dt)*amp;
                
                if sigVar==9, PN{2}=preNoise; end % save this time-series when sigVar=9
                
            case 'SpecMatch'
                % === parameters for middle circle (fig 3) matched noise ===
                aw=0.15; bt=7.29; bw=1.37; amp = ((bw+aw)/(bt+bw))^(-1/2)*sigVar;
                pnt = VariablePulseTrain2(tf/dt+1,aw/dt,bw/dt,bt/dt)*amp;
                
                % multitaper spectral smoothing of noise
                T=0.1; W=100;
                params.Fs = 1/(dt/1e3);
                params.tapers = [T*W 2*T*W-1];
                params.pad = 0;
                params.fpass = [0 5e3];                 % bandpass
                [Sx,f]=mtspectrumc(pnt,params);
                
                % match the gaussian noise to the pulse noise
                pntFFT = fft(pnt)';                             % FFT of target signal
                Np = floor((length(pntFFT)-1)/2);               % number of points
                phases = rand(Np,1)*2*pi;                       % generate random phase values
                phases = complex(cos(phases),sin(phases));      % partition phases into real and complex parts
                pntFFT(2:Np+1) = pntFFT(2:Np+1).*phases;
                pntFFT(end:-1:end-Np+1) = conj(pntFFT(2:Np+1)); % conjugate trailing end of FFT coefficients
                preNoise = real(ifft(pntFFT,[],1))';            % revert to time domain
                
                if sigVar==9, PN{3}=preNoise; end % save this time-series when sigVar=9
                
            case 'TSmimic'
                % === OU noise with low correlation rate ===
                ouSR = GenerateOU(K,dt,0,sigVar,0.5);
                
                % === parameters for middle circle (fig 3) matched noise ===
                aw=0.15; bt=7.29; bw=1.37; amp = ((bw+aw)/(bt+bw))^(-1/2)*sigVar;
                bpSR = VariablePulseTrain2(tf/dt+1,aw/dt,bw/dt,bt/dt)*amp;
                
                % mimicry algorithm
                preNoise = mimicry(bpSR,ouSR);
                
                if sigVar==1, PN{4}=preNoise; end % save this time-series when sigVar=1
        end
        
        % --- integrate model ---
        [u, uSpike, ix, SNR] = SimNeuron(@HHneuron,t,[0 0.05 0.32 0.60 0]',Ia,preNoise,50);
        % store data
        SNRstruct.(curType)(rr,kk) = SNR(:,2);
    end
    
    % progress report:
    fprintf('Simulation Progress: type - %3.2g, repitition - %3.2g\n',...
        nn/length(noiseTypes),rr/numSigs)
end  
end

% plot results
cmp = hsv(length(noiseTypes));
figure; hold on
for k=1:length(noiseTypes)
    errorbar(sigVarv,nanmean(SNRstruct.(noiseTypes{k}),2), ...
        nanstd(SNRstruct.(noiseTypes{k}),[],2), ...
        'color',cmp(k,:));
end
xlim(sigVarv([1 end])); ylim([0 0.25])
xlabel('RMS amp.'); ylabel('C_1')
legend(noiseTypes)


% also plot time-series spectra
figure
for k=1:length(noiseTypes)
    subplot(4,1,k)
    
    % parameters:
    T=2.0; W=10;                                % Time-Bandwidth
    params.Fs = 1/(dt/1e3);                     % sample rate
    params.fpass = [0 2.5e3];                   % bandpass
    params.tapers = [T*W 2*T*W-1];              % tapers
    params.pad = 0;                             % padding
    % spectral comparison
    [S, f] = mtspectrumc(PN{k},params);
    
    plot(f,S); xlabel('frequency (Hz)'); ylabel('power')
    title(noiseTypes{k})
end





%% Generate Figure 9 (behavioral impact estimates)

t0 = 0;             % start time
tf = 500;           % stop time
dt = 0.025;         % time incrememnt
t = t0:dt:tf;       % vector of simulation times
K = length(t);      % number of total iterations in simulation

% input signal
p.f = 1/35; p.dc = 0; p.duty = 45; p.i = 163; p.A = 6.5;
Ia = PhysEUSburst(t,p);
Ia(1:find(t>=60,1,'first'))=0;  % allow model to reach SS

% ko = 50;    numReps = 20; % <-- (values used in manuscrpt)
ko = 20; numReps = 8;
% parameter combinations for biphasic pulse noise
parCombs = [linspace(0.15,0.15,ko)',...
            linspace(0.8,1.84,ko)',...
            linspace(7.29,7.29,ko)',...
            linspace(2,14,ko)'];

% data stored in the following format: [aw bw bt RMS]
SNResm = zeros(20e3,10)-1;   % results storage

hw = waitbar(0,'progress');
for pc=1:ko             % loop over each parameter combination
for reps=1:numReps      % repetition at each parameter combination
    % alocate noise parameters
    aw=parCombs(pc,1); bw=parCombs(pc,2);
    bt=parCombs(pc,3); sigVar=parCombs(pc,4);
    amp = ((bw+aw)/(bt+bw))^(-1/2)*sigVar;
    % generate biphasic pulse perturbations
    preNoise = VariablePulseTrain2(tf/dt+1,aw/dt,bw/dt,bt/dt)*amp;
    
    % --- integrate model ---
    [u, uSpike, ix, SNR] = SimNeuron(@HHneuron,t,[0 0.05 0.32 0.60 0]',Ia,preNoise,50);
    
    
    % calculate bins and firing rates (FRs) in each bin
    bins = [1 find(diff(Ia-mean(Ia)>0)) K];     % bins
    binSoln = (-1).^(1:length(bins)-1)>0;       % 0/1 is there an event in each bin? (assuming alternating 0s & 1s for each bin)
    % loop over bins and count spikes
    spb = zeros(1,length(bins)-1)-1;
    FRpb = spb;
    pctCor = spb;
    for n=1:length(bins)-1
        % spikes per bin
        if isempty(ix)
            spb(n)=0;
        else
            spb(n) = sum( ix(:,1)>bins(n) & ix(:,1)<bins(n+1) );
        end
        % calculate average firing rate per bin
        FRpb(n) = spb(n)/(diff(t(bins([n n+1])))/1e3);
        
        % percent correct for this trial
        % enforce: cannot exceede 1 (100% id as pulse or 0 (never id as pulse)
        if binSoln(n)
            pctCor(n) = max(min(  FRpb(n)/67.021  ,1),0);
        else
            pctCor(n) = 1 - max(min(  FRpb(n)/67.021  ,1),0);
        end
    end
    
    % store results
    if isnan(SNR(2)), C1=0; else C1=SNR(2); end     % call no spikes zero rSNR
    ind0 = find(SNResm==-1,1,'first');
    SNResm(ind0,:) = [mean(spb) mean(FRpb) mean(pctCor) C1 ...
        mean(FRpb(1:2:end)) mean(FRpb(2:2:end-1)) parCombs(pc,:)];
end
    waitbar(pc/ko,hw)
end
close(hw)

% remove memory placeholders
SNResm = SNResm(1:find(SNResm(:,1)==-1,1,'first')-1,:);

% get grouped mean and std of our results
% Format: [spb FRpb pctCor SNR FRoff FRon parCombs(1:4)]
[unk,tmp,subs] = unique(SNResm(:,end),'rows');
dataLables = {'spbMean' 'spbStd'; 'frMean' 'frStd'; 'corMean' 'corStd'; 'rsnrMean' 'rsnrStd'; ... 
    'FRoffMean' 'FRoffstd'; 'FRonMean' 'FRonstd'};
% compile trial means
for ii=1:length(dataLables)
    totals.(dataLables{ii,1}) = accumarray(subs,SNResm(:,ii),[],@mean,nan);
    totals.(dataLables{ii,2}) = accumarray(subs,SNResm(:,ii),[],@std,nan);
end



% Calculate various outcomes:
% max C1 and its index
[v, ix] = max(totals.rsnrMean);     
% index of the first point to drop 0.02 (from fig 5) prior to ix
ix02 = find(totals.rsnrMean(1:ix-1)<=(v-0.02),1,'last');  
% value at this point
v02 = totals.rsnrMean(ix02);
% range of lost percent correct:
disp(['Lost range: ' num2str(( totals.corMean(ix)-totals.corMean(ix02) )/(max(totals.corMean)-0.5))])

% plot outcomes
figure
subplot(2,2,2); hold on
errorbar(unk,totals.FRoffMean,totals.FRoffstd)
errorbar(unk,totals.FRonMean, totals.FRonstd)
line(unk(ix)*[1 1],[0 70],'color','k')
xlim([min(unk) 14]); ylim([0 70]); ylabel('FR on & off'); xlabel('E[A_{rms}]')
subplot(2,2,1)
errorbar(unk,totals.corMean*100,totals.corStd*100)
line([unk(ix) unk(ix) nan unk(ix02) unk(ix02)],[0 90 nan 0 90],'color','k')
xlim([min(unk) 14]); ylim([50 90]); ylabel('%Cor')
subplot(2,2,3)
errorbar(unk,totals.rsnrMean,totals.rsnrStd)
line([unk(ix) unk(ix) nan unk(ix02) unk(ix02)],[0 0.25 nan 0 0.25],'color','k')
xlim([min(unk) 14]); ylim([0 0.25]); ylabel('C1')





%% Generate Supplemental Figure (FitzHugh-Nagumo model)
dataFlag = 0;

if dataFlag    
    
load FN_OU_CorTimeSearch
sigVec = unique(C(:,1))';
thVec = unique(C(:,2))';

else

% Threshold measures for FN model:
% individual simulation run parameters
t0 = 0;             % start time
tf = 2075;          % stop time
dt = 0.025;         % time incrememnt
t = t0:dt:tf;       % vector of simulation times
% list of amps to test
ampVec = 0.012:0.001:0.025;
% hold data
C = zeros(1,length(ampVec));    

hw = waitbar(0,'progress');
for k=1:length(ampVec)
    % input signal
    p.A = ampVec(k);
    p.f = 1/35; p.dc = 0; p.duty = 45; p.i = 163;
    Ia = PhysEUSburst(t,p);
    Ia(1:find(t>=60,1,'first'))=0;
    % noise
    preNoise = zeros(1,length(t));
    
    % --- integrate model ---
    [u, uSpike, ix, SNR] = SimNeuron(@FNneuron,t,[0 1 0 0 0]',Ia,preNoise,0.8);
    C(k) = SNR(2);
    waitbar(k/length(ampVec),hw)
end
close(hw)


C(isnan(C))=0;     % define no spikes as C=0;
figure
plot(ampVec,C,'-ko','markersize',10,'markerfacecolor','k','linewidth',3)
xlabel('RMS amp.'); ylabel('C_1')
% also plot the last run
figure
[AX,H1,H2] = plotyy(t,Ia,t,u(:,1)); hold(AX(2),'on')
plot(AX(2),t(ix(:,1)),ones(1,size(ix,1))*0.8,'kx','linewidth',2)


% Play with noise correlation rates for FN model:

% individual simulation run parameters
t0 = 0;             % start time
tf = 2075;          % stop time
dt = 0.025;         % time incrememnt
t = t0:dt:tf;       % vector of simulation times
% noise intensities
sigVec = [linspace(0,0.07,40) linspace(0.08,0.2,10)];
% noise correlation rates
thVec = sort([logspace(log10(0.005),log10(5),8) 0.008 0.02 0.07]);
% repetitions for each intensity
reps = 20;
% hold data
C = zeros(20e3,3)*-1;    

hw = waitbar(0,'progress');
cc=1;
for k=1:length(sigVec)
for m=1:length(thVec)
for j=1:reps
    % input signal
    p.A = 0.015;    % subthreshold signal
    p.f = 1/35; p.dc = 0; p.duty = 45; p.i = 163;
    Ia = PhysEUSburst(t,p);
    Ia(1:find(t>=60,1,'first'))=0;
    % noise
    preNoise = GenerateOU(length(t),dt,0,sigVec(k),thVec(m));
    % fprintf('E[Arms]=%g, Arms=%g\n',sigVec(k),rms(preNoise))        % be sure we get the desired process RMS
    
    % --- integrate model ---
    [u, uSpike, ix, SNR] = SimNeuron(@FNneuron,t,[0 1 0 0 0]',Ia,preNoise,0.8);
    C(cc,:) = [sigVec(k) thVec(m) SNR(2)];      % store data
    
    cc=cc+1;    % track steps
end
end
    waitbar(k/length(sigVec),hw)
end
close(hw)
C = C(1:cc-1,:);    % dump extra variable space
C(isnan(C))=0;      % define no spikes (which will appear as NaNs) as C=0

end

% Average over all repetitions based on parameter values
tmp = bsxfun(@eq,C(:,1),sigVec);    % matix of indicies of where each entry matches sigVec
[rowSig, col]=find(tmp');           % indicies into C (transpose tmp because find looks down rows first)
tmp = bsxfun(@eq,C(:,2),thVec);
[rowTh, col]=find(tmp');
Amean = accumarray([rowTh rowSig],C(:,3),[],@mean);     % accumulate averages into A (dimensions: theta x sigma)

% plot
figure
subplot(2,1,1)
set(gcf,'DefaultAxesColorOrder',jet(size(Amean,1)))
plot(sigVec,Amean','linewidth',2)
legend(arrayfun(@(u) num2str(u,3),thVec,'unif',false))
xlabel('RMS amp.'); ylabel('C_1')
title('FN Model Simulations: inset shows noise correlation rate (r_c)')
xlim([0 0.12])
subplot(2,1,2)
plot(thVec,max(Amean,[],2),'ob-','markerfacecolor','b');
set(gca,'xscale','log'); xlim([min(thVec) max(thVec)])
xlabel('Correlation Rate'); ylabel('max[C_1]')

