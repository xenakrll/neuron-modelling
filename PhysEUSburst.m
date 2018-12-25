function r = PhysEUSburst(t,p)
% physiological bursting pattern for EUS muscle
%
% corner spikes appear at t=2e-5 resolution


% Peng 2006 phasic bursting parameters
% AP=65.1ms, SP=98.1ms, -> p.i=163.2ms, p.duty=65.1/163.2=40%


w = p.i*p.duty/100; % pulse width: period times duty cycle
% square wave
r1 = pulstran(t,t(1):p.i:t(end),'rectpuls',w);  

% rise to square
rt = round(w*0.25); %12;    % rise time
r2 = pulstran(t,(t(1):p.i:t(end))-(w/2+rt/2),'tripuls',rt,1);

% fall to baseline
r3 = pulstran(t,(t(1):p.i:t(end))+(w/2+rt/2),'tripuls',rt,-1);

% compile all traces
r = p.A*(r1+r2+r3)+p.dc;

% clip blips
r(r>p.A+p.dc) = p.A+p.dc;









% === old code ===
% % paramater that determines the fraction of the pulse width in time that the
% % signial will rise to max and fall to min
% risefall = 0.25; %0.12; %
% 
% % square pulse
% r1 = p.A*0.5*(square(t*2*pi/p.i,p.duty)+1)+p.dc;
% 
% % rise signal
% width = round( (p.i*p.duty/100)*risefall );
% trigr = 0:p.i:t(end)+p.i;
% r2 = p.A*pulstran((t+width/2),trigr,'tripuls',width,1);
% 
% % fall signal
% r3 = p.A*pulstran((t-p.i*p.duty/100-width/2),trigr,'tripuls',width,-1);
% 
% % combine all signals
% r = ~(r1>0).*r2 + ~(r1>0).*r3 + r1;
% 
% 
% % hack to keep the signal within bounds and to avoid spikes at corner
% % currents
% if p.A<0
%     r( r<p.A ) = p.A;
% elseif p.A>0
%     r( r>p.A ) = p.A;
% end

