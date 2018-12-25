function vpt = VariablePulseTrain2(tf,aw,bw,bt)
% Returns a biphasic pulse train with variable width and period
%
% tf = number of samples
% aw = min pulse width
% bw = max pulse width
% bt = max ipi
%
%
% vpt = VariablePulseTrain2(tf,aw,bw,bt)
%
%
% Z. Danziger June 2014
%

% round because each parameter represents an index
tmp=floor([aw bw bt tf]);
aw=tmp(1); bw=tmp(2); bt=tmp(3); tf=tmp(4);

% do inputs make sense?
if any( diff([aw bw bt tf])<=0 ) || any([aw bw bt tf]<=0)
    error('inputs must respect: tf>bt>bw>ba>0')
end

% find all pulse locations
pulseLocs = cumsum(round( bw+(bt-bw)*rand(1,round(tf/bw)) ));
widthSize = round( (aw+(bw-aw)*rand(1,length(pulseLocs)))/2 );

% loop through and create the pulse train
vpt = zeros(1,tf+bt+bw);
for ii=1:length(pulseLocs)
    % fill in pulses to timeseries
    ixU = pulseLocs(ii):(widthSize(ii)+pulseLocs(ii))-1;
    ixD = ixU+widthSize(ii)+1-1;
    vpt(ixU) =  1;
    vpt(ixD) = -1;
    
    if any(ixD>tf)
        break
    end
end
    

% truncate padding
vpt = vpt(1:tf);






