function Y = HH_neuronKs(t, X, Ia)

gna = 229; ena = 45; gk = 14; ek = -97; gca = 65; eca = 61; gl = 1/11; el = -29; c = 5.5;  
V = X(1); M = X(2); N = X(3); H = X(4); R = X(5); F = X(6); 


Y(1) = (1/c)*(Ia + gna*M*M*M*H*(ena-V) + gk*(N^4)*(ek-V) + gl*(el-V) + gca*R*F*(eca-V));
Y(2) = (mi(V) - M)/tm(V);
Y(3) = (ni(V) - N)/tn(V);
Y(4) = (hi(V) - H)/th(V);
Y(5) = (ri(V) - R)/tr(V);
Y(6) = (fi(V) - F)/tf(V); 
Y = Y';
 
      

function y = mi(V)
y = 1./(1 + exp(-(V+32.5)./7.9));
end
function y = tm(V)
y = exp(-(V+286)./160);
end

function y = hi(V)
y = 1./(1 + exp((V+62)./5.5));
end
function y = th(V)
y = 0.51+exp(-(V+26.6)./7.1);
end

function y = ni(V)
y = 1./(1 + exp(-(V-14)./17)).^(0.25);
end
function y = tn(V)
y = exp(-(V-67)./68);
end

function y = ri(V)
y = 1./(1 + exp(-(V+25)./7.5));
end
function y = tr(V)
y = 3.1;
end

function y = fi(V)
y = 1./(1 + exp((V+260)./65));
end
function y = tf(V)
y = exp(-(V-444)./220);
end

    
end 
%[T, X] = ode45(@HH_neuron, 0:0.01:100, [-80 0.34 0.54 0.045 0.01 0.04]);
%[T, X] = ode45(@HH, 0:0.01:100, [10 0.5 0.5 0.5]);
