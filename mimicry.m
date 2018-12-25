function z = mimicry(x,y) 
% z = mimicry(x,y) 
% x, y: 2 real vectors of length T 
% z: real vector of length T in which the elements of x 
% occur in the same rank order as the elements of y 
%
%
% Cohen JE, Newman CM, Cohen AE, Petchey OL, & Gonzalez A (1999) Spectral
%   mimicry: A method of synthesizing matching time series with different
%   Fourier spectra. Circuits Systems and Signal Processing 18(4):431-442.
%


[xsort,xindex] = sort(x); 
[ysort,yindex] = sort(y); 
[zsort,zindex] = sort(yindex); 
z = xsort(zindex); 
