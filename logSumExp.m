function out = logSumExp(x)
% out = logSumExp(x)
% Compute log(sum_(exp(x_i)))
%
% Long Le <longle1@illinois.edu>
% University of Illinois
%

x = x(:);
maxX = max(x);
out = maxX + log(sum(exp(x-maxX)));