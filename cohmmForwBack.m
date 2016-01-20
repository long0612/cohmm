function [logProb,logAlpha,logBeta] = cohmmForwBack(cohmm,data)
% [logProb,logAlpha,logBeta] = cohmmForwBack(cohmm,data)
% 
% Forward-backward algorithm for continuous observation HMM.
%
% Long Le <longle1@illinois.edu>
% University of Illinois
%

T = size(data,2); % number of observations

logAlpha = cohmmForward(cohmm,data);
logBeta = cohmmBackward(cohmm,data);
logProb = logSumExp(logAlpha(:,T));



