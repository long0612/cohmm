function logProb = cohmmForwBack(cohmm,data)
% logProb = cohmmForwBack(cohmm,data)
% 
% Forward-backward algorithm for continuous observation HMM.
%
% Long Le <longle1@illinois.edu>
% University of Illinois
%

logAlpha = cohmmForward(cohmm,data);
logProb = logSumExp(logAlpha);



