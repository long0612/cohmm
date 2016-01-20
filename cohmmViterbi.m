function state = cohmmViterbi(cohmm, data)
% state = cohmmViterbi(cohmm, data)
% 
% Viterbi algorithm for continuous observation HMM.
%
% Long Le <longle1@illinois.edu>
% University of Illinois
%

N = numel(cohmm.pi); % number of states
T = size(data,2); % number of observations

