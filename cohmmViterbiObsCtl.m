function [states,maxLogDelta] = cohmmViterbiObsCtl(cohmm, data)
% state = cohmmViterbi(cohmm, data)
% 
% Viterbi algorithm for discrete observations with varying dimension HMM.
%
% Long Le <longle1@illinois.edu>
% University of Illinois
%

N = numel(cohmm.pi); % number of states
T = numel(data); % number of observations

% init
logDelta = zeros(N,T);
for k = 1:N
    bMax = max( cohmm.B(k,data{1}) );
    logDelta(k,1) = log(cohmm.pi(k))+log(bMax);
end
psi = zeros(N,T);

% recursion
for t = 1:T-1
    for k = 1:N
        bMax = max( cohmm.B(k,data{t+1}) );
        [logDelta(k,t+1),psi(k,t+1)] = max( logDelta(:,t)+log(cohmm.A(:,k))+log(bMax) );
    end
end

% termination
[maxLogDelta,idx] = max(logDelta(:,T));

% backtracking
states(T) = idx;
for t = T-1:-1:1
    states(t) = psi(states(t+1),t+1);
end