function states = cohmmViterbi(cohmm, data)
% state = cohmmViterbi(cohmm, data)
% 
% Viterbi algorithm for continuous observation HMM.
%
% Long Le <longle1@illinois.edu>
% University of Illinois
%

N = numel(cohmm.pi); % number of states
T = size(data,2); % number of observations

% init
logDelta = zeros(N,T);
for k = 1:N
    logDelta(k,1) = log(cohmm.pi(k))+log(cohmm.B(k,data(:,1)));
end
psi = zeros(N,T);

% recursion
for t = 1:T-1
    for k = 1:N
        [logDelta(k,t+1),psi(k,t+1)] = max( logDelta(:,t)+log(cohmm.A(:,k))+log(cohmm.B(k,data(:,t+1))) );
    end
end

% termination
[~,idx] = max(logDelta(:,T));

% backtracking
states(T) = idx;
for t = T-1:-1:1
    states(t) = psi(states(t+1),t+1);
end