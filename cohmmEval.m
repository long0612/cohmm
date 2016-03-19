function logProb = cohmmEval(cohmm,states,data)
% logProb = cohmmEval(cohmm,states,data)
% 
% Evaluate the log prob of a particular realization of state and data
%
% Long Le <longle1@illinois.edu>
% University of Illinois
%

T = size(data,2); % number of observations

% init
logProbAll = zeros(1,T);
logProbAll(1) = log(cohmm.pi(states(1)))+log(cohmm.B(states(1),data(:,1)));

% recursion
for t = 1:T-1
    logProbAll(t+1) = logProbAll(t)+log(cohmm.A(states(t),states(t+1)))+log(cohmm.B(states(t+1),data(:,t+1)));
end

logProb = logProbAll(T);
