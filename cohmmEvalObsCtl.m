function logProb = cohmmEvalObsCtl(cohmm,states,data)
% logProb = cohmmEvalObsCtl(cohmm,states,data)
% 
% Evaluate the log prob of a particular realization of state and data, with
% controlled observation
%
% Long Le <longle1@illinois.edu>
% University of Illinois
%

T = size(data,2); % number of observations

% init
logProbAll = zeros(1,T);
bMax = max( cohmm.B(states(1),data{1}) );
logProbAll(1) = log(cohmm.pi(states(1)))+log(bMax);

% recursion
for t = 1:T-1
    bMax = max( cohmm.B(states(t+1),data{t+1}) );
    logProbAll(t+1) = logProbAll(t)+log(cohmm.A(states(t),states(t+1)))+log(bMax);
end

logProb = logProbAll(T);
