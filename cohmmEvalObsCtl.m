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
if numel(data{1}) == 0
    bMax = cohmm.B0(states(1)); % no observation prob
else
    bMax = max( cohmm.B(states(1),data{1}) );
end
logProbAll(1) = log(cohmm.pi(states(1)))+log(bMax);

% recursion
for t = 1:T-1
    if numel(data{t+1}) == 0
        bMax = cohmm.B0(states(t+1));
    else
        bMax = max( cohmm.B(states(t+1),data{t+1}) );
    end
    logProbAll(t+1) = logProbAll(t)+log(cohmm.A(states(t),states(t+1)))+log(bMax);
end

logProb = logProbAll(T);
