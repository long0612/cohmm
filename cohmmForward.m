function logAlpha = cohmmForward(cohmm,data)
%
% Long Le <longle1@illinois.edu>
% University of Illinois
%

N = numel(cohmm.pi); % number of states
T = size(data,2); % number of observations

% init
logAlpha = zeros(N,T);
for k = 1:N
    logAlpha(k,1) = log(cohmm.pi(k))+log(cohmm.B(k,data(:,1)));
end

% induction
for t = 1:T-1
    for k = 1:N
        logAlpha(k,t+1) = logSumExp(logAlpha(k,t)+log(cohmm.A(:,k)))+log(cohmm.B(k,data(:,t+1)));
    end
end