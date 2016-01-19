function logAlpha = cohmmForward(cohmm,data)
%
% Long Le <longle1@illinois.edu>
% University of Illinois
%

N = numel(cohmm.pi); % number of states
T = size(data,2); % number of observations

% init
logAlpha = zeros(N,1);
for k = 1:N
    logAlpha(k) = log(cohmm.pi(k))+log(cohmm.B(k,data(:,1)));
end

% induction
for t = 2:T
    for k = 1:N
        logAlpha(k) = logSumExp(logAlpha+log(cohmm.A(:,k)))+log(cohmm.B(k,data(:,t)));
    end
end