function logBeta = cohmmBackward(cohmm,data)
%
% Long Le <longle1@illinois.edu>
% University of Illinois
%

N = numel(cohmm.pi); % number of states
T = size(data,2); % number of observations

% init
logBeta = zeros(N,1);

% induction
for t = T:-1:1
    logBetaNew = zeros(N,1);
    for k = 1:N
        B = zeros(N,1);
        for l = 1:N
            B(l) = log(cohmm.B(l,data(:,t)));
        end
        logBetaNew(k) = logSumExp(logBeta+log(cohmm.A(:,k))+B);
    end
    logBeta = logBetaNew;
end