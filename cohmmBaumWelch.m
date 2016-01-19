function newCohmm = cohmmBaumWelch(cohmm, data)
% cohmm = cohmmBaumWelch(cohmm, data)
% data - DxN
% 
% BaumWelch algorithm for continuous observation HMM.
%
% Long Le <longle1@illinois.edu>
% University of Illinois
%

N = numel(cohmm.pi); % number of states
T = size(data,2); % number of observations

logAlpha = zeros(N,T);
for t = 1:T-1
    logAlpha(:,t) = cohmmForward(cohmm,data(:,1:t));
end
logBeta = zeros(N,T);
for t = T:-1:2
    logBeta(:,t) = cohmmBackward(cohmm,data(:,t+1:T));
end

logEta = zeros(N,N,T-1);
for t = 1:T-1
    for k = 1:N
        for l = 1:N
            logEta(k,l,t) = logAlpha(k,t)+log(cohmm.A(k,l))+log(cohmm.B(l,data(:,t+1)))+logBeta(l,t+1);
        end
    end
    logEta(:,:,t) = logEta(:,:,t)-logSumExp(logEta(:,:,t));
end

logGamma = zeros(N,T-1);
for t = 1:T-1
    for k = 1:N
        logGamma(k,t) = logSumExp(logEta(k,:,t));
    end
end

newCohmm.pi = exp(logGamma(:,1));
for k = 1:N
    for l = 1:N
        newCohmm.A(k,l) = exp(logSumExp(logEta(k,l,:)) - logSumExp(logGamma(k,:)));
    end
end
newCohmm.B = cohmm.B;