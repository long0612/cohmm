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

logAlpha = zeros(N,T-1);
for t = 1:T-1
    logAlpha(:,t) = cohmmForward(cohmm,data(:,1:t));
end
logBeta = zeros(N,T-1);
for t = T-1:-1:1
    logBeta(:,t) = cohmmBackward(cohmm,data(:,t+1:T));
end

eta = zeros(N,N,T-1);
for t = 1:T-1
    for k = 1:N
        for l = 1:N
            eta(k,l,t) = exp(logAlpha(k,t)+log(cohmm.A(k,l))+log(cohmm.B(l,data(:,t+1)))+logBeta(l,t+1));
        end
    end
    eta(k,l,t) = eta(k,l,t)/sum(sum(eta(:,:,t)));
end

gamma = zeros(N,T-1);
for t = 1:T-1
    for k = 1:N
        gamma(k,t) = sum(eta(k,:,t));
    end
end

newCohmm.pi = gamma(:,1);
newCohmm.A = sum(eta,3)./repmat(sum(gamma,2),1,N);
newCohmm.B = cohmm.B;