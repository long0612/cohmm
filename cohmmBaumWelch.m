function newCohmm = cohmmBaumWelch(cohmm, data)
% cohmm = cohmmBaumWelch(cohmm, data)
% data - DxN
% 
% BaumWelch algorithm for continuous observation HMM.
%
% Long Le <longle1@illinois.edu>
% University of Illinois
%

newCohmm = cohmm;

N = numel(newCohmm.pi); % number of states
T = size(data,2); % number of observations

nIter = 0;
maxIter = 500;
tol = 1e-6;
while nIter < maxIter
    nIter = nIter + 1;
    
    % ================
    % Compute alpha, beta, eta, gamma
    % ================
    logAlpha = cohmmForward(newCohmm,data);

    logBeta = cohmmBackward(newCohmm,data);

    logEta = zeros(N,N,T-1);
    for t = 1:T-1
        for k = 1:N
            for l = 1:N
                logEta(k,l,t) = logAlpha(k,t)+log(newCohmm.A(k,l))+log(newCohmm.B(l,data(:,t+1)))+logBeta(l,t+1);
            end
        end
        logEta(:,:,t) = logEta(:,:,t)-logSumExp(logEta(:,:,t));
    end

    logGamma = zeros(N,T);
    for t = 1:T
        for k = 1:N
            logGamma(k,t) = logAlpha(k,t)+logBeta(k,t);
        end
        logGamma(:,t) = logGamma(:,t)-logSumExp(logGamma(:,t));
    end

    % ================
    % update the model
    % ================
    prevVec = [newCohmm.pi;newCohmm.A(:)];
    newCohmm.pi = exp(logGamma(:,1));
    for k = 1:N
        for l = 1:N
            newCohmm.A(k,l) = exp(logSumExp(logEta(k,l,:)) - logSumExp(logGamma(k,1:T-1)));
        end
    end
    currVec = [newCohmm.pi;newCohmm.A(:)];
    
    % update observation distribution under special cases
    if isfield(newCohmm,'BType') && strcmp(newCohmm.BType, 'discrete')
        prevVec = [prevVec;newCohmm.B(:)];
        for k = 1:N
            for l = 1:size(newCohmm.B,2)
                newCohmm.B(k,l) = exp(logSumExp(logGamma(k,:)+log(1*(data==l))) - logSumExp(logGamma(k,:)));
            end
        end
        currVec = [currVec;newCohmm.B(:)];
    % TODO: continuous Gaussian
    end
    
    if norm(currVec - prevVec) < tol
        disp('converged!')
        break;
    end
end