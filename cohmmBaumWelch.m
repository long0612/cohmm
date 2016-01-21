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
E = numel(data);
nIter = 0;
maxIter = 500;
tol = 1e-6;
while nIter < maxIter
    nIter = nIter + 1;
    
    % ================
    % Compute alpha, beta, eta, gamma
    % ================
    logAlpha = cell(1,E);
    logBeta = cell(1,E);
    logEta = cell(1,E);
    logGamma = cell(1,E);
    for j = 1:E
        T = size(data{j},2); % number of observations
        
        logAlpha{j} = cohmmForward(newCohmm,data{j});
        
        logBeta{j} = cohmmBackward(newCohmm,data{j});
    
        logEta{j} = zeros(N,N,T-1);
        for t = 1:T-1
            for k = 1:N
                for l = 1:N
                    logEta{j}(k,l,t) = logAlpha{j}(k,t)+log(newCohmm.A(k,l))+log(newCohmm.B(l,data{j}(:,t+1)))+logBeta{j}(l,t+1);
                end
            end
            logEta{j}(:,:,t) = logEta{j}(:,:,t)-logSumExp(logEta{j}(:,:,t));
        end

        logGamma{j} = zeros(N,T);
        for t = 1:T
            for k = 1:N
                logGamma{j}(k,t) = logAlpha{j}(k,t)+logBeta{j}(k,t);
            end
            logGamma{j}(:,t) = logGamma{j}(:,t)-logSumExp(logGamma{j}(:,t));
        end
    
    end

    % ================
    % update the model
    % ================
    prevIter = [];
    currIter = [];
    
    %newCohmm.pi = exp(logGamma(:,1));
    
    prevIter = [prevIter;newCohmm.A(:)];
    for k = 1:N
        for l = 1:N
            logEtaMul = zeros(1,E);
            logGammaMul = zeros(1,E);
            for j = 1:E
                logEtaMul(j) = logSumExp(logEta{j}(k,l,:));
                logGammaMul(j) = logSumExp(logGamma{j}(k,1:end-1));
            end
            newCohmm.A(k,l) = exp( logSumExp(logEtaMul)-logSumExp(logGammaMul) );
        end
    end
    currIter = [currIter;newCohmm.A(:)];
    
    % update observation distribution under special cases
    if isfield(newCohmm,'BType') && strcmp(newCohmm.BType, 'discrete')
        prevIter = [prevIter;newCohmm.B(:)];
        for k = 1:N
            for l = 1:size(newCohmm.B,2)
                logGammaIndMul = zeros(1,E);
                logGammaMul = zeros(1,E);
                for j = 1:E
                    logGammaIndMul(j) = logSumExp(logGamma{j}(k,:)+log(1*(data{j}==l)));
                    logGammaMul(j) = logSumExp(logGamma{j}(k,:));
                end
                newCohmm.B(k,l) = exp( logSumExp(logGammaIndMul)-logSumExp(logGammaMul) );
            end
        end
        currIter = [currIter;newCohmm.B(:)];
    % TODO: continuous Gaussian
    end
    
    if norm(currIter - prevIter) < tol
        disp('converged!')
        break;
    end
end