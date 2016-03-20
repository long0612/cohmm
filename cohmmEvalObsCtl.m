function nLogProb = cohmmEvalObsCtl(cohmm,data,states,mode)
% nLogProb = cohmmEvalObsCtl(cohmm,data,states,mode)
% 
% Evaluate the normalized log prob of a particular realization of state and data, with
% controlled observation. If mode = 1, count no-obs in the normalization
%
% Long Le <longle1@illinois.edu>
% University of Illinois
%

T = size(data,2); % number of observations
cnt = 0;

% init
logProbAll = zeros(1,T);
if numel(data{1}) == 0
    bMax = cohmm.B0(states(1)); % no observation prob
    if mode == 1
        cnt = cnt+1;
    end
else
    bMax = max( cohmm.B(states(1),data{1}) );
    cnt = cnt+1;
end
logProbAll(1) = log(cohmm.pi(states(1)))+log(bMax);

% recursion
for t = 1:T-1
    if numel(data{t+1}) == 0
        if mode == 1
            cnt = cnt+1;
        end
        bMax = cohmm.B0(states(t+1));
    else
        bMax = max( cohmm.B(states(t+1),data{t+1}) );
        cnt = cnt+1;
    end
    logProbAll(t+1) = logProbAll(t)+log(cohmm.A(states(t),states(t+1)))+log(bMax);
end

nLogProb = logProbAll(T)/cnt;
