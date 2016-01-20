% Sanity check with discrete observation
%
% Long Le <longle1@illinois.edu>
% University of Illinois
%

clear all; close all;

addpath(genpath('../voicebox/'))
rng('default');

% generate hmm seq
TRANS = [.9 .1; .5 .95];
EMIS = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6;...
7/12, 1/12, 1/12, 1/12, 1/12, 1/12];
[seq,states] = hmmgenerate(1000,TRANS,EMIS);

% ============== 
% test BaumWelch alg
% ============== 
TRANS_GUESS = [.85 .15; .1 .9];
EMIS_GUESS = [.17 .16 .17 .16 .17 .17;.6 .08 .08 .08 .08 .08];
[TRANS_EST, EMIS_EST] = hmmtrain(seq, TRANS_GUESS, EMIS_GUESS);

cohmm.pi = [1; 0];
cohmm.A = TRANS_GUESS;
cohmm.B = EMIS_GUESS;
%cohmm.B = EMIS;
newCohmm = cohmmBaumWelch(cohmm,seq);

% ==============
% test forward-backward alg
% ==============
PSTATES = hmmdecode(seq,TRANS,EMIS);

cohmm.pi = [1; 0];
cohmm.A = TRANS;
cohmm.B = EMIS;
N = numel(cohmm.pi);
T = size(seq,2); 
[logProb,logAlpha,logBeta] = cohmmForwBack(cohmm,seq);
mPstates = zeros(N,T);
for k = 1:T
	mPstates(:,k) = exp((logAlpha(:,k)+logBeta(:,k))-logSumExp(logAlpha(:,k)+logBeta(:,k)));
end
figure; hold on; plot(PSTATES(1,:),'b'); plot(mPstates(1,:),'r')

% ============== 
% test Viterbi alg
% ============== 
likelystates = hmmviterbi(seq, TRANS, EMIS);
mean(states==likelystates)

cohmm.pi = [1; 0];
cohmm.A = TRANS;
cohmm.B = EMIS;
mEstStates = cohmmViterbi(cohmm, seq);
mean(states==mEstStates)

figure; hold on; plot(likelystates,'b'); plot(mEstStates,'r')