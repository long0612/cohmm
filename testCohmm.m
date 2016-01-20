% test cohmm functions
%
% Long Le <longle1@illinois.edu>
% University of Illinois
%

clear all; close all;

addpath(genpath('../voicebox/'))
rng('default');

%% Sanity check with discrete observation
TRANS = [.9 .1; .5 .95];
EMIS = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6;...
7/12, 1/12, 1/12, 1/12, 1/12, 1/12];
[seq,states] = hmmgenerate(1000,TRANS,EMIS);

% ============== 
%test BaumWelch alg
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
%test Viterbi alg
% ============== 

%% Test with continuous observation
[y,fs] = audioread('../network-paper/genCascade/data/GCW/GCW-A-(17).wav');
featMFCC = melcepst(y,fs,'Mtaz',3)';

cohmm.pi = [1;0;0;0];
cohmm.A = [0.95 0.05 0 0; 0 0.9 0.1 0; 0 0 0.7 0.3; 0 0 0 1];
cohmm.B = @(k,feat) ...
    (k==1)*mvnpdf(feat,zeros(3,1),eye(3)*6)+...
    (k==2)*mvnpdf(feat,zeros(3,1),eye(3)*1.3)+...
    (k==3)*mvnpdf(feat,zeros(3,1),eye(3)*4)+...
    (k==4)*mvnpdf(feat,zeros(3,1),eye(3)*0.8);

newCohmm2 = cohmmBaumWelch(cohmm,featMFCC);
logProb2 = cohmmForwBack(newCohmm2,featMFCC);
