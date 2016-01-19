% test cohmm functions
%
% Long Le <longle1@illinois.edu>
% University of Illinois
%

clear all; close all;

addpath(genpath('../voicebox/'))

%% Sanity check with discrete observation
TRANS = [.9 .1; .05 .95;];

EMIS = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6;...
7/12, 1/12, 1/12, 1/12, 1/12, 1/12];

[seq,states] = hmmgenerate(1000,TRANS,EMIS);

TRANS_GUESS = [.85 .15; .1 .9];
EMIS_GUESS = [.17 .16 .17 .16 .17 .17;.6 .08 .08 .08 .08 08];
[TRANS_EST2, EMIS_EST2] = hmmtrain(seq, TRANS_GUESS, EMIS_GUESS);
PSTATES = hmmdecode(seq,TRANS,EMIS);

cohmm.pi = [1; 0];
cohmm.A = [.85 .15; .1 .9];
cohmm.B = @(k,l) ...
    (k==1)*1/6+...
    (k==2)*(l==1)*7/12+...
    (k==2)*(l~=1)*1/12;

newCohmm = cohmmBaumWelch(cohmm,seq);
logProb = cohmmForwBack(newCohmm,seq);

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

newCohmm = cohmmBaumWelch(cohmm,featMFCC);
logProb = cohmmForwBack(newCohmm,featMFCC);
