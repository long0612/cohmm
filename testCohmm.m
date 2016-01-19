% test cohmm functions
%
% Long Le <longle1@illinois.edu>
% University of Illinois
%

clear all; close all;

addpath(genpath('../voicebox/'))
[y,fs] = audioread('../network-paper/genCascade/data/GCW/GCW-A-(17).wav');
featMFCC = melcepst(y,fs,'Mtaz',16)';

cohmm.pi = [1;0;0;0];
cohmm.A = [0.7 0.3 0 0; 0 0.7 0.3 0; 0 0 0.7 0.3; 0 0 0 1];
cohmm.B = @(k,feat) ...
    (k==1)*mvnpdf(feat,zeros(16,1),eye(16)*5e-4)+...
    (k==2)*mvnpdf(feat,zeros(16,1),eye(16)*3e-3)+...
    (k==3)*mvnpdf(feat,zeros(16,1),eye(16)*3e-3)+...
    (k==4)*mvnpdf(feat,zeros(16,1),eye(16)*3e-6);

newCohmm = cohmmBaumWelch(cohmm,featMFCC);
logProb = cohmmForwBack(newCohmm,featMFCC);
