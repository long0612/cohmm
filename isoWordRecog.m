% Simple isolated word recognition
%
% Long Le <longle1@illinois.edu>
% University of Illinois
%

clear all; close all;

addpath(genpath('../voicebox/'))

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
