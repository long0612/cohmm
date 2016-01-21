% Simple isolated word recognition
%
% Long Le <longle1@illinois.edu>
% University of Illinois
%

clear all; close all;

addpath(genpath('../voicebox/'))
addpath(genpath('../node-paper/'))

%% inspect data for parameters
[y,fs] = audioread('../network-paper/genCascade/data/GCW/GCW-A-(17).wav');
%figure; plot([1:size(y,1)]/fs,abs(y));
blockSize=256;
[S,tt,ff] = mSpectrogram(y,fs,blockSize);
figure; imagesc(tt,ff,S); axis xy
frameSize = 2^floor(log2(0.03*fs));
featMFCC = melcepst(y,fs,'Mtaz',3,floor(3*log(fs)),frameSize)';
%figure; imagesc([1:size(featMFCC,2)]*frameSize/2/fs,[1:size(featMFCC,1)],featMFCC);
figure; imagesc(featMFCC);

states = ones(1,size(featMFCC,2));
states(40:127) = 2;
states(128:166) = 3;
states(167:188) = 4;

[len,first,last] = SplitVec(states, [], 'length','first','last');
p = zeros(4,1);
for k = 1:4
    p(k) = 1-1/len(k);
end

figure; hold on;
col = 'kbrg';
for k = 1:4
    plot3(featMFCC(1,states==k),featMFCC(2,states==k),featMFCC(3,states==k),['x' col(k)])
end
xlabel('1');ylabel('2');zlabel('3')

mu = zeros(3,4);
K = zeros(3,3,4);
for k = 1:4
    mu(:,k) = mean(featMFCC(:,states==k),2); 
    K(:,:,k) = cov(featMFCC(:,states==k)');
end

%% model fitting
% initialize model
cohmm.pi = [1;0;0;0];
cohmm.A = [p(1) 1-p(1) 0 0; 0 p(2) 1-p(2) 0; 0 0 p(3) 1-p(3); 1-p(4) 0 0 p(4)];
funStr = '@(k,feat) ';
for k = 1:4
    funStr = [funStr sprintf('(k==%d)*mvnpdf(feat,',k)];
    funStr = [funStr sprintf('[%.4f;%.4f;%.4f],',mu(:,k))];
    funStr = [funStr sprintf('[%.4f %.4f %.4f;%.4f %.4f %.4f;%.4f %.4f %.4f])+',K(:,:,k))];
end
funStr(end) = ';';
eval(['cohmm.B = ' funStr]);

% optimize the model
newCohmm = cohmmBaumWelch(cohmm,featMFCC);
estStates = cohmmViterbi(newCohmm,featMFCC);
logProb = cohmmForwBack(newCohmm,featMFCC);

figure;
subplot(211); imagesc(tt,ff,S); axis xy
subplot(212); plot([1:size(estStates,2)]*frameSize/2/fs,estStates); axis tight
