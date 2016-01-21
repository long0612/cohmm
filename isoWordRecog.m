% Simple isolated word recognition
%
% Long Le <longle1@illinois.edu>
% University of Illinois
%

clear all; close all;

addpath(genpath('../voicebox/'))
addpath(genpath('../node-paper/'))


%% Read all data
files = dir('../network-paper/genCascade/data/GCW/');
nS = 5;

for k = 14:numel(files)
    figure('units','normalized','outerposition',[0 0 1 1]);
    disp(files(k).name)
    
    [y,fs] = audioread(['../network-paper/genCascade/data/GCW/' files(k).name]);
    %figure; plot([1:size(y,1)]/fs,abs(y));
    blockSize=256;
    [S,tt,ff] = mSpectrogram(y,fs,blockSize);
    subplot(311); imagesc(tt,ff,S); axis xy
    
    frameSize = 2^floor(log2(0.03*fs));
    featMFCC = melcepst(y,fs,'Mtaz',3,floor(3*log(fs)),frameSize)';
    %figure; imagesc([1:size(featMFCC,2)]*frameSize/2/fs,[1:size(featMFCC,1)],featMFCC);
    subplot(312); imagesc(featMFCC);
    
    % manual segmentation
    states = ones(1,size(featMFCC,2));
    vals = zeros(1,nS);
    firstIdx = 1;
    for l = 1:nS
        vals(l) = input(sprintf('Input state %d last value in seconds: ',l));
        lastIdx = round(vals(l)*fs/(frameSize/2));
        states(firstIdx:lastIdx) = l;
        
        firstIdx = lastIdx + 1;
    end
    
    subplot(313); hold on;
    col = 'kbrgm';
    for l = 1:nS
        plot3(featMFCC(1,states==l),featMFCC(2,states==l),featMFCC(3,states==l),['x' col(l)])
    end
    xlabel('1');ylabel('2');zlabel('3')
    
    [fpath,fname,fext] = fileparts(files(k).name);
    save(sprintf('localLogs/%s_seg.mat',fname),'vals','y','fs','frameSize','states','featMFCC');
    
    input('Press enter to continue');
    close;
end
%% inspect data for parameters
allLen = zeros(nS,numel(files)-2);
allFeatMFCC = cell(nS,numel(files)-2);
for k = 3:numel(files)
    [fpath,fname,fext] = fileparts(files(k).name);
    load(sprintf('localLogs/%s_seg.mat',fname),'states','featMFCC');
    [len,first,last] = SplitVec(states, [], 'length','first','last');
    allLen(:,k-2) = len(1:nS);
    for l = 1:nS
        allFeatMFCC{l,k-2} = featMFCC(:,states==l);
    end
end

p = 1-1./mean(allLen,2);

mu = zeros(3,nS);
K = zeros(3,3,nS);
for k = 1:nS
    mu(:,k) = mean(cell2mat(allFeatMFCC(k,:)),2); 
    K(:,:,k) = cov(cell2mat(allFeatMFCC(k,:))');
end

%% model fitting
% initialize model
cohmm.pi = [1;0;0;0;0];
cohmm.A = [p(1) 1-p(1) 0 0 0; 0 p(2) 1-p(2) 0 0; 0 0 p(3) 1-p(3) 0; 0 0 0 p(4) 1-p(4); 1-p(5) 0 0 0 p(5)];
funStr = '@(k,feat) ';
for k = 1:nS
    funStr = [funStr sprintf('(k==%d)*mvnpdf(feat,',k)];
    funStr = [funStr sprintf('[%.4f;%.4f;%.4f],',mu(:,k))];
    funStr = [funStr sprintf('[%.4f %.4f %.4f;%.4f %.4f %.4f;%.4f %.4f %.4f])+',K(:,:,k))];
end
funStr(end) = ';';
eval(['cohmm.B = ' funStr]);

% optimize the model
mulFeatMFCC = cell(1,numel(files)-2);
for k = 3:numel(files)
    [fpath,fname,fext] = fileparts(files(k).name);
    load(sprintf('localLogs/%s_seg.mat',fname),'featMFCC');
    mulFeatMFCC{k-2} = featMFCC;
end
newCohmm = cohmmBaumWelch(cohmm,mulFeatMFCC);

for k = 3:numel(files)
    [fpath,fname,fext] = fileparts(files(k).name);
    load(sprintf('localLogs/%s_seg.mat',fname),'y','fs','featMFCC');
    
    blockSize=256;
    [S,tt,ff] = mSpectrogram(y,fs,blockSize);
    estStates = cohmmViterbi(newCohmm,featMFCC);
    logProb = cohmmForwBack(newCohmm,featMFCC);
    
    figure;
    subplot(211); imagesc(tt,ff,S); axis xy
    subplot(212); plot([1:size(estStates,2)]*frameSize/2/fs,estStates); axis tight
    suptitle(sprintf('file %s, logProb is %.4f',fname,logProb))
end

% try a random input
[y,fs] = audioread('audio.wav');
frameSize = 2^floor(log2(0.03*fs));
featMFCC = melcepst(y,fs,'Mtaz',3,floor(3*log(fs)),frameSize)';

blockSize=256;
[S,tt,ff] = mSpectrogram(y,fs,blockSize);
estStates = cohmmViterbi(newCohmm,featMFCC);
logProb = cohmmForwBack(newCohmm,featMFCC);

figure;
subplot(211); imagesc(tt,ff,S); axis xy
subplot(212); plot([1:size(estStates,2)]*frameSize/2/fs,estStates); axis tight
suptitle(sprintf('logProb is %.4f',logProb))