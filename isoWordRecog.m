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

for k = 13:numel(files)
    figure(1);
    disp(files(k).name)
    
    [y,fs] = audioread(['../network-paper/genCascade/data/GCW/' files(k).name]);
    %figure; plot([1:size(y,1)]/fs,abs(y));
    blockSize=256;
    [S,tt,ff] = mSpectrogram(y,fs,blockSize);
    subplot(211); imagesc(tt,ff,S); axis xy
    
    frameSize = 2^floor(log2(0.03*fs));
    featMFCC = melcepst(y,fs,'Mtaz',3,floor(3*log(fs)),frameSize)';
    %figure; imagesc([1:size(featMFCC,2)]*frameSize/2/fs,[1:size(featMFCC,1)],featMFCC);
    subplot(212); imagesc(featMFCC);
    
    % manual segmentation
    states = ones(1,size(featMFCC,2));
    vals = zeros(1,4);
    firstIdx = 1;
    for l = 1:4
        vals(l) = input(sprintf('Input state %d last value in seconds: ',l));
        lastIdx = round(vals(l)*fs/(frameSize/2));
        states(firstIdx:lastIdx) = l;
        
        firstIdx = lastIdx + 1;
    end
    
    figure; hold on;
    col = 'kbrg';
    for l = 1:4
        plot3(featMFCC(1,states==l),featMFCC(2,states==l),featMFCC(3,states==l),['x' col(l)])
    end
    xlabel('1');ylabel('2');zlabel('3')
    
    [fpath,fname,fext] = fileparts(files(k).name);
    save(sprintf('localLogs/%s_seg.mat',fname),'vals','y','fs','frameSize','states','featMFCC');
    
    close(1);
end
%% inspect data for parameters
allFeatMFCC = cell(4,numel(files)-2);
allLen = zeros(4,numel(files)-2);
for k = 3:numel(files)
    [fpath,fname,fext] = fileparts(files(k).name);
    load(sprintf('localLogs/%s_seg.mat',fname),'states','featMFCC');
    [len,first,last] = SplitVec(states, [], 'length','first','last');
    allLen(:,k-2) = len(1:4);
    for l = 1:4
        allFeatMFCC{l,k-2} = featMFCC(:,states==l);
    end
end

p = 1-1./mean(allLen,2);

mu = zeros(3,4);
K = zeros(3,3,4);
for k = 1:4
    mu(:,k) = mean(cell2mat(allFeatMFCC(k,:)),2); 
    K(:,:,k) = cov(cell2mat(allFeatMFCC(k,:))');
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
newCohmm = cohmm;
for k = 3:numel(files)
    [fpath,fname,fext] = fileparts(files(k).name);
    load(sprintf('localLogs/%s_seg.mat',fname),'featMFCC');
    newCohmm = cohmmBaumWelch(newCohmm,featMFCC);
end

for k = 3:numel(files)
    [fpath,fname,fext] = fileparts(files(k).name);
    load(sprintf('localLogs/%s_seg.mat',fname),'featMFCC');
    estStates = cohmmViterbi(newCohmm,featMFCC);
    logProb = cohmmForwBack(newCohmm,featMFCC);
    
    figure;
    subplot(211); imagesc(tt,ff,S); axis xy
    subplot(212); plot([1:size(estStates,2)]*frameSize/2/fs,estStates); axis tight
    title(sprintf('logProb is %.4f',logProb))
end