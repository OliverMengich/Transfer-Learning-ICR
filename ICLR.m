
category ={'stem_rust','leaf_rust','healthy_wheat'};
imds = imageDatastore(fullfile(category),'IncludeSubfolders',true,'LabelSource','foldernames');

%% Now to extract the features with bag of features technique

%tic
%bag = bagOfFeatures(imds,'VOcabularySize',250,'PointSelection','Detector');
%toc
%%
%tic
%scenedata = double(encode(bag,imds));
%toc
%% COunt Each LABEL
tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});

%%
net = alexnet();
%analyzeNetwork(net)
%%
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imds.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
%%
pixelRange = [-30 30];
imageAugment = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augmentedimds = augmentedImageDatastore([227 227],imds,'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugment);

%%
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',20, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augmentedimds,layers,options);

%%
rootfolder = fullfile('D:\Matlab Files\ICLR');
 testimds = imageDatastore(fullfile(rootfolder,'test'),'IncludeSubfolders',true,'LabelSource','foldernames');
augsTest = augmentedImageDatastore([227 227],testimds,'ColorPreprocessing','gray2rgb');

%%
%predictedLabels = predict(netTransfer,augsTest);
[labels,err_test] = classify(netTransfer,augsTest);
testLabels = testimds.Labels;

%%
% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels,labels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

mean(diag(confMat))
%%
for i=1:4
    image = readimage(testimds,i);
    label = classify(netTransfer,imresize(image,[227 227]));
    figure;
    subplot(1,i,i); imshow(image); title(label);
end