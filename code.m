clc;
clear all;
close all;
close all hidden;
warning off
%%
% Load the input image
[file, path] = uigetfile('*.*', 'Select an image');
img = imread([path, file]);
figure,
imshow(img);
title('Input Image');
%% Image Resize

resize_img = imresize(img, [224, 224]);
figure,
imshow(resize_img);
title('Resize Image');

%% Gray Conversion

I = rgb2gray(resize_img);
figure,
imshow(I)
title("Grayscale Image");
%% Segmentation

thresh = multithresh(I,2);
labels = imquantize(I,thresh);

labelsRGB = label2rgb(labels);
figure,
imshow(labelsRGB)
title("Segmented Image");
%% Resnet50 Training and Testing

matlabroot = cd;    % Dataset path
datasetpath = fullfile(matlabroot,'Resized_Images');   %Build full file name from parts
imds = imageDatastore(datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');    %Datastore for image data

[imdsTrain, imdsValidation] = splitEachLabel(imds,0.8,'randomized');     %Split ImageDatastore labels by proportions
augimdsTrain = augmentedImageDatastore([224 224 3],imdsTrain);  %Generate batches of augmented image data
augimdsValidation = augmentedImageDatastore([224 224 3],imdsValidation);


layers = [
    imageInputLayer([224 224 3])
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];  

options = trainingOptions("sgdm",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.001,...
    "MaxEpochs",100,...
    "MiniBatchSize",25,...
    "Shuffle","every-epoch",...
    "ValidationData",augimdsValidation, ...
    "ValidationFrequency",50,...
    "Verbose",true, ... 
    "Plots","training-progress");

% [net, Traininfo] = trainNetwork(augimdsTrain, layers, options);  

load net
load Traininfo
%% Classification

[YPred,score] = classify(net,labelsRGB);
msgbox(char(YPred))
%%
output = char(YPred);
if strcmp( output , 'Locusta migratoria');
    disp('PESTISIDES: Apply Pyrethrin-based pesticide directly on Locusta migratoria infestations for control');
    
     elseif strcmp( output , 'Atractomorpha sinensis');
    disp('PESTISIDES: Apply Neem oil pesticide on Atractomorpha sinensis for effective pest management.');
    
    elseif strcmp( output , 'Chrysochus chinensis');
    disp('PESTISIDES: Spray Bacillus thuringiensis on Chrysochus chinensis for targeted pest elimination.');
    
    elseif strcmp( output , 'Empoasca flavescens');
    disp('PESTISIDES: Use Spinosad-based pesticide to treat Empoasca flavescens infestations effectively.');
    
    elseif strcmp( output , 'Gypsy moth larva');
    disp('PESTISIDES: Apply Bacillus thuringiensis directly on Gypsy moth larva for control');
    
    elseif strcmp( output , 'Laspeyresia pomonella');
    disp('PESTISIDES: Spray Spinosad pesticide on Laspeyresia pomonella for effective pest control.');
    
    elseif strcmp( output , 'Laspeyresia pomonella larva');
    disp('PESTISIDES: Apply Bacillus thuringiensis directly on Laspeyresia pomonella larva for control.');
    
    elseif strcmp( output , 'Parasa lepida');
    disp('PESTISIDES: Use Neem oil pesticide to control Parasa lepida infestations effectively.');
    
    elseif strcmp( output , 'Spodoptera exigua');
    disp('PESTISIDES: Apply Pyrethrin-based pesticide directly on Spodoptera exigua infestations for control.');
    
else 
    disp('PESTISIDES: Spray Bacillus thuringiensis on Spodoptera exigua larva for effective control.');
    
 
end

%% Accuracy

accuracy = mean(Traininfo.TrainingAccuracy);
fprintf('Accuracy of classified Model is: %0.4f\n',accuracy);

predictedLabels = classify(net, augimdsTrain);
testLabels = imdsTrain.Labels;

% Create the confusion matrix
C = confusionmat(predictedLabels,testLabels);
figure
confu = confusionchart(C);

% Calculate Matrcies 
accuracy = sum(diag(C)) / sum(C(:));
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

precision = diag(C) ./ sum(C, 1)';
fprintf('Precision: %.2f%%\n', mean(precision) * 100);

recall = diag(C) ./ sum(C, 2);
fprintf('Recall: %.2f%%\n', mean(recall) * 100);

f1score = 2 * (precision .* recall) ./ (precision + recall);
fprintf('f1score: %.2f%%\n', mean(f1score) * 100);


%%
