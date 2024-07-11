clear all;close all;clc;
srcFiles = dir('Y:\OTHER WORKS\2023-24\BUSINESS PROJECTS\JANUARY - 2024\TK137337  -  Leaf Diseases Prediction Pest Detection and Pesticides Recommendation using Deep Learning Techniques\BASE\CODE\DATASET\Spodoptera exigua\*.JPG');  % the folder in which ur images exists
for i = 1 : length(srcFiles)
filename = strcat('Y:\OTHER WORKS\2023-24\BUSINESS PROJECTS\JANUARY - 2024\TK137337  -  Leaf Diseases Prediction Pest Detection and Pesticides Recommendation using Deep Learning Techniques\BASE\CODE\DATASET\Spodoptera exigua\',srcFiles(i).name);
img = imread(filename);
resize_img = imresize(img, [224, 224]);

I = rgb2gray(resize_img);

thresh = multithresh(I,2);
labels = imquantize(I,thresh);

labelsRGB = label2rgb(labels);
figure,
imshow(labelsRGB)
title("Segmented Image")

newfilename=fullfile('Y:\OTHER WORKS\2023-24\BUSINESS PROJECTS\JANUARY - 2024\TK137337  -  Leaf Diseases Prediction Pest Detection and Pesticides Recommendation using Deep Learning Techniques\BASE\CODE\Resized_Images\Spodoptera exigua\',srcFiles(i).name);
imwrite(labelsRGB,newfilename,'png');
end

