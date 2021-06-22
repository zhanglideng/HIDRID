%% Environment requires (1)Windows system (2)matlab software

%%
clc,clear;
%mex TSSP.c
addpath Gradient
%addpath img
gamma = 1;
radius = 3;
numsuper = 400;

filename='00129.png';
img = imread(filename);    
img_pad = padarray(img,[radius,radius],'symmetric');
img_lab = double(rgb2lab(img_pad));
img_gray = double(rgb2gray(img_pad));
[magnitude,direction]=HybridGradient(img,radius);
[labels, numlabels] = TSSP(numsuper,img,img_lab,img_gray,magnitude,direction,radius,gamma);
new_DisplaySuperpixel(labels,img,filename); 