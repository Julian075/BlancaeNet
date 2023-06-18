
% Cargar los datos de entrenamiento y prueba
imdsTrain = imageDatastore('F:\Proyecto_grado\Imagenes\Segmentation_RPCA\Segmentado\Train', 'IncludeSubfolders', true,'LabelSource','foldernames');
imds_val = imageDatastore('F:\Proyecto_grado\Imagenes\Segmentation_RPCA\Segmentado\Test', 'IncludeSubfolders', true,'LabelSource','foldernames');
inputSize = [224 224 3];
imdsTrain = augmentedImageDatastore(inputSize, imdsTrain);
imds_val = augmentedImageDatastore(inputSize, imds_val);

%% CNN ********************************************************************
% Cargar una red ResNet-18 pre-entrenada con ImageNet
net = resnet18;
analyzeNetwork (net)
% Reemplazar la última capa completamente conectada (fc1000) con una nueva capa completamente conectada con el número de clases
numClasses = 4
lgraph = layerGraph(net);
newLayers = fullyConnectedLayer(numClasses, 'Name','fc', 'WeightLearnRateFactor',20, 'BiasLearnRateFactor',20);
lgraph = replaceLayer(lgraph,'fc1000',newLayers);
newLayers = softmaxLayer('Name','softmax');
%lgraph = replaceLayer(lgraph,'fc1000_softmax',newLayers);
lgraph = replaceLayer(lgraph,'prob',newLayers);
newLayers = classificationLayer('Name','classoutput');
%lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newLayers);
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newLayers);


%% CCN parameters *********************************************************
options = trainingOptions('sgdm',... %stochastic gradient descent with momentum
    'MiniBatchSize',16,...
    'MaxEpochs',5,...
    'InitialLearnRate',1e-4,...
    'ValidationData',imds_val,...
    'ValidationFrequency',500,...
    'ValidationPatience',Inf,...
    'Verbose',1,...
    'ExecutionEnvironment','gpu',...
    'Plots','training-progress');
    %'L2Regularization',0.001,...
    %'Shuffle','every-epoch',...
gpuDevice(1); %select the GPU to be used

%% train neural network %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
analyzeNetwork (lgraph)
trainedNet = trainNetwork(imdsTrain,lgraph,options);
save('CNN_resnet18','trainedNet')