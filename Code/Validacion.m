clear memory; clear all; clc
load('Resnet18_AtlapetesBlancae.mat');

imds_val=imageDatastore('F:\Proyecto_grado\Imagenes\Segmentation_RPCA\Segmentado\Test', 'IncludeSubfolders', true,'LabelSource','foldernames');
labels_val = imds_val.Labels;
inputSize = [224 224 3];
imds_val = augmentedImageDatastore(inputSize, imds_val);
%% Validate Neural Network %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[YPred,scores] = classify(trainedNet,imds_val,'ExecutionEnvironment','cpu');
results=double(YPred);


[labels_val1 I]=sort(labels_val);
results1=results(I);

plot(labels_val1,'k','LineWidth',6) %plot labels
hold on
plot(results1,'.','MarkerSize',15) %plot centers
set(gca,'FontSize',16,'FontName','Arial');
set(gcf,'Color',[1 1 1])
xlabel('Call','FontSize',20,'FontName','Arial');
ylabel('Individual','FontSize',20,'FontName','Arial');
legend({'True Class','Predicted Class'},'Location','southeast','NumColumns',2)
hold on
% obtener la matriz de confusión
confMat = confusionmat(labels_val, YPred);

% visualizar la matriz de confusión
confusionchart(confMat)