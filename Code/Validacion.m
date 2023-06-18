clear memory; clear all; clc
load('CNN_resnet18.mat');

imds_val=imageDatastore('F:\Proyecto_grado\Imagenes\Segmentation_RPCA\Segmentado\Validation', 'IncludeSubfolders', true,'LabelSource','foldernames');
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
hold off
% obtener la matriz de confusión
confMat = confusionmat(labels_val, YPred);

% visualizar la matriz de confusión
confusionchart(confMat)
% Calcular el número de clases
numClasses = size(confMat, 1);

% Inicializar los vectores de precisión (precision) y sensibilidad (recall)
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);

% Calcular la precisión y sensibilidad para cada clase
for i = 1:numClasses
    % Calcular la precisión
    precision(i) = confMat(i,i) / sum(confMat(:,i));
    
    % Calcular la sensibilidad
    recall(i) = confMat(i,i) / sum(confMat(i,:));
end

% Calcular el F1-score para cada clase
f1Score = 2 * (precision .* recall) ./ (precision + recall);

% Calcular el promedio del F1-score
averageF1Score = mean(f1Score);

% Imprimir el promedio del F1-score en consola
disp(['F1-score: ', num2str(averageF1Score)]);