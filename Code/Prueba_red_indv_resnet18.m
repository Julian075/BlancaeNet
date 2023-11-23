function  Prueba_red_indv_resnet18(trainedNet)
%clear memory; clear all; clc
%load('Resnet18_AtlapetesBlancae.mat');
try
trainedNet=trainedNet.('trainedNet');
Name_excel='Video_Result';
Ruta_carpetas=[pwd '/MultiLayerRPCA_CNN/Recortes/'];
videos=dir(Ruta_carpetas);
Tabla_Total=table();
catch ME
    errordlg('Error CNN Part 1');
    errordlg(ME.identifier);
end
for k=3:length(videos)
    %try
        ruta=[Ruta_carpetas videos(k).name] ;
        val_data_not_empty=dir(ruta);
        if length(val_data_not_empty)>2
        imds_val=imageDatastore(ruta);
        
        numImages = length(dir(ruta))-2;
        inputSize = [224 224 3];
        imds_val = augmentedImageDatastore(inputSize, imds_val);
        
        table_frame_rate = readtable('frame_rates.csv','Format', '%s%s');
        % Iterar sobre el conjunto de datos de validación
        %% Validate Neural Network %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [YPred,scores] = classify(trainedNet,imds_val,'ExecutionEnvironment','cpu');
        YPred(YPred ~= 'Atlapetes_Blancae' ) = "0";
        YPred(YPred == 'Atlapetes_Blancae') = "1";
        Names=struct2table(dir(ruta));
        Names=Names(:,1);
        Names=Names(3:end,:);
        Names=table2array(Names);
        
        Time_s=zeros(length(Names),1);
        Name_Video=string(zeros(length(Names),1));
        
        %frame rate para saber seg
    %catch ME
    %    errordlg('Error load CNN');
    %    errordlg(ME.identifier);
    %    
    %end

        for i=1:length(Names)
            frame=split(Names(i),'_');
            frame=frame(2);
            frame=str2double(frame{1});
            vid_name=videos(k).name;
            for j=1:length(table_frame_rate.VideoName)
                if strcmp(cell2mat(table_frame_rate.VideoName(j)),vid_name)==1
                    indice=j;
                    break
                end
            end
            frame_rate=30;%table_frame_rate.FrameRate (indice);
            segundo = (frame - 1) / frame_rate;%str2double(frame_rate);
            segundo=round(segundo,1);
            Time_s(i)=segundo;
            Name_Video(i)=vid_name;
        end
        Y=table(Name_Video,Time_s,YPred);
        Tabla_Total=[Tabla_Total;Y];
        end
        
end
try
rutaout=[pwd '/MultiLayerRPCA_CNN'];
if(exist([rutaout '/' Name_excel '.csv'])==2)%2 for file
        delete([rutaout '/' Name_excel '.csv'])
        end
writetable(Tabla_Total,[rutaout '/' Name_excel '.csv']);
catch ME
    errordlg('Error write cvs');
    errordlg(ME.identifier);
end

