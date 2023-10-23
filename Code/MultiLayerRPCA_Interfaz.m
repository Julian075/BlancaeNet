
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This code summarizes the experiments of the paper "Camera-Trap Images
%Segmentation using Multi-Layer Robust Principal Component Analysis". The
%Visual Computer, 2017.
%
%Authors: Jhony Heriberto Giraldo Zuluaga, Augusto Salazar, Alexander
%Gomez, Angélica Diaz Pulido.
%Universidad de Antioquia, Medellín, Colombia.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function MultiLayerRPCA_Interfaz(VideoFolder)

VideosDir=[VideoFolder '/'];
FramesDir=[pwd '/MultiLayerRPCA_CNN/Frames/'];
subfolders = dir(FramesDir);
subfolders = subfolders([subfolders.isdir]); % selecciona solo los subdirectorios
for i = 3:length(subfolders)
        rmdir(fullfile(FramesDir, subfolders(i).name),'s');
end
BmpDir=[pwd '/MultiLayerRPCA_CNN/algoritmo'];
subfolders = dir(BmpDir);
subfolders = subfolders([subfolders.isdir]); % selecciona solo los subdirectorios
for i = 3:length(subfolders)
        rmdir(fullfile(BmpDir, subfolders(i).name),'s');
end
listVideos = dir(VideosDir);


data=[];
for(j=1:size(listVideos,1)-2)
    VideoPath = [VideosDir listVideos(j+2).name];
    FramePath = [FramesDir listVideos(j+2).name(1:end-4) '/'];
    shuttleVideo = VideoReader(VideoPath);
    %guradar frame rate
    frameRate = shuttleVideo.FrameRate;
    name=shuttleVideo.Name;
    data = [data; {name(1:end-4), frameRate}];
    %%
    mkdir(FramePath)
    
    ii = 1;
    
    while hasFrame(shuttleVideo)
       img = readFrame(shuttleVideo);
       filename = [sprintf('%03d',ii) '.jpg'];
       fullname = fullfile(FramePath,filename);
       imwrite(img,fullname,'jpg')    % Write out to a JPEG file (img1.jpg, img2.jpg, etc.)
       ii = ii+1;
    end
end
% guardar cvs de frame rates
csvFileName = 'frame_rates.csv';
if exist(csvFileName, 'file') == 2
    delete(csvFileName);
end
data=cell2table(data);
data.Properties.VariableNames = {'VideoName', 'FrameRate'};
writetable(data, csvFileName);


beta = [0.6:0.05:0.6];
%groundPath = '/pathTo/GroundTruthForShare/Ground/'; %Path to Ground path
groundPath =FramesDir; %Path to Ground path
listGround = dir(groundPath);
%methods = {'EALM','IALM','APG_PARTIAL','APG',...
   % 'LSADM','NSA1','NSA2'};                         %PCP algorithms
   methods = {'NSA1'}; 
for(h=1:size(beta,2))
    for(i=1:size(methods,2))
        methodPath = [BmpDir '/'];
        mkdir(methodPath);
        for(j=1:size(listGround,1)-2)
            colorPath = [groundPath listGround(j+2).name '/'];
            listColor = dir(colorPath);
            automaticColorPath = [methodPath listGround(j+2).name '/'];
            mkdir(automaticColorPath);
            M = [];
            imageName = [];
            OriginalImage = {};
            for(k=1:size(listColor,1)-2)
                imageName = listColor(k+2).name;
                imageName = imageName(1:end-4);
                
                imageNames{k} = imageName;
                imagePath = [colorPath listColor(k+2).name];
                image = imread(imagePath);
                %image = image(1:2348,:,:);
                image = imresize(image,0.1);
                if(size(image,3) == 3)
                    image = rgb2gray(image);
                end
                OriginalImage{k,1} = image;
                colorOrInfrared = listGround(j+2).name;
                colorOrInfrared = colorOrInfrared(1:3);
                if(strcmp(colorOrInfrared,'Inf'))
                    image = imgaussfilt(image); %Gaussian filter
                end
                nFiltSize = 8;
                nFiltRadius = 1;
                filtR = generateRadialFilterLBP(nFiltSize, nFiltRadius);
                effLBP = efficientLBP(image);   %LBP descriptor
                if(strcmp(colorOrInfrared,'Col'))
                    image = histeq(image);  %Histogram equalization
                end
                resultImg = beta(h)*effLBP + (1-beta(h))*image; %Linear combination
                [vidHeight vidWidth z] = size(image);
                M(:,k) = reshape(resultImg,[],1);
            end
            M = im2double(M);
            try
            out = run_algorithm_2('RPCA', methods{1,i}, M, []);   %Run PCP algorithm
            catch ME
                disp('Error capturado:');
                disp(ME.identifier);
                errordlg(ME.identifier)
                break
            end
            segmentedImages = out.O;
            for(l=1:size(segmentedImages,2))
                segmentedImage = reshape(segmentedImages(:,l),vidHeight,vidWidth);
                groundImg = OriginalImage{l,1};
                %%
                %Postprocessing
                imgMedian = medfilt2(segmentedImage, [3 3]);    %Median filter
                se = strel('disk',3);   %Morphological structuring element
                afterOpening = imopen(imgMedian,se);    %Morphological oppening
                %bw = activecontour(groundImg,afterOpening,200,'edge','contractionBias',-0.3);   %Active contours
                bw = imclose(afterOpening,se);    %Morphological clossing
                bw = imopen(bw,se);     %Morphological openning
                l;
                bw = imbinarize(bw);
                imwrite(bw,[automaticColorPath imageNames{l} '.bmp'],'bmp');    %Saving the segmented image
            end
        end
    end
end
end