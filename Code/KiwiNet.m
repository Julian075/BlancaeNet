
function  KiwiNet(AudioFolder,KiwiAtlapetesNet,krgb)
KiwiAtlapetesNet=KiwiAtlapetesNet.('KiwiAtlapetesNet');
krgb=krgb.('krgb');
rutout_results=[pwd '/Kiwinet'];
fol_audios=AudioFolder; %folder with audios
Name_excel='Results_Audio';
nfft=1024; %number of sampling points to calculate the discrete Fourier transform.
window=1024; %spectrogram's window length
overlap=768; %spectrogram's window overlap
ArchSize=224; %input size of the image for the CNN
 %colormap black - red -green -blue
cmap=krgb;

%%%% folders where images are going to be classification
images_classification   = [pwd '/Kiwinet/Espectogramas'];
%Erase old spectrograms
    if(exist(images_classification)==7)%7 porque en la doc de la fun , dice que return 7 cuando es un folder
        delete([images_classification '/*'])
    end

%%  ***                                                             ***  %%
%% %%%%%%%%%%%%%%%%%%% Algorithm %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  ***                                                             ***  %%

%%  ***************************************************************
rnam=dir(fol_audios); %% recording names
if  filesep=='/' rnam=rnam(4:length(rnam)); else rnam=rnam(3:length(rnam)); end %% this line removes the dots (...) from the file names
for jj=1:length(rnam) 
    jj
    %% additional folder information ******************************************
        
        %% Recording **************************************************************
        [s fs]=audioread([fol_audios '/' rnam(jj,1).name]); %% read the recording
        %% pre-processing *********************************************************
        [data1] = preprocessing(s,fs,window,overlap,nfft);
        data2=imresize(data1, [ArchSize, ArchSize], 'method','bilinear');
        %% Features for Deep learning *********************************************
        fh = figure('Menu','none','ToolBar','none','visible','off');
        ah = axes('Units','Normalize','Position',[0 0 1 1]);
        imagesc(data2)
        set(gca,'xTick',[])
        set(gca,'yTick',[])
        colormap(cmap)
        saveas(gcf,[[images_classification filesep] rnam(jj,1).name(1:end-4) '.png'])
        close all
        
end

try
clear memory

spectrograms=dir(images_classification);
Noise=zeros(length(spectrograms)-2,1);
Atlapetes_Blancae=zeros(length(spectrograms)-2,1);
catch ME
                disp('Error capturado:');
                disp(ME.identifier);
                errordlg('Error KiwiNet Pre')
                errordlg(ME.identifier)
end

%%% load spectrograms as images %%%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k=3:length(spectrograms)
    k
    try
    img=imread([[images_classification '/'] spectrograms(k,1).name]);
    rgb_calls=imresize(img,[ArchSize,ArchSize]); %Match image size to CNN input
    catch ME
                disp('Error capturado:');
                disp(ME.identifier);
                errordlg('Error KiwiNet imread')
                errordlg(ME.identifier)
   end
    try
    y= predict(KiwiAtlapetesNet,rgb_calls);
   %[YPred,scores] = classify(KiwiAtlapetesNet,rgb_calls,'ExecutionEnvironment','cpu');
    catch ME
                disp('Error capturado:');
                disp(ME.identifier);
                errordlg('Error KiwiNet predict')
                errordlg(ME.identifier)
                break
   end
    Noise(k-2)=y(1);
    Atlapetes_Blancae(k-2)=y(2);
    
end 

Names=struct2table(spectrograms);
Names=Names(:,1);
Names=Names(3:end,:);
Names=table2array(Names);
Noise=round(Noise);
Atlapetes_Blancae=round(Atlapetes_Blancae);
Y=table(Names,Noise,Atlapetes_Blancae);

if(exist([rutout_results '/' Name_excel '.csv'])==2)%2 for file
        delete([rutout_results '/' Name_excel '.csv'])
        end
writetable(Y,[rutout_results '/' Name_excel '.csv'])


% info = audioinfo('E:\Kiwi_Net_Atlapetes\Augmented_training\Inputs\Noise\Site1\32.wav')
% info2= audioinfo('E:\Kiwi_Net_Atlapetes\Augmented_training\Inputs\Individuals\Ausencias\20211113_160307.wav')
% 
% [y,Fs] = audioread('E:\Kiwi_Net_Atlapetes\Augmented_training\Inputs\Noise\Site1\32.wav');
% a=resample(y,48000,44100);
% audiowrite('E:\New.wav',a,48000);
% audioinfo('E:\New.wav')


%% %%%%%%%%% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [data1] = preprocessing(s,fs,window,overlap,nbins)
%% Median Equalizer *******************************************************

%Se agrego el if para verificar convertir el audio a mono en caso de ser
%estereo
if size(s, 2)>1
    s=s(:,1);
end

spec=spectrogram(s,hann(window),overlap,nbins,fs);
spec1=abs(spec.^2); %power spectrogram
spec4=(spec1'./median(spec1'))'; %noise filter
spec4 = medfilt2(spec4,[5 1]);   %median filter
[a,b]=max(spec4(:)); % next 3 lines restore the power values
[c,d]=min(spec4(:));
data1=(spec1(b)-spec1(d)).*(spec4./(max(spec4(:))))+spec1(d);
data1=10*log10(abs(data1)); %power spectrogram in dB
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
end