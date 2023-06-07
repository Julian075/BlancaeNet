%clear all;clc;
function  Recortar_imagenes_boundingbox()
dir_ima=[pwd '/MultiLayerRPCA_CNN/Frames/'];
dir_algor=[pwd '/MultiLayerRPCA_CNN/algoritmo'];
dir_out=[pwd '/MultiLayerRPCA_CNN/Recortes/'];
subfolders = dir(dir_out);
subfolders = subfolders([subfolders.isdir]); % selecciona solo los subdirectorios
for i = 3:length(subfolders)
        rmdir(fullfile(dir_out, subfolders(i).name),'s');
end
images=dir(dir_ima);


cont=0;
for i=3:length(images)
    frames=dir([dir_ima images(i).name]);
    mkdir([dir_out '/' images(i).name]);
    cont=cont+1;
    cont2=0;
    for j=3:length(frames)
        ima=imread([dir_ima images(i).name '/' frames(j).name ]);
        
        ima=ima(1:height(ima)-30,:,:);
        bmp= imread([dir_algor '/' images(i).name  '/' frames(j).name(1:end-4) '.bmp' ]);
        bmp=bmp(1:height(bmp)-3,:,:);
 
        CC = bwconncomp(bmp).NumObjects; %Detecto cuantas Regiones de interes hay
        Box=regionprops(bmp,'BoundingBox');
        if CC>1 %Si hay ROI 
              cont2=cont2+1;
              %Box=regionprops(bmp,'BoundingBox'); %Calcula los bounding box
              interception=0;
              cont3=0;
              for reg=1:CC
                  box_eval=Box(reg);
                  if ~(reg==CC)
                      
                      for reg2=reg+1:CC
                          if (isa(box_eval,'struct')==1)
                              a=struct2array(box_eval);
                          else
                              a=box_eval;
                          end
                          b=struct2array(Box(reg2));
                          if(bboxOverlapRatio(a,b)==1)
                            if reg2==CC
                                interception=1;
                            end
                            xmin=min(a(1),b(1));
                            ymin=min(a(2),b(2));
                            xmax=max(a(1)+a(3),b(1)+b(3));
                            ymax=max(a(2)+a(4),b(2)+b(4));
                            box_eval=[xmin  ymin (-xmin+(xmax)) (-ymin+(ymax))];
                          end

                      end
                      if (isa(box_eval,'struct')==1)
                              box_eval=struct2array(box_eval);
                      end
                      cont3=cont3+1;
                      Icropped = imcrop(ima,box_eval*10) ;
                      %imshow(Icropped);
                      imwrite(Icropped,[dir_out '/' images(i).name '/' 'Frame_' sprintf('%03d', cont2) '_region_' sprintf('%03d', cont3) '.jpg'])
                  end
                  if interception==0 && reg==CC
                      if (isa(box_eval,'struct')==1)
                              box_eval=struct2array(box_eval);
                      end
                      cont3=cont3+1;
                      Icropped = imcrop(ima,box_eval*10) ;
                      %imshow(Icropped);
                      if ~(isempty(Icropped))
                      imwrite(Icropped,[dir_out '/' images(i).name '/' 'Frame_' sprintf('%03d', cont2) '_region_' sprintf('%03d', cont3) '.jpg'])
                      end
                  end
              end
  

        elseif(CC==1)
            
            cont2=cont2+1;
            box_eval=Box;
            box_eval=struct2array(box_eval);
            Icropped = imcrop(ima,box_eval*10) ;
            if ~(isempty(Icropped))
            imwrite(Icropped,[dir_out '/' images(i).name '/' 'Frame_' sprintf('%03d', cont2) '_region_001' '.jpg'])
            end
        end 
    end

end
    
end