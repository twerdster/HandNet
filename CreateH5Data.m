

% This script creates hdf5 files for use in programs such as caffe. It
% splits the data into useful categories and for each splits up the data into files with
% no more than maxExamples examples in each h5 file.
%
% This script is split into 3 main sections
% 1) Setup
% 2) Loop over input files
% 3) File list generation

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 1: Setup
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all

expName = 'TrainData_DerotGT/'; 

dataDir = ['/home/gipuser/Documents/BMVC/OriginalData/' expName];
outDir = ['/home/gipuser/Documents/BMVC/' expName];
mkdir(outDir);

d=dir([dataDir 'Data*.mat']);

len = length(d);
tform = maketform('affine',eye(3));

maxExamples = 20000; % Allow only 20K examples per HDF5 file
alpha = 1.0; % Bounding box scale 

imCnt = 0;
oldId = -1;
newId = -1;

centerInd = 6; % The palm center is id 6. The thumb to pinky are 1 - 5
maxDepth = 250/2; % Half the size of an enclosing volume box around hand
hsz = 18; % Heatmap is 18x18x6
imsz = 96; % Image size is 96x96
hmpInds = [1 2 3 4 5 6]; % We use all the data here
lenh = length(hmpInds);

h = fspecial('gaussian',9,5); %Multiscale filter

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 2: Loop over input to create output data
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for cnt=1:len
    %% Load data and format it
    fprintf('%i/%i\n',imCnt,len);
    load([dataDir d(cnt).name]);
    
    % If you wish to create your own hand detection and segmentation routine then
    % this is where you would apply it.
    % i.e. [handPos, segmap] = detectAndFindHandCenter(depth);
    handPos = handPos;
    segmap = segmap;
    
    % Here we slightly update the data
    hmap = hmap(:,:,hmpInds);
    
    % Small annoyance: heatmap data wasnt normalized between 0 and 1 so we
    % fix that here
    minVal = min(hmap(:)); % should be exp(-5)
    maxVal = 1;
    hmap = (hmap-minVal)/(maxVal-minVal);
    
    % Heatmaps with the sensor out of the frame include nans
    % which we need to remove.
    hmap(isnan(hmap))=0;
    
    % Finalize data mask and data capping
    msk = depth==0 | ~segmap; % activated wherever there is no data or no hand segments
    depth = single(depth);
    depth = (depth - handPos(3)) / maxDepth; % Remove the depth of the detected hand
    depth(msk)=1; %Background pixels are set to 1
    depth(depth>1)=1; %and pixels further than maxdepth are capped at 1
    depth(depth<-1)=-1; % and pixels closer than -maxdepth are capped to -1
    
    %     % we zero out any data beyond the segmentation mask
    %     % This forces the areas where there is not data to not predict
    %     hmap(repmat(msk,[1 1 lenh]))=0;
    
    % Set bounding box
    bbox = double(bbox);    
    bbox = [bbox(1:2)+0.5*bbox(3:4)-bbox(3:4)*0.5*alpha bbox(3:4)*alpha];
    
    % Filling in missing data when cropping with out of picture: I will
    % guess in these areas a non-depth i.e. background = 1. The problem
    % with this is likely to be that we dont learn how to explicitly deal
    % with image edges and instead create false non-data where we could
    % somehow encode the edges of the image. Will leave this for future
    % training.
    D = imtransform(depth,tform,'nearest','FillValues',1,'UData',[1 size(ir,2)], 'VData',[1 size(ir,1)], ...
        'XData',[bbox(1) bbox(1)+bbox(3)],'YData',[bbox(2) bbox(2)+bbox(4)], ...
        'Size', [imsz imsz]);
    
    D_{1} = D-imfilter(D,h,'replicate');
    D_{2} = imresize(D,[imsz/2 imsz/2]) - imfilter(imresize(D,[imsz/2 imsz/2]),h,'replicate');
    D_{3} = imresize(D,[imsz/4 imsz/4]) - imfilter(imresize(D,[imsz/4 imsz/4]),h,'replicate');
    
    % Filling in missing data for fingertips. I would prefer to guess that
    % there is background there. This will be useful for the tree learner.
    % I.e. bg = 0
    LBL = imtransform(lbl,tform,'nearest','FillValues',0,'UData',[1 size(ir,2)], 'VData',[1 size(ir,1)], ...
        'XData',[bbox(1) bbox(1)+bbox(3)],'YData',[bbox(2) bbox(2)+bbox(4)], ...
        'Size', [imsz imsz]);
    
    % Filling in missing data for heatmap: I dont want to kill the data
    % there. I want to say there is a small chance of there being data.
    % I.e. 0.1. But currently it is set to 0 so we do kill data if it is
    % out of bounds.
    HMP = imtransform(hmap,tform,'linear','FillValues',0,'UData',[1 size(ir,2)], 'VData',[1 size(ir,1)], ...
        'XData',[bbox(1) bbox(1)+bbox(3)],'YData',[bbox(2) bbox(2)+bbox(4)], ...
        'Size', [hsz hsz lenh]);
    
%     HMP_big = imtransform(hmap,tform,'linear','FillValues',0,'UData',[1 size(ir,2)], 'VData',[1 size(ir,1)], ...
%         'XData',[bbox(1) bbox(1)+bbox(3)],'YData',[bbox(2) bbox(2)+bbox(4)], ...
%         'Size', [imsz imsz lenh]);
    
    
    %% Data processing for being appended to h5 files 
    
    D = reshape(single(D'),[imsz imsz 1 1]); % Full depth crop
    D_{1} = reshape(single(D_{1}'),[imsz imsz 1 1]); % First level multiscale data
    D_{2} = reshape(single(D_{2}'),[imsz/2 imsz/2 1 1]);% Second level multiscale data
    D_{3} = reshape(single(D_{3}'),[imsz/4 imsz/4 1 1]);% Third level multiscale data
    
    LBL =  reshape(single(LBL'),[imsz imsz 1 1]); %Label
    HMP = reshape(single(permute(HMP,[2 1 3])),[hsz hsz lenh 1]); % Heatmap
    %HMP_big = reshape(single(permute(HMP_big,[2 1 3])),[imsz imsz lenh 1]);
    
    Rot = rot(:,:,centerInd); % Rotation
    Angle=derotate(Rot); % Direct derotation angle
    [Eul(1), Eul(2), Eul(3)] = dcm2angle(Rot); %Euler angles
    Quat = dcm2quat(Rot); % Quaternion
    Thumb = (pos(1:3,1)-handPos)/maxDepth; % Thumb offset from center
    PalmCenter =  (pos(1:3,centerInd) - handPos)/maxDepth; % Palm offset from center
    FullFingers = (pos(1:3,(1:5)) - repmat(handPos,[1 5]))/maxDepth; % All sensors
    
    imCnt_=mod(imCnt,maxExamples)+1; % Allow only maxExamples examples per file.
    
    % Simple file number id
    if imCnt_==1
        newId = newId + 1; % New file id
        maxId = newId;
    end
    if newId~=oldId % then create a new set of files
        try
            h5create(sprintf('%s%s%.3i.h5',outDir,'labelQuat_',newId), '/labelQuat', [4, inf],'Chunksize',[4 1] );
            h5create(sprintf('%s%s%.3i.h5',outDir,'labelEul_',newId), '/labelEul', [3, inf],'Chunksize',[3 1] );
            h5create(sprintf('%s%s%.3i.h5',outDir,'labelRot_',newId), '/labelRot', [9, inf],'Chunksize',[9 1] );
            h5create(sprintf('%s%s%.3i.h5',outDir,'labelThumb_',newId), '/labelThumb', [3, inf],'Chunksize',[3 1] );
            h5create(sprintf('%s%s%.3i.h5',outDir,'labelPalm_',newId), '/labelPalm', [3, inf],'Chunksize',[3 1] );
            h5create(sprintf('%s%s%.3i.h5',outDir,'labelFingers_',newId), '/labelFingers', [3*5, inf],'Chunksize',[3*5 1] );
            h5create(sprintf('%s%s%.3i.h5',outDir,'labelAngle_',newId), '/labelAngle', [1, inf],'Chunksize',[1 1] );
            
            h5create(sprintf('%s%s_%.3i.h5',outDir,'data_D_full',newId), '/data_D_full', [imsz,imsz,1,inf], 'Chunksize',[imsz imsz 1 1]);
            h5create(sprintf('%s%s%i_%.3i.h5',outDir,'data_D_',imsz,newId), sprintf('%s%i','/data_D_',imsz), [imsz,imsz,1,inf], 'Chunksize',[imsz imsz 1 1]);
            h5create(sprintf('%s%s%i_%.3i.h5',outDir,'data_D_',imsz/2,newId), sprintf('%s%i','/data_D_',imsz/2), [imsz/2,imsz/2,1,inf], 'Chunksize',[imsz/2 imsz/2 1 1]);
            h5create(sprintf('%s%s%i_%.3i.h5',outDir,'data_D_',imsz/4,newId),  sprintf('%s%i','/data_D_',imsz/4), [imsz/4,imsz/4,1,inf], 'Chunksize',[imsz/4 imsz/4 1 1]);
            
            h5create(sprintf('%s%s%.3i.h5',outDir,'labelHeatmap_',newId), '/labelHeatmap',  [hsz,hsz,lenh,inf], 'Chunksize',  [hsz hsz lenh 1]);
            %h5create(sprintf('%s%s%.3i.h5',outDir,'labelHeatmapBig_',newId), '/label',  [imsz,imsz,lenh,inf], 'Chunksize',  [imsz imsz lenh 1]);
            
            h5create(sprintf('%s%s%.3i.h5',outDir,'labelClass_',newId), '/labelClass',  [imsz,imsz,1,inf], 'Chunksize',  [imsz imsz 1 1]);
            
        catch e
            fprintf('Creating databases error: probably already exist\n');
        end
        oldId = newId;
    end
    
    h5write(sprintf('%s%s%.3i.h5',outDir,'labelQuat_',newId), '/labelQuat', Quat(:),[1 imCnt_] ,[4 1] );
    h5write(sprintf('%s%s%.3i.h5',outDir,'labelEul_',newId), '/labelEul',Eul(:),[1 imCnt_] ,[3 1] );
    h5write(sprintf('%s%s%.3i.h5',outDir,'labelRot_',newId), '/labelRot',Rot(:),[1 imCnt_] ,[9 1] );
    h5write(sprintf('%s%s%.3i.h5',outDir,'labelThumb_',newId), '/labelThumb',Thumb(:),[1 imCnt_], [3 1] );
    h5write(sprintf('%s%s%.3i.h5',outDir,'labelPalm_',newId), '/labelPalm',PalmCenter(:),[1 imCnt_], [3 1] );
    h5write(sprintf('%s%s%.3i.h5',outDir,'labelFingers_',newId), '/labelFingers',FullFingers(:),[1 imCnt_] ,[3*5 1] );
    h5write(sprintf('%s%s%.3i.h5',outDir,'labelAngle_',newId), '/labelAngle',Angle(:),[1 imCnt_] ,[1 1] );
    
    h5write(sprintf('%s%s_%.3i.h5',outDir,'data_D_full',newId), '/data_D_full', D ,[1 1 1 imCnt_],[imsz imsz 1 1]);
    h5write(sprintf('%s%s%i_%.3i.h5',outDir,'data_D_',imsz,newId), sprintf('%s%i','/data_D_',imsz), D_{1} ,[1 1 1 imCnt_],[imsz imsz 1 1]);
    h5write(sprintf('%s%s%i_%.3i.h5',outDir,'data_D_',imsz/2,newId), sprintf('%s%i','/data_D_',imsz/2), D_{2} ,[1 1 1 imCnt_],[imsz/2 imsz/2 1 1]);
    h5write(sprintf('%s%s%i_%.3i.h5',outDir,'data_D_',imsz/4,newId), sprintf('%s%i','/data_D_',imsz/4), D_{3} ,[1 1 1 imCnt_],[imsz/4 imsz/4 1 1]);
    
    h5write(sprintf('%s%s%.3i.h5',outDir,'labelHeatmap_',newId), '/labelHeatmap', HMP ,[1 1 1 imCnt_],[hsz hsz lenh 1]);
    %h5write(sprintf('%s%s%.3i.h5',outDir,'labelHeatmapBig_',newId), '/label', HMP_big ,[1 1 1 imCnt_],[imsz imsz lenh 1]);
    
    h5write(sprintf('%s%s%.3i.h5',outDir,'labelClass_',newId), '/labelClass', LBL ,[1 1 1 imCnt_],[imsz imsz 1 1]);
    
    
    imCnt = imCnt + 1;

    drawnow
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 3: Create listing files of generated data
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

generateList(outDir, outDir, 'labelQuat_',maxId);
generateList(outDir, outDir, 'labelEul_',maxId);
generateList(outDir, outDir, 'labelRot_',maxId);
generateList(outDir, outDir, 'labelThumb_',maxId);
generateList(outDir, outDir, 'labelPalm_',maxId);
generateList(outDir, outDir, 'labelFingers_',maxId);
generateList(outDir, outDir, 'labelAngle_',maxId);

generateList(outDir, outDir, 'data_D_full_',maxId);
generateList(outDir, outDir, sprintf('data_D_%i_',imsz),maxId);
generateList(outDir, outDir, sprintf('data_D_%i_',imsz/2),maxId);
generateList(outDir, outDir, sprintf('data_D_%i_',imsz/4),maxId);
generateList(outDir, outDir, 'labelHeatmap_',maxId);
generateList(outDir, outDir, 'labelClass_',maxId);

%%

