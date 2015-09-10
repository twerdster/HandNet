%% Handnet Database
% Download from : http://www.cs.technion.ac.il/~twerd/HandNet/
% This database and code is made available for academic purposes only.
% For commercial applications please contact the authors.

% Please cite the following if you use it in your work:
%
% @inproceedings{WetzlerBMVC15,
% 	title={Rule Of Thumb: Deep derotation for improved fingertip detection},
% 	author={Aaron Wetzler and Ron Slossberg and Ron Kimmel},
% 	year={2015},
% 	month={September},
% 	pages={33.1-33.12},
% 	articleno={33},
% 	numpages={12},
% 	booktitle={Proceedings of the British Machine Vision Conference (BMVC)},
% 	publisher={BMVA Press},
% 	editor={Xianghua Xie, Mark W. Jones, and Gary K. L. Tam},
% 	isbn={1-901725-53-7},
% }

% Contact: aaronwetzler@gmail.com
% http://www.cs.technion.ac.il/~twerd/HandNet/

%% Choose a folder to visualize its data

% Each folder is created by selecting a subset of images from the
% final database after it has been filtered for quality and shuffled. 
% There are 2773  validation files, 10000 test files and 202198 training 
% files. The database was constructed from 10 participants (6 male, 
% 4 female) each recorded for approximately 45 minutes. Although 
% the images were sequentially recorded there are a number of time gaps. 
% The ordering of the images is stored in each files 'idx' field. If 
% necessary the full database before the train/test/validation split and 
% random shuffling could be reconstructed by copying all the files into 
% one folder and renaming each according to their 'idx' field.
% For convenience the ordered file names are stored in Indices.mat

% Choose a folder to visualize
dirName = 'ValidationData'; % 'TrainData', 'TestData'

d_ = dir([dirName '/Data*.mat']);

% Get intrinsic matrix of depth camera for converting depth to point cloud 
load('Parameters.mat');

% Get ordered data indices (the supplied data folders are already shuffled)
load('Indices.mat');
[~, dataName, ~] = fileparts(dirName); 
inds = eval(lower(dataName(1:end-4)));

% Setup ray directions for 3D visualization of point clouds
[x,y] = meshgrid((1:320)-1,(1:240)-1);
preDepth = ((inv(KK)*[x(:) y(:) y(:)*0+1]')');

%% Visualize available data


showOrdered = true; % Reorganize the data chronologically
chooseSession = 0; % 0 - 9.  If we show the reorganized data then we also choose a session. 
skipFrames = 1; % Skip through frames


%% Go through data

if showOrdered == true
    [~,J] = sort(inds.gID);
    d = d_(J);
    d = d(inds.sorted_sID == chooseSession);
else
   d = d_; 
end

colormap jet

for i=1:skipFrames:length(d)
    
    data = load([dirName '/' d(i).name]);
    
    fprintf('%s, session frame: %i, global frame: %i\n',data.session, data.idx, data.gidx);

    %% ------------- Show Depth image -------------
    if 1
        subplot(221); cla;
        
        imagesc(data.depth,[0 700]); hold on;
        rectangle('Position',data.bbox, 'EdgeColor',[1 0 0]);
    end
    
    %% ------------- Show Label ground truth -------------
    if 1
        subplot(222); cla;
        
        imagesc(data.lbl,[0 7]);
    end
    
    %% ------------- Show Heatmap locations on the same image -------------
    if 1
        subplot(223); cla;
        
        % Heatmaps were created by taking exp(5*(distance2Sensor-1))
        % so lets normalize between 0 and 1
        minVal = min(data.hmap(:)); % should be exp(-5)
        maxVal = 1;
        data.hmap = (data.hmap-minVal)/(maxVal-minVal);
        % Images with the sensor out of the frame include nans
        % which we need to remove.
        data.hmap(isnan(data.hmap))=0;
        
        imagesc(sum(data.hmap,3));
    end
    
    %% ------------- Show 3d data -------------
    if 1
       
       subplot(224); cla; hold on;
       % Get the 3d point cloud from the camera diluted to every "skip" frames
       skip = 5;
       depth = repmat(single(data.depth(1:skip:end)'),1,3);
       xyz = preDepth(1:skip:end,:).*depth;
 
       % Show pointcloud
       plot3(xyz(:,1),xyz(:,2),xyz(:,3),'m.','MarkerSize',2);
       axis equal;
       axis([-300 300 -300 300 0 1000]);
       
       % Get sensor locations and rotations
       % (note that these have been aligned and centered to each finger
       % per session)
       pos = data.pos;
       rot = reshape(data.rot,3,numel(data.rot)/3);
       
       % Show sensors
       len = 60; 
       rot = kron(pos,[1 1 1]) + rot*len;
       plot3(data.pos(1,:),data.pos(2,:),data.pos(3,:),'r.','MarkerSize',20);
       
       h=line([pos(1,:); rot(1,1:3:end)],[pos(2,:); rot(2,1:3:end)],[pos(3,:); rot(3,1:3:end)]);  arrayfun(@(h_)set(h_,'Color',[1 0 0],'LineSmoothing','on', 'LineWidth',3),h);
       h=line([pos(1,:); rot(1,2:3:end)],[pos(2,:); rot(2,2:3:end)],[pos(3,:); rot(3,2:3:end)]);  arrayfun(@(h_)set(h_,'Color',[0 1 0],'LineSmoothing','on', 'LineWidth',3),h);
       h=line([pos(1,:); rot(1,3:3:end)],[pos(2,:); rot(2,3:3:end)],[pos(3,:); rot(3,3:3:end)]);  arrayfun(@(h_)set(h_,'Color',[0 0 1],'LineSmoothing','on', 'LineWidth',3),h);
       
	   if 1 % Plot 3d Bounding box using our detected hand center. Can replace data.handPos for data.pos(:,6) to get the sensor location
          cx = data.handPos(1);cy = data.handPos(2);cz = data.handPos(3);
          sz = 250/2;
          plot3(cx+[-1 +1  +1 -1 -1]*sz,cy+[+1 +1 -1 -1 +1]*sz,cz+[-1 -1 -1 -1 -1]*sz,'r');
          plot3(cx+[-1 +1  +1 -1 -1]*sz,cy+[+1 +1 -1 -1 +1]*sz,cz+[+1 +1 +1 +1 +1]*sz,'r');
          
          plot3(cx+[-1 -1 -1 -1 -1]*sz,cy+[+1 +1 -1 -1 +1]*sz,cz+[-1 +1  +1 -1 -1]*sz,'r');
          plot3(cx+[+1 +1 +1 +1 +1]*sz,cy+[+1 +1 -1 -1 +1]*sz,cz+[-1 +1  +1 -1 -1]*sz,'r');
       end
	   
       if ~exist('setview')
          view(25,10);
          setview = 1;
       end
       
    end
    
    drawnow;
end

