%% Testing

% The results are loaded here into a structure called Results.
% It has the following format
% Results.(dataset).(learner).(derotater);
% The dataset type is either base (TrainData.mat) or
% derotgt (TrainData_DerotGT.mat)
load('TestData_Perturbed_Results.mat');

%%
num = 5
M = 10000;
thr = 15; % Threshold distance in millimetres

% which_GT defines the groundtruth used to measure the success of the learner
% groundtruth_XYZ was created from the generated labels and is effectively a proxy for
% the sensor data. sensor_XYZ are the adjusted 3d locations of the sensors.
% In fact in the paper we dont use either. Instead we use the score.tip results
% which were computed by finding the pixel distance between groundtruth 2d and predicted 2d.
% But that is somewhat less useful in most contexts.
which_GT = 'groundtruth_XYZ' ; % groundtruth_XYZ  sensor_XYZ

% The derotater is the method that was used to move an input image into a canonical
% orientation according to the predicted orientation. The 'base' derotater doesnt
% do any derotation and leaves the data as is. The rest of the methods are groundtruth (gt),
% cnn prediction and procrustes like prediction (pca)
derotater1 = 'base'; % base derotgt derotcnn derotpca <-- RED
derotater2 = 'derotcnn'; % base derotgt derotcnn derotpca <-- GREEN

% This determines the learner that was tested. The forest (which is actually only a single tree) is 'for'
% and the convolutional network is 'cnn'
learner = 'for'; % for cnn

for i=1:num
    fprintf('Index: %i\n',i);
   
    A=Results.base.(learner).(derotater1);
    ap=[A.predicted_XYZ];
    agt=[A.(which_GT)];
    ac = [A.conf];
    ac = reshape(ac, [], M);
    da = abs(agt(1:num,:)-ap(1:num,:));
    da = reshape(da,[num 3 M]);
    sda = squeeze(sum(da.^2,2));
    gta = sda(:,:)<thr^2;
    
    B=Results.derotgt.(learner).(derotater2);
    bp=[B.predicted_XYZ];
    bgt=[B.(which_GT)]; 
    bc = [B.conf];
    bc = reshape(bc, [], M);
    db = abs(bgt(1:num,:)-bp(1:num,:));
    db = reshape(db,[num 3 M]);
    sdb = squeeze(sum(db.^2,2));
    gtb = sdb(:,:)<thr^2;
    
    %RES = [sum(gta,2) sum(gtb,2)]
    
    subplot(121);cla; subplot(122); cla;
    [ap,prec,rec, sconf]= compute_maP(gta(i,:)'*2-1,ac(i,:)','r');
    subplot(121);hold on; subplot(122); hold on;
    [ap,prec,rec, sconf]= compute_maP(gtb(i,:)'*2-1,bc(i,:)','r.');
    
    pause
end