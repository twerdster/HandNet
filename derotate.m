function [alpha, pointId, upId, toCam] = derotate2(Rin)

% alpha - Derotation angle. use this angle to perform in-plane derotation around the center of an image
% pointId,upIdtoCam can be used as classifiers determining which axis is pointing upwards and towards the camera

%Can consider this as being a clamp-to-nearest function
%for each of the possible states of the right handed axis.

 % Find the axis with a z component 

[~,pointId] = max(abs([1 1 1]'.*diag([0 0 0; 0 0 0; 1 1 1]'*Rin)));

% Rout(3,pointId)<0 means pointing towards camera
toCam = Rin(3,pointId)<0; 
switch (pointId)
    case {1,3} %red,blue
        
        upId=2;     
        alpha = double(atan2d(Rin(1,upId),Rin(2,upId))) + 90;
     case 2,%green
         
        upId=3;      
        alpha = double(atan2d(Rin(1,upId),Rin(2,upId))) + 90 + single(toCam)*180;
end

function w = atan2d(x,y)
rad2deg = @(x) x/pi*180;
w = rad2deg(atan2(x,y));