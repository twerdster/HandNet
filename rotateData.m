function data = rotateData(dataIn, KK, theta)
% Takes HandNet data and rotates it by an angle theta for both 2d and 3d data

data = dataIn;

p = double([KK(7),KK(8)]);

data.lbl = uint8(rotateAround(data.lbl,p(2),p(1),-theta,'nearest'));

i0 = data.hmap(1,1,1); 

for j=1:size(data.hmap,3)
    data.hmap(:,:,j) =  rotateAround(data.hmap(:,:,j),p(2),p(1),-theta,'bilinear');
end

data.hmap(data.hmap<0.007)=i0;

data.segmap = uint8(logical(rotateAround(data.segmap,p(2),p(1),-theta,'bilinear')));
data.depth = rotateAround(data.depth,p(2),p(1),-theta,'nearest');
%data.depth(~data.segmap)=1;
data.ir = rotateAround(data.ir,p(2),p(1),-theta,'bilinear');

bbox = data.bbox;
pos = data.pos;
rot = data.rot;
handPos = data.handPos;

rot2d  = @(a) [cosd(a) sind(a); -sind(a) cosd(a)];
rot3dz = @(a) [cosd(a) sind(a) 0; -sind(a) cosd(a) 0; 0 0 1];

bboxc = [bbox(1)+bbox(3)/2; bbox(2)+bbox(4)/2]-[p(1) ; p(2)];
bboxc = rot2d(-theta)*bboxc + [p(1) ; p(2)];
data.bbox = [bboxc(1)-bbox(3)/2  bboxc(2)-bbox(4)/2  bbox(3)  bbox(4)];
data.pos = rot3dz(-theta)*pos;
for i=1:size(rot,3)
    data.rot(:,:,i) = rot3dz(-theta)*data.rot(:,:,i);
end

data.handPos = rot3dz(-theta)*handPos;

