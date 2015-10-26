

function generateList(listOutDir,outDir,fileBase,maxId)
% This function creates a file with a list of files which
% is used when providing a list of training files for caffe

f = fopen(sprintf('%s%s_list.txt',listOutDir,fileBase),'wt');

for i=0:maxId
    fwrite(f,sprintf('%s%s%.3i.h5\n',outDir,fileBase,i));    
end
fclose(f);