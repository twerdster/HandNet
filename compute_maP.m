function [ap,prec,rec, sconf]= compute_maP(gt,conf,draw)
% gt is a binary vector with a 1 for a correct result and a -1 for an
% incorrect result at each position

% ap is average precision
% prec is precision
% rec is recall
% sconf is sorted confidence according to recall
% So if we wanted to know what confidence is required for
% 80% recall we would find the closest value in recall to 0.8
% and then extract the confidence with the same index

[so,si]=sort(-conf);
tp=gt(si)>0;
fp=gt(si)<0;

fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/sum(gt>0);
prec=tp./(fp+tp);

ap=VOCap(rec,prec);

if draw
    % plot precision/recall
    subplot(121)
    plot(rec,prec,draw);
    grid;
    xlabel 'recall'
    ylabel 'precision'
    xlim([0 1]);ylim([0 1]);
    %title(sprintf('AP = %.3f',ap));
    subplot(122);
    sconf = conf(si);
    plot(rec,sconf,draw);
    xlabel 'recall'
    ylabel 'confidence'
    xlim([0 1]);ylim([0 1]);
end

end

function ap = VOCap(rec,prec)

mrec=[0 ; rec ; 1];
mpre=[0 ; prec ; 0];
for i=numel(mpre)-1:-1:1
    mpre(i)=max(mpre(i),mpre(i+1));
end
i=find(mrec(2:end)~=mrec(1:end-1))+1;
ap=sum((mrec(i)-mrec(i-1)).*mpre(i));

end