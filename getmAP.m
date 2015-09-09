function [ap,p_at_100,p_at_80, prec,rec, sconf]= getmap(gt,tips,conf,draw)
% compute precision/recall
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

% find precision at 100% recall and at 80% recall
[~,ind100] = min(abs(rec-1.0));
p_at_100 = prec(ind100);
[~,ind80] = min(abs(rec-0.8));
p_at_80  = prec(ind80);

ap=VOCap(rec,prec);

if draw
    % plot precision/recall
    subplot(121)
    [x,ind] = unique(rec); y=prec(ind); xi = linspace(0.0,1,500);; yi = interp1(x,y,xi);
    %plot(xi(end),yi(end),'k.','MarkerSize',20);
    h = plot(xi,yi,draw,'LineWidth',2);
     
    %title(sprintf('AP = %.3f',ap));
    
    
    % Plot threshold precisions
    subplot(122)
    
    thrLim = 2*6;
    xlim([0 thrLim]);ylim([0 1]);
    thr = linspace(0,thrLim,20);
    total = sum(tips<1e4);
    for i=1:length(thr)
        tp_=sum(tips<thr(i));
        fp_=sum(tips>thr(i));
        
        rec_=tp_/total;
        prec_=tp_/(fp_+tp_);
        
        suma(i) = prec_;
    end
    %plot(thr(end/2)/6,suma(end/2),'k.','MarkerSize',20);
    plot(thr/6,suma,draw,'LineWidth',2);
    
    
    %     % Plot the confidence
    %     subplot(133);
    %     sconf = conf(si);
    %     plot(sconf,prec,draw,'LineWidth',3);
    %     xlabel 'confidence'
    %     ylabel 'precision'
    %     xlim([0 1]);ylim([0 1]);
    %     set(gca,'XTick',[0:0.2:1],'YTick',[0:0.2:1]);
    %     gridxy(get(gca,'xtick'),get(gca,'ytick'),'color',[.8 .8 .8],'linewidth',1);
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