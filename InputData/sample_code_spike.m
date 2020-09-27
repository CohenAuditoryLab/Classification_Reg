clear
clc
% load data
% load 20180727_activeCh %primary auditory cortex
load 20190123_activeCh % non-primary auditory cortex

% flatten the array: with 3000_data went across all clusters and then all trials
Raster_flat=reshape(permute(Raster,[2,3,1]),size(Raster,2),size(Raster,1)*size(Raster,3))';
clInfo.all_cluster_flat=repmat(clInfo.all_cluster',size(Raster,1),1);
clInfo.channel_flat=repmat(clInfo.channel',size(Raster,1),1);
lever_release_flat=reshape(repmat(lever_release,1,size(Raster,3))',size(Raster,3)*size(Raster,1),1);
trialInfo.target_time_flat=reshape(repmat(trialInfo.target_time,1,size(Raster,3))',size(Raster,3)*size(Raster,1),1);
trialInfo.semitone_diff_flat=reshape(repmat(trialInfo.semitone_diff,1,size(Raster,3))',size(Raster,3)*size(Raster,1),1);
trialInfo.behav_ind_flat=reshape(repmat(trialInfo.behav_ind,1,size(Raster,3))',size(Raster,3)*size(Raster,1),1);
SpikeTiming_flat=reshape(repmat(SpikeTiming,1,size(Raster,3))',size(Raster,3)*size(Raster,1),1);
trialNum = reshape(repmat([1:size(Raster,1)],size(Raster,3),1),[],1);
% filter out useful trials
Trialindex = find(trialInfo.behav_ind_flat<=2);
% filter out active neural clusters
ActiveIndex = find(ismember(clInfo.all_cluster_flat,clInfo.active_cluster)==1);
% % delete zero spike vectors in raster data
% ZeroInd = find(sum(Raster_flat,2)>0);

index=intersect(Trialindex,ActiveIndex);
New_Raster = Raster_flat(index,:);
clInfo.all_cluster_flat_New = clInfo.all_cluster_flat(index);
clInfo.channel_flat_New=clInfo.channel_flat(index);
New_lever_release =  lever_release_flat(index);
New_trialInfo.target_time = trialInfo.target_time_flat(index);
New_trialInfo.semitone_diff = trialInfo.semitone_diff_flat(index);
New_trialInfo.behav_ind = trialInfo.behav_ind_flat(index);
New_SpikeTiming = SpikeTiming_flat(index);
New_trialNum = trialNum(index);

% response time vs target time
ResTime = New_lever_release-New_trialInfo.target_time; 
behave_data = New_trialInfo.behav_ind;
neuron_lab = clInfo.all_cluster_flat_New;
channel_lab = clInfo.channel_flat_New;
targetT = New_trialInfo.target_time;
semiToneDiff = New_trialInfo.semitone_diff;

% display trials/performance distribution in histogram  
figure;
histogram2(neuron_lab,behave_data,'BinMethod','integers')
xlabel('NeuroLab')
xticks(unique(neuron_lab))
ylabel('BehavPerform')
zlabel('Trials#')
set(gca,'YTick',[0,1,2,3,4],'YTickLabel',{'Hit','Miss','FA','StError','TError'})

% display channel & neuron distribution in histogram
qq=unique([channel_lab,neuron_lab],'rows');
for i = 1:length(qq)
    % hit trial
    qq(i,3)=length(intersect(intersect(find(channel_lab==qq(i,1)),find(neuron_lab==qq(i,2))),find(behave_data==0)));
    % miss trial
    qq(i,4)=length(intersect(intersect(find(channel_lab==qq(i,1)),find(neuron_lab==qq(i,2))),find(behave_data==1)));
    % false alarm trial
    qq(i,5)=length(intersect(intersect(find(channel_lab==qq(i,1)),find(neuron_lab==qq(i,2))),find(behave_data==2)));
end
figure;
scatter3(qq(:,1),qq(:,2),qq(:,3)+qq(:,4)+qq(:,3),'filled')
xticks(unique(qq(:,1)))
xlabel('channel')
yticks(unique(qq(:,2)))
ylabel('NeuroLab')
zlabel('Trials#')


% % get the raster 75ms/125ms before/after the onset of target
% TargetTime = unique(New_trialInfo.target_time);
% Raster_cut=[];
% for i = 1:length(TargetTime)
%     ind_temp=find(New_trialInfo.target_time==TargetTime(i));
%     Raster_cut(ind_temp,:) = New_Raster(ind_temp,find(t_raster==TargetTime(i))-300:find(t_raster==TargetTime(i))+499);
% end
% 
% 
% Firerate=[];
% % get firing rate
% 
% SmthWind=75;%ms
% 
% for i=1:size(Raster_cut,1)
%     for j =1: (size(Raster_cut,2)-SmthWind+1)
%         Firerate(i,j)=sum(Raster_cut(i,j:j+SmthWind-1))/(SmthWind/1000);
%     end    
% end
% 
% %zero score firing rate of each neuron's raster series
% neurons=unique(neuron_lab);
% Zscore_Firerate=[];
% for k=1:length(neurons)
%    ind_temp=find(neuron_lab==neurons(k)); 
%    Firerate_temp=reshape(Firerate(ind_temp,:),1,[]);
%    Mean_Firate=mean(Firerate_temp); 
%    Std_Firate= std(Firerate_temp);
%    Zscore_Firerate(ind_temp,:)=(Firerate(ind_temp,:)-Mean_Firate)./Std_Firate;   
% end

% %zero score firing rate of each neuron's raster series
% comlab=[neuron_lab,semiToneDiff];
% combs=unique(comlab,'rows');
% Zscore_Firerate=[];
% for k=1:length(combs)
%    ind_temp=intersect(find(comlab(:,1)==combs(k,1)),find(comlab(:,2)==combs(k,2))); 
%    Firerate_temp=reshape(Firerate(ind_temp,:),1,[]);
%    Mean_Firate=mean(Firerate_temp); 
%    Std_Firate= std(Firerate_temp);
%    Zscore_Firerate(ind_temp,:)=(Firerate(ind_temp,:)-Mean_Firate)./Std_Firate;   
% end

 
% % combine firerates of numNeurns of neurons in each trial
% numNeurns=2;
% uniqtriNum=unique(New_trialNum);
% combInd=[];
% comb_behave_data=[];
% for u=1:length(uniqtriNum)
%     ind=find(New_trialNum==uniqtriNum(u));
%     combInd_temp=nchoosek(ind,numNeurns);
%     comb_behave_data=[comb_behave_data;repmat(behave_data(ind(1)),size(combInd_temp,1),1)];
%     combInd=[combInd;combInd_temp];
% end
% combZscore_Firerate=[Zscore_Firerate(combInd(:,1),:),Zscore_Firerate(combInd(:,2),:)];

save('Spike_RT_behave_nonAC_neuronwise.mat','targetT','New_trialNum','New_Raster','semiToneDiff','neuron_lab','channel_lab','ResTime','behave_data')









% figure;
% for i =1:length(TargetTime)
%     subplot(2,2,i)
%     histogram(ResTime(intersect(intersect([find(New_trialInfo.behav_ind==0);find(New_trialInfo.behav_ind==2)],find(New_trialInfo.semitone_diff==24)),find(New_trialInfo.target_time==TargetTime(i)))),'BinWidth',100,'BinLimits',[-1400,600])
%     title(['TargetTime: ',num2str(TargetTime(i)),'ms'])
%     xlabel('ResponseTime(ms)')
%     ylabel('Trials#')
% end
% 
% % raster
% for j = 1:length(TargetTime)
%     figure(j);
%     figure('Name',['24ST-',num2str(TargetTime(j)),'ms'])
%     % obtain hit trials for ST(j) diff
%     raster_temp = New_Raster(intersect(intersect(find(New_trialInfo.behav_ind==0),find(New_trialInfo.semitone_diff==24)),find(New_trialInfo.target_time==TargetTime(j))),:,:); %
%     for i = 1:length(ActiveIndex)
%         % choose unit
%         subplot(2,7,i)
%         % display raster plot
%         imagesc(raster_temp(:,:,i));
%         colormap(1-gray);
%         set(gca,'XTick',500:500:3000,'XTickLabel',0:5:25);
%         xlabel('Time [100ms]'); ylabel('Trial');
%         hold on
%     end
% end