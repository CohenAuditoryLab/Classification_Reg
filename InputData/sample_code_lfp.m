clear
clc

% load data
load 20180727_ABBA_LFP%primary auditory cortex
% load 20190123_ABBA_LFP % non-primary auditory cortex


% flatten the array with 3052_data went across all trials and then all channels
LFP_flat=reshape(permute(LFP,[2,3,1]),size(LFP,2),size(LFP,1)*size(LFP,3))';
all_channel_flat=reshape(repmat([0:param.nChannel-1],size(LFP,3),1),size(LFP,3)*size(LFP,1),1);
leverRelease_flat=repmat(leverRelease,size(LFP,1),1);
LeverReleaseTime_flat=repmat(LeverReleaseTime,size(LFP,1),1);
targetTime_flat=repmat(targetTime,size(LFP,1),1);
index_flat=repmat(index,size(LFP,1),1);
stimOn_flat=repmat(stimOn,size(LFP,1),1);
trialNum = repmat([1:size(LFP,3)]',size(LFP,1),1);

% filter out useful trials
IND = find(index_flat<=2);
New_index = index_flat(IND);
New_leverRelease = leverRelease_flat(IND);
New_LeverReleaseTime = LeverReleaseTime_flat(IND);
New_LFP = LFP_flat(IND,:)*10e3;% convert unit to mV
New_targetTime = targetTime_flat(IND);
New_stimOn=stimOn_flat(IND);
New_all_channel=all_channel_flat(IND);
New_trialNum=trialNum(IND);

% New_Rew = Rew(IND,:);
% New_stDiff = stDiff(IND);
% New_Stim1 = Stim1(IND,:);
% New_Stim2 = Stim2(IND,:);
% New_stimOn = stimOn(IND);
% New_stimOnTime = StimOnTime(IND);

% ST = unique(New_stDiff);
% TargetTime = unique(New_targetTime);

% response time vs target time
ResTime = New_leverRelease-New_stimOn-New_targetTime;


% % get the LFP 600ms before the onset of target
% LFPBefTar=[];
% for i = 1:size(New_LFP,1)
%     t_temp=abs(t-New_targetTime(i));
% % LFPBefTar(i,:) = New_LFP(i,find(t_temp==min(t_temp))-500:find(t_temp==min(t_temp))+299);
% LFPBefTar(i,:) = New_LFP(i,find(t_temp==min(t_temp))-499:find(t_temp==min(t_temp)))-fliplr(New_LFP(i,find(t_temp==min(t_temp))+1:find(t_temp==min(t_temp))+500));
% 
% end
 
behave_data = New_index;
save('LFP_RT_behave_AC_neuronwise.mat','New_LFP','ResTime','behave_data','New_trialNum','New_targetTime','New_all_channel','t')

% figure;
% for i = 1:length(TargetTime)
%     y_pos = 0;
%     subplot(2,2,i)
% %     ind = intersect(intersect(find(New_index==0),find(New_stDiff==24)),find(New_targetTime==TargetTime(i)));
%     ind = intersect(find(New_index==0),find(New_targetTime==TargetTime(i)));
% 
%     for j=1:16
%         plot(t,mean(New_LFP(j,:,ind),3)+y_pos);
%         y_pos = y_pos - 1;
%         hold on
%     end
%     set(gca,'YTick',-15:0,'YTickLabel',16:-1:1);
%     xlabel('Time [ms]'); ylabel('Channel');
%     title(['TargetTime: ',num2str(TargetTime(i)),'ms'])
% end


%
% % obtain hit trials
% lfp_hit = LFP(:,:,index==0); % LFP
% df_hit = stDiff(index==0); % delta frequency
% tt_hit = targetTime(index==0); % target time
%
% % 24 semitone difference trials
% lfp_hit_24df = lfp_hit(:,:,df_hit==24);
% tt_hit_24df = tt_hit(df_hit==24);
%
% lfp_hit_24df = lfp_hit_24df * 10e3; % convert unit to mV
%
% % plot mean LFP
% target = 1125; % set target time
% y_pos = 0;
% figure; hold on
% for i=1:16
%     plot(mean(LFPBefTar(intersect(find(New_all_channel==i),find(behave_data==0)),:))-mean(LFPBefTar(intersect(find(New_all_channel==i),find(behave_data==1)),:))+y_pos);
% %     y_pos = y_pos - 1;
% end
% % set(gca,'YTick',-15:0,'YTickLabel',16:-1:1);
% xlabel('Time [ms]'); ylabel('Channel');