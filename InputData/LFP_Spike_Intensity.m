clear
clc
% load data
%primary auditory cortex
load ('20180727_activeCh','trialInfo','clInfo','Raster','t_raster') 
load('20180727_ABBA_LFP','LFP','leverRelease','stimOn','t')
load 20180727_StimSequence_dB 
str='AC';

% % non-primary auditory cortex
% load ('20190123_activeCh','trialInfo','clInfo','Raster','t_raster') 
% load('20190123_ABBA_LFP','LFP','leverRelease','stimOn','t')
% load 20190123_StimSequence_dB 
% str='Belt';

% filter out useful trials
Trialindex = find(trialInfo.behav_ind<=2);
% filter out active neural clusters
ActiveIndex = find(ismember(clInfo.all_cluster,clInfo.active_cluster)==1);

% filter raster data 
New_Raster = Raster(Trialindex,:,ActiveIndex);
target_time = trialInfo.target_time(Trialindex);
semitone_diff = trialInfo.semitone_diff(Trialindex);
behav_ind = trialInfo.behav_ind(Trialindex);
channelLab_R = clInfo.active_channel+1; % change channel data range to 1-16
neuronLab_R = clInfo.active_cluster;

% filter LFP data
New_LFP = permute(LFP(:,:,Trialindex),[3,2,1]);
NewLeverRlsTime = leverRelease(Trialindex)-stimOn(Trialindex);
t_LFP=t;

% filter stimu_db
startErrindex = find(trialInfo.behav_ind~=3);
index = find(trialInfo.behav_ind(startErrindex)<=2);
stim_db=stim_db(index,:);

soundOffTime=NewLeverRlsTime;
for i = 1:length(behav_ind)
    if behav_ind(i)==1
       soundOffTime(i)= target_time(i)+600-25;
    end
    if behav_ind(i)==0 & soundOffTime(i)>2225
       soundOffTime(i)=2225;
    end
end

save(['LFP_Raster_ToneIntensity_',str,'.mat'],'New_Raster','target_time','semitone_diff','behav_ind',...
    'channelLab_R','neuronLab_R','New_LFP','soundOffTime','t_raster','t_LFP','stim_db')