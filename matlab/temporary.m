% sensors_values = old_val;

% deg = 3;

% v = sensors_values(:,1);
% t = 1:length(v);
% t = t';
% [p,s,mu] = polyfit(t,v,deg);
% f_y = polyval(p,t,[],mu);
% v = v - f_y;
% %v = v + rand(length(v), 1)/16 - 0.4;
% sensors_values(:,1) = v;

% v = sensors_values(:,2);
% t = 1:length(v);
% t = t';
% [p,s,mu] = polyfit(t,v,deg);
% f_y = polyval(p,t,[],mu);
% v = v - f_y;
% sensors_values(:,2) = v;

% v = sensors_values(:,3);
% t = 1:length(v);
% t = t';
% [p,s,mu] = polyfit(t,v,deg);
% f_y = polyval(p,t,[],mu);
% v = v - f_y;
% sensors_values(:,3) = v;


% old_sensors_values = sensors_values;
% [sensors_values, var_ret, U, S] = Helper.pca(sensors_values);

% positions = [1.5 0 -1.5];
% subplot(2, 1, 1)
% plot(old_sensors_values+positions)
% legend("X", "Y", "Z")

% subplot(2, 1, 2)
% plot(sensors_values+positions*4)
% legend("X", "Y", "Z")

% aa = [old_sensors_values, sensors_values]




% w = abs(WT);
% [m, n] = size(w);
% for i=1:m
% 	fprintf("%.5f", w(i, 1));
% 	for j=2:n
% 		fprintf(",%.5f", w(i, j));
% 	end
% 	fprintf("\n");
% end

% %% Save to HDF5 file
% h5filename = 'handrate-data.h5';
% h5create(h5filename, '/scalogram_16_voices_per_octave_trans', size(w'))
% h5write(h5filename, '/scalogram_16_voices_per_octave_trans', w')

% h5create(h5filename, '/F_scalogram_16_voices_per_octave', size(F))
% h5write(h5filename, '/F_scalogram_16_voices_per_octave', F)

% h5create(h5filename, '/T_scalogram_16_voices_per_octave', size(T))
% h5write(h5filename, '/T_scalogram_16_voices_per_octave', T)

% % Spectrogram
% h5create(h5filename, '/spectrogram_99percent_0leakage', size(P))
% h5write(h5filename, '/spectrogram_99percent_0leakage', P)

% h5create(h5filename, '/F_spectrogram_99percent_0leakage', size(F))
% h5write(h5filename, '/F_spectrogram_99percent_0leakage', F)

% h5create(h5filename, '/T_spectrogram_99percent_0leakage', size(T))
% h5write(h5filename, '/T_spectrogram_99percent_0leakage', T)



% cfs = WT;
% f = F;
% image(abs(cfs),'XData',t,'YData',f,'CDataMapping','scaled')
% % set(gca,'YScale','log')
% % axis tight
% xlabel('Time (s)')
% ylabel('Frequency (Hz)')
% set(gca,'colorscale','log')

% 'XData',t,'YData',f,

% surface(t, F, abs(WT))
% axis tight
% shading flat
% xlabel('Time (s)')
% ylabel('Frequency (Hz)')
% set(gca, 'yscale', 'log')
% yticks([0 0.1 0.3 1 4 16 50])


freq = 100;
cwt_voices_per_octave = 16;
cwt_time_bandwidth = 10;
cwt_freq_limits = [2 50];

t = 1:2800;
t = 1101:2100;
t = 101:2100;
l = length(t);
t = (1:l)/100;
x = sensors_values_save(1:l, 1);
ecg = ecg_values_save(1:l);

subplot(2, 1, 1)
plot(t, 0.8 * ecg / max(ecg))
hold on
target = zeros(l, 1);
[pks, locs] = findpeaks(ecg, 'MinPeakDistance', 80);
target(locs) = 1;
target(locs-1) = 1;
target(locs+1) = 1;
target(locs-2) = 1;
target(locs+2) = 1;
target(locs-3) = 1;
target(locs+3) = 1;
plot(t, target)
xlim([min(t) max(t)])
xticks(0:2:max(t))
xticklabels([])
legend("ECG", "Target Output")

subplot(2, 1, 2)
[wt, f] = cwt(x, freq, ...
				'VoicesPerOctave', cwt_voices_per_octave, ...
			    'TimeBandWidth', cwt_time_bandwidth, ...
			    'FrequencyLimits', cwt_freq_limits);
surface(t, f)
axis tight
shading flat
xlabel('Time (s)')
% ylabel('Frequency (Hz)')
set(gca, 'yscale', 'log')
yticks([0 0.1 0.3 1 4 16 50])
xticks(0:2:max(t))
colormap(special_colormap)
drawnow

close all
m = max(abs(wt));
% findpeaks(m, 'MinPeakDistance', 50)
[pks, locs] = findpeaks(m, 'MinPeakDistance', 50);
res1 = [t', m'];
res2 = [locs'/100, pks'];

w = res1;
[m, n] = size(w);
for i=1:m
	fprintf("%.5f", w(i, 1));
	for j=2:n
		fprintf(",%.5f", w(i, j));
	end
	fprintf("\n");
end