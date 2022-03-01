% plot_spectrograms_from_dir - Plot the spectrograms of patterns contained in directory
%
% MAT-files required: extract_values.m
%
% See also: extract_values
% Author: Kevin Jiokeng
% email: kevin.jiokeng@enseeiht.fr
% January 2020; Last revision: 09-January 2020
%------------- BEGIN CODE --------------


% directory = '../../measures/chest/comp-08-11-19';
directory = '../../measures/chest/comp-12-11-19-unique';
directory = '../../measures/hand/pixel/';
directory = '../../measures/hand/nexus/gyro/';
directory = '../../measures/hand/s8/analysis/';
% directory = '../../measures/tmp';

output_dir = './results/hand/s8/viz/';

start_ats = 0;
duration = 25;
% period = 0.0194; % Pixel 2
% period = 0.005; % Nexus 5
period = 0.01; % S8
gyr_period = 0.002;

heartbeat_files = dir(directory);
heartbeat_files = heartbeat_files(3:end);
num_files = length(heartbeat_files);

filenames = {};
ids = {};
ages = [];
genders = []; % true for MALE
for k=1:num_files
	filenames{k} = strcat(heartbeat_files(k).folder, '/', heartbeat_files(k).name);

	% Extract the user id from the filename
	tmp = strsplit(filenames{k}, '/');
	tmp = tmp{end};
	tmp = strsplit(tmp, '-');
	ids{k} = tmp{1};
	% ids{k} = strcat(tmp{1}, '-', tmp{5}); % Take the id and the position

	genders(k) = strcmp(tmp{2}, 'MALE');
	ages(k) = str2num(strrep(tmp{3}, 'yrs', ''));
end

% Plot the values
% fig = figure('units','normalized','outerposition',[0 0 1 1]);
fig = figure('units','normalized','outerposition',[0 0 1 0.5]); % Autocorr
n_cols = 1;
n_rows = ceil(length(start_ats)/n_cols);
vv = zeros(num_files, 510, 3);
for k=1:num_files
	fprintf("Current file: %s\n", heartbeat_files(k).name)

	for s=1:length(start_ats)
		start_at = start_ats(s);
		% Acc
		values = extract_values(filenames{k}, 'start_at', start_at, 'duration', duration, ...
			'period', period, 'sensor', 'ACC');
		t = values(:,1);
		l = length(t);
		% vv(k, 1:l, :) = values(:,2:end) - mean(values(:,2:end));

		% pause
		% continue
		
		% PCA
		% [vals, vr, U] = Helper.pca(values(:,2:end));
		% v = vals(:, 1);
		% v = Helper.filter_noise(v);

		% subplot(n_rows, n_cols, s);
		% spectrogram(v, 16, [], 512, 1/period, 'yaxis', 'power');
		% instfreq(v, 1/period);
		% c = spectrogram(v, 16, [], 512, 1/period, 'yaxis', 'power');
		for dim=1:3
			if dim==1
				title(strcat("CWT - ", ids{k}))
			end
			subplot(3, 1, dim)
			v = values(:, dim);
			cwt(v, 1/period);
		end

		% c = abs(c);
		% c = abs(c(floor(end/2):end,:));

		% image(c, 'CDataMapping','scaled')
		% colormap default
		% colorbar
		% grid on

		% title(strcat("CWT - ", ids{k}))
		% pause
	end

	% sgtitle(strrep(method, '_', '\_'))
	% mtit(ids{k})
	% saveas(gcf, strcat(output_dir, 'pca-', ids{k}, '.png'));
	% mtit(sprintf('%s - PCA', ids{k}))
	% saveas(gcf, strcat(output_dir, 'autocorr-', ids{k}, '.png'));
	% saveas(gcf, strcat(output_dir, 'spectrogram-pca-', ids{k}, '.png'));
	saveas(gcf, strcat(output_dir, 'cwt-', ids{k}, '.png'));
end
