% hr_from_dir - Compare heartbeat patterns contained in directory
%
% MAT-files required: extract_values.m
%
% See also: extract_values
% Author: Kevin Jiokeng
% email: kevin.jiokeng@enseeiht.fr
% December 2019; Last revision: 12-December-2019
%------------- BEGIN CODE --------------


directory = '../../measures/hand/pixel/hr/25-02-2020/';
output_dir = './results/hand/pixel/hr/';

start_at = 0;
duration = 15;
period = 0.0194; % Pixel 2
gyr_period = 0.0025; % Pixel 2
% period = 0.005; % Nexus 5
% period = 0.01; % S8
% gyr_period = 0.002;

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
gtt = [71, 63, 71, 56, 83, 54, 59, 61, 67, 72, 70, 52]';

% Plot the values
% fig = figure('units','normalized','outerposition',[0 0 1 1]);
fig = figure('units','normalized','outerposition',[0 0 0.5 1]); % Autocorr
% n_cols = 2;
n_cols = 1; % Autocorr
n_rows = 6;
heart_rates = zeros(num_files, 6); % Save the computed heart rates, for statistics
var_retained = zeros(num_files, 6); % Save the amounts of variance retained
n_meas = zeros(num_files, 2); % Save the number of measures for each user
proj_mats = zeros(num_files, 2*3*3); % Save the projection matrices
n_filtering_points = 3;
for k=1:num_files
	fprintf("Current file: %s\n", heartbeat_files(k).name)

	% Acc
	values = extract_values(filenames{k}, 'start_at', start_at, 'duration', duration, ...
		'period', period, 'sensor', 'ACC');
	t = values(:,1);
	n_meas(k,1) = length(t);
	pause
	continue
	
	% PCA
	% [vals, vr, U] = Helper.pca(values(:,2:end));
	% var_retained(k,1:3) = vr;
	% proj_mats(k, 1:9) = reshape(U, 1, 9);


	id_tile = 1;
	for dim=1:3
		v = values(:,dim+1);
		% v = vals(:,dim); % PCA
		v = v - mean(v);
		% v = Helper.filter_noise(v, 'n_points', n_filtering_points);
		v = v / max(abs(v)); % Normalization
		% v = abs(v);

		one_second = ceil(1/period);
		vt = v(one_second:2*one_second);
		[autocor, lags] = xcorr(vt, v);
		[pks, locs] = findpeaks(autocor, 'MinPeakDist', 0.6/period);
		p = mean(diff(locs)) * period;
		rate = 60/p;
		rate = Helper.compute_heart_rate(v, 'period', period, 'method', 'corr');
		p = 60/rate;
		fprintf("\tDim: Acc.%d, Computed period: %f => %.1fbpm\n", dim, p, rate);
		heart_rates(k, dim) = rate;
		subplot(n_rows, n_cols, id_tile);
		plot(lags*period, autocor)
		ylabel(sprintf("%.1f bpm", rate))
		hold on
		pks = plot(lags(locs)*period,pks,'or');
		hold off
		% pause
		id_tile = id_tile + 1;
		continue

		% Temporal
		subplot(n_rows, n_cols, id_tile);
		plot(t, v, 'r', 'LineWidth', 1)
		ylim([-1 1])
		xlim([start_at start_at+duration])
		switch dim
			case 1
				ylabel({"Acc.x", sprintf("(%.1f bpm)", rate)})
				title("Temp")
				% % title([ids{k}, "PCA - Autocorr"])
				% title([ids{k}, "Autocorr"])
			case 2
				ylabel({"Acc.y", sprintf("(%.1f bpm)", rate)})
			case 3
				ylabel({"Acc.z", sprintf("(%.1f bpm)", rate)})
		end

		% Frequential
		[z, f] = Helper.to_frequential(v, 'period', period, 'fft_size', 16*8096);
		subplot(n_rows, n_cols, id_tile+1);
		plot(60*f, z, 'b', 'LineWidth', 1)
		xlim(60*[0.9 2])
		line([rate rate], ylim(), 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--')
		line([gtt(k) gtt(k)], ylim(), 'Color', 'k', 'LineWidth', 1, 'LineStyle', '--')
		switch dim
			case 1
				title("Freq")
		end	

		id_tile = id_tile + n_cols;
	end



	% Gyr
	values = extract_values(filenames{k}, 'start_at', start_at, 'duration', duration, ...
		'period', gyr_period, 'sensor', 'GYR');
	t = values(:,1);
	n_meas(k,2) = length(t);

	% PCA
	% [vals, vr, U] = Helper.pca(values(:,2:end));
	% var_retained(k,4:6) = vr;
	% proj_mats(k, 10:18) = reshape(U, 1, 9);

	% id_tile = 7;
	id_tile = 4; % Autocorr
	for dim=1:3
		v = values(:,dim+1);
		% v = vals(:,dim); % PCA
		v = v - mean(v);
		% v = Helper.filter_noise(v, 'n_points', floor(n_filtering_points*period/gyr_period));
		v = v / max(abs(v)); % Normalization
		% v = abs(v);

		one_second = ceil(1/gyr_period);
		vt = v(one_second:2*one_second);
		[autocor, lags] = xcorr(vt, v);
		[pks, locs] = findpeaks(autocor, 'MinPeakDist', 0.6/gyr_period);
		p = mean(diff(locs)) * gyr_period;
		rate = 60/p;
		rate = Helper.compute_heart_rate(v, 'period', gyr_period, 'method', 'corr');
		p = 60/rate;
		fprintf("\tDim: Gyr.%d, Computed period: %f => %.1fbpm\n", dim, p, rate);
		heart_rates(k, 3+dim) = rate;
		subplot(n_rows, n_cols, id_tile);
		plot(lags*gyr_period, autocor)
		ylabel(sprintf("%.1f bpm", rate))
		hold on
		pks = plot(lags(locs)*gyr_period,pks,'or');
		hold off
		% pause
		id_tile = id_tile + 1;
		continue	

		% Temporal
		subplot(n_rows, n_cols, id_tile);
		plot(t, v, 'r', 'LineWidth', 1)
		ylim([-1 1])
		xlim([start_at start_at+duration])
		switch dim
			case 1
				ylabel({"Gyr.x", sprintf("(%.1f bpm)", rate)})
			case 2
				ylabel({"Gyr.y", sprintf("(%.1f bpm)", rate)})
			case 3
				ylabel({"Gyr.z", sprintf("(%.1f bpm)", rate)})
				xlabel("Time (s)")
		end

		% Frequential
		[z, f] = Helper.to_frequential(v, 'period', gyr_period, 'fft_size', 16*8096);
		subplot(n_rows, n_cols, id_tile+1);
		plot(60*f, z, 'b', 'LineWidth', 1)
		xlim(60*[0.9 2])
		line([rate rate], ylim(), 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--')
		switch dim
			case 3
				xlabel("Freq (Hz)")
		end	

		id_tile = id_tile + n_cols;
	end

	% mtit(ids{k})
	% saveas(gcf, strcat(output_dir, 'pca-', ids{k}, '.png'));
	mtit(sprintf('%s - corr', ids{k}))
	saveas(gcf, strcat(output_dir, 'hr-', ids{k}, '.png'));
	% saveas(gcf, strcat(output_dir, 'pca-autocorr-', ids{k}, '.png'));
end
