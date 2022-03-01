% plot_values_from_dir - Compare heartbeat patterns contained in directory
%
% MAT-files required: extract_values.m
%
% See also: extract_values
% Author: Kevin Jiokeng
% email: kevin.jiokeng@enseeiht.fr
% December 2019; Last revision: 12-December-2019
%------------- BEGIN CODE --------------


% directory = '../../measures/chest/comp-08-11-19';
directory = '../../measures/chest/comp-12-11-19-unique';
directory = '../../measures/hand/pixel/';
directory = '../../measures/hand/nexus/gyro/';
directory = '../../measures/hand/s8/analysis/';
% directory = '../../measures/hand/pixel/analysis2/';
% directory = '../../measures/hand/pixel/tmp/';
% directory = '../../measures/tmp';

output_dir = './results/hand/s8/viz/';

start_at = 5;
duration = 5;
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
	
	% PCA
	[vals, vr, U] = Helper.pca(values(:,2:end));
	var_retained(k,1:3) = vr;
	proj_mats(k, 1:9) = reshape(U, 1, 9);


	id_tile = 1;
	for dim=1:3
		v = values(:,dim+1);
		% v = vals(:,dim); % PCA
		v = v - mean(v);
		v = Helper.filter_noise(v, 'n_points', n_filtering_points);
		v = v / max(abs(v)); % Normalization

		one_second = ceil(1/period);
		vt = v(one_second:2*one_second);
		[autocor, lags] = xcorr(vt, v);
		[pks, locs] = findpeaks(autocor, 'MinPeakDist', 0.6/period);
		p = mean(diff(locs)) * period;
		rate = 60/p;
		% rate = Helper.compute_heart_rate(v, 'period', period, 'method', 'corr');
		% p = 60/rate;
		fprintf("\tDim: Acc.%d, Computed period: %f => %.1fbpm\n", dim, p, rate);
		heart_rates(k, dim) = rate;
		subplot(n_rows, n_cols, id_tile);
		plot(lags*period, autocor)
		hold on
		pks = plot(lags(locs)*period,pks,'or');
		hold off
		% pause
		% continue

		% Temporal
		% subplot(n_rows, n_cols, id_tile);
		% plot(t, v, 'r', 'LineWidth', 1)
		% ylim([-1 1])
		% xlim([start_at start_at+duration])
		switch dim
			case 1
				ylabel({"Acc.x", sprintf("(%.1f bpm)", rate)})
				% title("Temp")
				% title([ids{k}, "PCA - Autocorr"])
				title([ids{k}, "Autocorr"])
			case 2
				ylabel({"Acc.y", sprintf("(%.1f bpm)", rate)})
			case 3
				ylabel({"Acc.z", sprintf("(%.1f bpm)", rate)})
		end

		% Frequential
		% [z, f] = Helper.to_frequential(v, 'period', period);
		% subplot(n_rows, n_cols, id_tile+1);
		% plot(f, z, 'b', 'LineWidth', 1)
		% switch dim
		% 	case 1
		% 		title("Freq")
		% end	

		id_tile = id_tile + n_cols;
	end



	% Gyr
	values = extract_values(filenames{k}, 'start_at', start_at, 'duration', duration, ...
		'period', gyr_period, 'sensor', 'GYR');
	t = values(:,1);
	n_meas(k,2) = length(t);

	% PCA
	[vals, vr, U] = Helper.pca(values(:,2:end));
	var_retained(k,4:6) = vr;
	proj_mats(k, 10:18) = reshape(U, 1, 9);

	% id_tile = 7;
	id_tile = 4; % Autocorr
	for dim=1:3
		v = values(:,dim+1);
		% v = vals(:,dim); % PCA
		v = v - mean(v);
		v = Helper.filter_noise(v, 'n_points', floor(n_filtering_points*period/gyr_period));
		v = v / max(abs(v)); % Normalization

		one_second = ceil(1/gyr_period);
		vt = v(one_second:2*one_second);
		[autocor, lags] = xcorr(vt, v);
		[pks, locs] = findpeaks(autocor, 'MinPeakDist', 0.6/gyr_period);
		p = mean(diff(locs)) * gyr_period;
		rate = 60/p;
		% rate = Helper.compute_heart_rate(v, 'period', gyr_period, 'method', 'corr');
		% p = 60/rate;
		fprintf("\tDim: Gyr.%d, Computed period: %f => %.1fbpm\n", dim, p, rate);
		heart_rates(k, 3+dim) = rate;
		subplot(n_rows, n_cols, id_tile);
		plot(lags*gyr_period, autocor)
		hold on
		pks = plot(lags(locs)*gyr_period,pks,'or');
		hold off
		% pause
		% continue	

		% Temporal
		% subplot(n_rows, n_cols, id_tile);
		% plot(t, v, 'r', 'LineWidth', 1)
		% ylim([-1 1])
		% xlim([start_at start_at+duration])
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
		% [z, f] = Helper.to_frequential(v, 'period', gyr_period);
		% subplot(n_rows, n_cols, id_tile+1);
		% plot(f, z, 'b', 'LineWidth', 1)
		% switch dim
		% 	case 3
		% 		xlabel("Freq (Hz)")
		% end	

		id_tile = id_tile + n_cols;
	end

	% sgtitle(strrep(method, '_', '\_'))
	% mtit(ids{k})
	% saveas(gcf, strcat(output_dir, 'pca-', ids{k}, '.png'));
	% mtit(sprintf('%s - PCA autocorr', ids{k}))
	saveas(gcf, strcat(output_dir, 'autocorr-', ids{k}, '.png'));
	% saveas(gcf, strcat(output_dir, 'pca-autocorr-', ids{k}, '.png'));
end

% Stats
hr_mean = mean(heart_rates, 2);
hr_std = std(heart_rates, 1, 2);


fig = figure('units','normalized','outerposition',[0 0 1 1]);
subplot(2, 2, 1)
histogram(ages, 'BinMethod', 'integers', 'FaceColor', 'b', 'FaceAlpha', 0.8)
xlabel("Ages")
ylabel("Count")
title({"Age histogram", sprintf("Nb users: %d", num_files)})
ages_mean = mean(ages);
ages_std = std(ages);
legend(sprintf("Mean: %.1f\nStd: %.1f", ages_mean, ages_std))

subplot(2, 2, 2)
histogram(genders, 'FaceColor', 'g', 'FaceAlpha', 0.8)
xlabel("Genders")
xticks([0 1])
xticklabels(["Female", "Male"])
ylabel("Count")
title({"Gender histogram", sprintf("Nb users: %d", num_files)})

subplot(2, 2, 3)
histogram(hr_mean, 'BinMethod', 'integers', 'FaceColor', 'y', 'FaceAlpha', 0.8)
xlabel("Estimated HR")
ylabel("Count")
title({"Heart rates histogram", "Avg over all 6 axes"})
hr_mean_mean = mean(hr_mean);
hr_mean_std = std(hr_mean);
legend(sprintf("Mean: %.1f\nStd: %.1f", hr_mean_mean, hr_mean_std))

subplot(2, 2, 4)
histogram(hr_std, 'BinMethod', 'integers', 'FaceColor', 'r', 'FaceAlpha', 0.8)
xlabel("HR Std")
ylabel("Count")
title({"Heart rates deviation histogram", "Computed over all 6 axes"})
hr_std_mean = mean(hr_std);
hr_std_std = std(hr_std);
legend(sprintf("Mean: %.1f\nStd: %.1f", hr_std_mean, hr_std_std))

% saveas(gcf, strcat(output_dir, 'stats-5s-hr_corr.png'));
% saveas(gcf, strcat(output_dir, 'stats-5s-pca-hr_corr.png'));


% Plot variance retained
vr = var_retained * 100;
fig = figure('units','normalized','outerposition',[0 0 1 0.6]);
% subplot(1, 2, 1)
bar(vr(:,1:3), 'stacked', 'BarWidth', 1)
xlabel("Users")
ylabel("Percentage")
title("Acc")
legend(sprintf("Dim 1. Mean: %.1f. Std: %.1f", mean(vr(:,1)), std(vr(:,1))), ... 
	sprintf("Dim 2. Mean: %.1f. Std: %.1f", mean(vr(:,2)), std(vr(:,2))), ...
	sprintf("Dim 3. Mean: %.1f. Std: %.1f", mean(vr(:,3)), std(vr(:,3))))

% subplot(1, 2, 2)
% bar(vr(:,4:6), 'stacked', 'BarWidth', 1)
% xlabel("Users")
% ylabel("Percentage")
% title("Gyr")
% legend(sprintf("Dim 1. Mean: %.1f. Std: %.1f", mean(vr(:,4)), std(vr(:,4))), ... 
% 	sprintf("Dim 2. Mean: %.1f. Std: %.1f", mean(vr(:,5)), std(vr(:,5))), ...
% 	sprintf("Dim 3. Mean: %.1f. Std: %.1f", mean(vr(:,6)), std(vr(:,6))))

mtit('PCA - 5s \newlineAmout of variance retained')

% saveas(gcf, strcat(output_dir, 'stats-5s-pca-vr.png'));


% PCA proj matrices stats
% acc_pms = squeeze(proj_mats(:, 1:9));
% gyr_pms = squeeze(proj_mats(:, 10:18));

% acc_pms_mean = mean(acc_pms);
% gyr_pms_mean = mean(gyr_pms);
% acc_pms_std = std(acc_pms);
% gyr_pms_std = std(gyr_pms);

% rot_angs = zeros(num_files, 6);
% for k=1:num_files
% 	rot_angs(k, 1:3) = Helper.rotm2euler(reshape(acc_pms(k, :), 3, 3));
% 	rot_angs(k, 4:6) = Helper.rotm2euler(reshape(gyr_pms(k, :), 3, 3));
% end
% rot_angs = unwrap(rot_angs) * 180 / pi;
% rot_angs_std = std(rot_angs);


% fig = figure('units','normalized','outerposition',[0 0 1 1]);
% subplot(2, 2, 1)
% bar(acc_pms_mean)
% hold on
% errorbar(acc_pms_mean, acc_pms_std/2, 'Color', 'k', 'LineStyle', 'none')
% hold off
% xlabel("Matrix element index (column-wise)")
% ylabel("Value")
% title("Acc projection matrix")
% legend(strcat("Stats of Std", ...
% 	sprintf("\nAvg: %.1f", mean(acc_pms_std)), ...
% 	sprintf("\nMin: %.1f", min(acc_pms_std)), ...
% 	sprintf("\nMax: %.1f", max(acc_pms_std)) ...
% 	))

% subplot(2, 2, 2)
% bar(gyr_pms_mean)
% hold on
% errorbar(gyr_pms_mean, gyr_pms_std/2, 'Color', 'k', 'LineStyle', 'none')
% hold off
% xlabel("Matrix element index (column-wise)")
% ylabel("Value")
% title("Gyr projection matrix")
% legend(strcat("Stats of Std", ...
% 	sprintf("\nAvg: %.1f", mean(gyr_pms_std)), ...
% 	sprintf("\nMin: %.1f", min(gyr_pms_std)), ...
% 	sprintf("\nMax: %.1f", max(gyr_pms_std)) ...
% 	))

% subplot(2, 2, 3)
% plot(rot_angs(:, 1:3), 'LineWidth', 2);
% xlabel("User index")
% ylabel("Unwrapped angle (°)")
% title("Acc Euler angles")
% legend(sprintf("\\theta_x. Std: %.1f", rot_angs_std(1)), ...
% 	sprintf("\\theta_y. Std: %.1f", rot_angs_std(2)), ...
% 	sprintf("\\theta_z. Std: %.1f", rot_angs_std(3)))

% subplot(2, 2, 4)
% plot(rot_angs(:, 4:6), 'LineWidth', 2);
% xlabel("User index")
% ylabel("Unwrapped angle (°)")
% title("Gyr Euler angles")
% legend(sprintf("\\theta_x. Std: %.1f", rot_angs_std(4)), ...
% 	sprintf("\\theta_y. Std: %.1f", rot_angs_std(5)), ...
% 	sprintf("\\theta_z. Std: %.1f", rot_angs_std(6)))

% mtit('PCA - 5s \newlineProjection')

% saveas(gcf, strcat(output_dir, 'stats-5s-pca-proj.png'));