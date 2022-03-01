
% directory = '../../measures/bcg-tests/bed';
% directory = '../../measures/bcg-tests/noise';
% directory = '../../measures/bcg-tests/feet';
% directory = '../../measures/bcg-tests/22-04-20';
% directory = '../../measures/bcg-tests/navel';
directory = '../../measures/bcg-tests/hand-tmp';
% directory = '../../measures/data/sensors/';
% output_dir = './results/bcg-tests/feet/';

start_at = 2;
duration = 30;
period = 0.0194; % Pixel 2
% period = 0.005; % Nexus 5
% period = 0.01; % S8
% gyr_period = 0.002;

heartbeat_files = dir(directory);
heartbeat_files = heartbeat_files(3:end);
num_files = length(heartbeat_files);

filenames = {};
for k=1:num_files
	filenames{k} = strcat(heartbeat_files(k).folder, '/', heartbeat_files(k).name);
end

% Plot the values
% fig = figure('units','normalized','outerposition',[0 0 1 1]);
fig = figure('units','normalized','outerposition',[0 1 1 0.5]); % Autocorr
n_cols = 1;
n_rows = ceil(num_files/n_cols);
id_tile = 1;
Q_kurts = zeros(num_files, 3);
fft_size = 2048;
for k=1:num_files

	% Acc
	values = extract_values(filenames{k}, 'start_at', start_at, 'duration', duration, ...
		'period', period, 'sensor', 'ACC');
	t = values(:,1)-start_at;
	s = values(:,2:end);
	s = s - mean(s);
	s = s ./ max(abs(s));

	% subplot(3, 1, 1)
	% plot(s(:,1))

	% subplot(3, 1, 2)
	% plot(s(:,2))

	% subplot(3, 1, 3)
	% plot(s(:,3))

	% t = s(:, 2)
	% fprintf("Current file: %s\n", heartbeat_files(k).name)
	% pause

	for a=1:3
		x = s(:, a);
		q_kurt = Helper.compute_q_kurt(x, 'period', period, 'fft_size', fft_size);
		Q_kurts(k, a) = q_kurt;
	end

	% pause
	continue
	% s = Helper.detrend(s);
	% s = s - mean(s);
	% s = sort(abs(s))
	% close all
	% return
	% s = s - mean(s);
	% s = s ./ max(abs(s));
	positions = [0.2 0 -0.2];
	s_save = s;
	s = s + positions;

	maxlag = 200;
	v = s_save(:,3);
	f = ceil(1/period)
	v_template = v(1:f);
	% corrs = xcorr(s(:,3), maxlag, 'coeff');
	[corrs, lags] = xcorr(v, v_template);
	corrs = corrs / max(corrs);
	lags = lags * period;
	% corrs = Helper.detrend(corrs, 'polynomial_order', 1);
	plot(lags, corrs)
	ylim([0 1])
	pause
	continue
	subplot(n_rows, n_cols, id_tile);
	plot(t, s);
	legend("x", "y", "z")
	xlim([min(t) max(t)])
	if k>num_files - n_rows
		xlabel("Time (s)")
	end
	% ylabel("Normalized amplitude")
	title(strcat("Expe ", num2str(k)))
	id_tile = id_tile + 1;

	pause
end

Q_kurts = [Q_kurts, max(Q_kurts, [], 2)]