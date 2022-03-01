% compare_patterns_from_dir - Compare heartbeat patterns contained in directory
%
% MAT-files required: compare_patterns.m, extract_values.m
%
% See also: compare_patterns, extract_values
% Author: Kevin Jiokeng
% email: kevin.jiokeng@enseeiht.fr
% November 2019; Last revision: 07-November-2019
%------------- BEGIN CODE --------------

plot_values = false;
autocorr_comp = true;

% directory = '../../measures/chest/comp-08-11-19';
directory = '../../measures/chest/comp-12-11-19-unique';
directory = '../../measures/hand/pixel/';
directory = '../../measures/hand/nexus/gyro/';
directory = '../../measures/hand/s8/all/';
directory = '../../measures/hand/s8/analysis2/';
directory = '../../measures/hand/pixel/analysis2/';
% directory = '../../measures/tmp';
start_at = 10;
duration = 5;
period = 0.0194; % Pixel 2
gyr_period = 0.0025; % Pixel 2
% period = 0.005; % Nexus 5
% period = 0.01; % S8
% gyr_period = 0.002; % S8

heartbeat_files = dir(directory);
heartbeat_files = heartbeat_files(3:end);
num_files = length(heartbeat_files);

filenames = {};
for k=1:num_files
	filenames{k} = strcat(heartbeat_files(k).folder, '/', heartbeat_files(k).name);
end

% Reading the files content
values = {};
for k=1:num_files
	values{k} = extract_values(filenames{k}, 'start_at', start_at, 'duration', duration, ...
		'period', period);		
	
	% Extract the user id from the filename
	tmp = strsplit(filenames{k}, '/');
	tmp = tmp{end};
	tmp = strsplit(tmp, '-');
	ids{k} = tmp{1};
	% ids{k} = strcat(tmp{1}, '-', tmp{5}); % Take the id and the position
end

% Plot the values
if plot_values
	figure
	n_cols = 4;
	n_rows = ceil(num_files / (n_cols/2));
	id_tile = 2;
	for k=1:num_files
		t = values{k}(:,1);
		v = values{k}(:,2);
		% v = Helper.filter_noise(v);
		v = v / max(abs(v)); % Normalization
			
		% Frequential
		[z, f] = Helper.to_frequential(v, 'period', period);
		subplot(n_rows, n_cols, id_tile);
		plot(f, z, 'b', 'LineWidth', 1)
		ylim([0 3])
		if k==1 || k==2
			title("Frequency domain")
		elseif k==num_files || k==num_files-1
			xlabel("Frequency (Hz)")
		end
		

		% Temporal
		% bpm = Ubicomp18.compute_heart_rate(values{k});
		subplot(n_rows, n_cols, id_tile-1);
		plot(t, v, 'r', 'LineWidth', 1)
		xlabel("Time (s)")
		ylabel(ids{k})
		% ylabel({ids{k}, sprintf("(%.1f bpm)", bpm)})
		ylim([-1 1])
		xlim([start_at start_at+duration])
		if k==1 || k==2
			title("Time domain")
		elseif k==num_files
			xlabel("Time (s)")
		end

		% For the next user
		id_tile = id_tile + 2;
	end
end


% Augment the data: take multiple fragments of a pattern
if autocorr_comp
	avg_bpm = 73;
	duration = 60/avg_bpm;
	% start_at = 0:duration:30-duration;
	start_at = 10:25;
	duration = 5;
	values = {};
	ids = {};
	for k=1:num_files
	% for k=50:60
		% Extract the user id from the filename
		tmp = strsplit(filenames{k}, '/');
		tmp = tmp{end};
		tmp = strsplit(tmp, '-');
		user_id = tmp{1};

		for l=1:length(start_at)
			v = extract_values(filenames{k}, 'start_at', start_at(l), 'duration', duration, ...
				'period', period, 'sensor', 'ACC');
				% 'period', gyr_period, 'sensor', 'GYR');
			% v_y = v(:,4);
			% v_y = v_y - mean(v_y);
			% v = v_y / max(abs(v_y));
			% v_y = Helper.filter_noise(v_y, 'sampling_freq', 1/period);

			[vals, vr, U] = Helper.pca(v(:,2:end));
			v = vals(:, 1);

			% v = vals;
			% tmp = v(:, 1);
			% t = Helper.to_frequential(tmp);
			% tmp = v(:, 2);
			% t = [t; Helper.to_frequential(tmp)];
			% tmp = v(:, 3);
			% t = [t; Helper.to_frequential(tmp)];
			% v = t;

			values{end+1} = v;
		end
		ids{end+1} = user_id;
	end


	% Comparison methods
	% comp_methods = {'fft_euclidian', 'dtw', 'fft_cos', 'cos', 'fft_dtw', 'euclidian'};
	% comp_methods = {'fft_euclidian', 'dtw', 'corrcoef', 'fft_corrcoef', 'dwt_corrcoef'};
	% comp_methods = {'fft_euclidian', 'dtw', 'dwt_corrcoef', 'fft_corrcoef'};
	% comp_methods = {'fft_euclidian', 'dtw', 'fft_corrcoef', 'corrcoef'};
	% comp_methods = {'dwt_euclidian', 'dwt_dtw', 'dwt_corrcoef'};
	% comp_methods = {'fft_euclidian', 'fft_corrcoef', 'stft_euclidian', 'stft_corrcoef'};
	% comp_methods = {'fft_euclidian', 'fft_corrcoef', 'stft_euclidian', 'stft_corrcoef', 'dtw', 'dwt_corrcoef'};
	% comp_methods = {'fft_corrcoef', 'dtw'};
	comp_methods = {'fft_corrcoef', 'dtw', 'cwt_euclidian', 'cwt_corrcoef'};
	% comp_methods = {'fft_euclidian'};
	% comp_methods = {'stft_euclidian'};
	% comp_methods = {'corrcoef', 'fft_euclidian', 'fft_corrcoef', 'fft_cos'};
	% comp_methods = {'cceps_euclidian', 'cceps_corrcoef', 'stft_euclidian', 'stft_corrcoef'};
	% comp_methods = {'fft_corrcoef'};
	% comp_methods = {'dtw'};
	% comp_methods = {'dwt_euclidian', 'dwt_corrcoef', 'dwt_cos'};
	% comp_methods = {'cceps_euclidian', 'cceps_corrcoef'};
	n_methods = length(comp_methods);

	% Actual comparison
	figure
	n_cols = max(1, 2*(n_methods>1));
	n_rows = ceil(n_methods / n_cols);
	for m=1:n_methods
		fprintf("--- Method %d/%d (%.1f%%): %s\n", ...
			m, n_methods, (m-1)*100/n_methods, comp_methods{m})
		subplot(n_rows, n_cols, m);
		comp = compare_patterns(values, ids, 'method', comp_methods{m});
	end
end