classdef Handrate_scripts
% Handrate_scripts - A class which implements Handrate_scripts
% 
% Author: Kevin Jiokeng
% email: kevin.jiokeng@enseeiht.fr
% February 2020; Last revision: 27-February-2020
%------------- BEGIN CODE --------------

   % Constants
   properties (Constant)
      
   end

   properties
      
   end

   methods  	
      	function save(obj)
      		% Save the current object to a file

	      	tmp = strsplit(obj.dir_path, '/');
	      	target_filename = tmp{end};
	      	save(target_filename, 'obj')
      	end
   end

   methods (Static)
   		function [ecg_and_sensors_values, heart_rates, filenames] = read_data_from_directory(directory, varargin)
   			% read_data_from_directory - Process files in batch
			% 	
			% Syntax:  [ecg_and_sensors_values, heart_rates] = Helper.read_data_from_directory(directory, opt)
			%
			% Inputs:
			%		directory - The path of the directory to process files of
			%  		opt - Optional name-value parameters	 		
			% 	 		metadata_file - The name of the file containing metadata for this processing
			% 	 		sensors_dir - The name of the directory containing sensors readings files
			% 	 		ecg_file_name - The name of the ECG readings files
			% 	 		sensors_period - The period of the sensors values 
			% 	 		ecg_period - The period of the ecg values
			% 	 		resample_freq - The frequency to which all the readings should be resampled
			%      
			% Outputs:
			%  	 	ecg_and_sensors_values - A cell array containing the ecg and sensors values
			%  	 		for each (ecg, sensors) correspondance
			%  	 	heart_rates - The corresponding heart rates
			%------------- BEGIN CODE --------------

			p = inputParser;
			addRequired(p, 'directory');
			addParameter(p, 'metadata_file', 'metadata.csv');
			addParameter(p, 'sensors_dir', 'sensors');
			addParameter(p, 'ecg_file_name', 'ecg.dat');
			addParameter(p, 'sensors_period', 0.0194);
			addParameter(p, 'ecg_period', 1/250);
			addParameter(p, 'resample_freq', 100);
			addParameter(p, 'skip_delta', 0);
			parse(p, directory, varargin{:});

			directory = p.Results.directory;
			metadata_file = p.Results.metadata_file;
			sensors_dir = p.Results.sensors_dir;
			ecg_file_name = p.Results.ecg_file_name;
			sensors_period = p.Results.sensors_period;
			ecg_period = p.Results.ecg_period;
			resample_freq = p.Results.resample_freq;
			skip_delta = p.Results.skip_delta;
			metadata_file = strcat(directory, '/', metadata_file);
			sensors_dir = strcat(directory, '/', sensors_dir);

			% Useful variables
			ecg_file_id_column_name = 'ecg_file_id';
			time_length_column_name = 'time_length';
			check_time_column_name = 'check_time';
			heart_rate_column_name = 'heart_rate';
			sensors_file_column_name = 'sensors_file';
			realign_delta_column_name = 'realign_delta';
			should_ignore_column_name = 'should_ignore';

			% Actual computation
			T = readtable(metadata_file, 'DatetimeType', 'text');
			[n_rows, n_cols] = size(T);
			has_realign_delta = any(strcmp(T.Properties.VariableNames, realign_delta_column_name));
			has_should_ignore = any(strcmp(T.Properties.VariableNames, should_ignore_column_name));

			ecg_and_sensors_values = {};
			heart_rates = {};
			filenames = {};
			for k=1:n_rows
				ecg_file_id = table2array(T(k, ecg_file_id_column_name));
				time_length = table2array(T(k, time_length_column_name));
				heart_rate = table2array(T(k, heart_rate_column_name));
				check_time = table2array(T(k, check_time_column_name));
				check_time = check_time{1};
				sensors_file = table2cell(T(k, sensors_file_column_name));
				sensors_file = sensors_file{1};

				% if ecg_file_id <= 413
				% if ecg_file_id <= 503
				if ecg_file_id <= 0
				% if ecg_file_id ~= 53
					fprintf('Skipping file number %d\n', k);
					continue
				end

				if isempty(sensors_file) || isempty(check_time) || isempty(ecg_file_id) || isempty(time_length)
					fprintf('Skipping file number %d\n', k);
					continue
				end

				if has_should_ignore
					should_ignore = table2array(T(k, should_ignore_column_name));
					if should_ignore{1}
						fprintf('Skipping file number %d\n', k);
						continue
					end
				end
				
				realign_delta = 0;
				if has_realign_delta
					realign_delta = table2array(T(k, realign_delta_column_name));
					if isnan(realign_delta)
						realign_delta = 0;
					end
				end

				tmp = strsplit(check_time, ' ');
				ecg_date = tmp{1};
				ecg_file = strcat(directory, '/', ecg_date, '/', num2str(ecg_file_id), '/', ecg_file_name);
				sensors_file = strcat(sensors_dir, '/', sensors_file);

				[ecg_values, sensors_values] = Handrate.align_readings(sensors_file, ecg_file, check_time, ...
					'skip_delta', skip_delta, 'ecg_time_length', time_length, 'sensors_period', sensors_period, ...
					'realign_delta', realign_delta);


				% Resample the readings
				fs = 1/sensors_period;
				[b, a] = Helper.resample_readings(sensors_values(:,2), fs, resample_freq);
				[c, a] = Helper.resample_readings(sensors_values(:,3), fs, resample_freq);
				[d, a] = Helper.resample_readings(sensors_values(:,4), fs, resample_freq);
				sensors_values = [a; b; c; d]';

				fs = 1/ecg_period;
				[b, a] = Helper.resample_readings(ecg_values(:,2), fs, resample_freq);
				ecg_values = [a; b]';

				% Make sure that the dimensions match
				m = min(size(ecg_values, 1), size(sensors_values, 1));
				ecg_values = ecg_values(1:m, :);
				sensors_values = sensors_values(1:m, :);

				% Concatenate the results in a single matrix in the following format
				% Column 1: time
				% Column 2: ecg values
				% Columns 3-5: sensors values
				res = [ecg_values, sensors_values(:, 2:end)];

				% Save the result
				ecg_and_sensors_values{end+1} = res;
				heart_rates{end+1} = heart_rate;
				filenames{end+1} = sensors_file;	


				% ------------------- Tmp ---------------------
				% sensors_values(:,2) = Helper.filter_noise(sensors_values(:,2), ...
				% 	'method', 'wavelet');
				% sensors_values(:,3) = Helper.filter_noise(sensors_values(:,3), ...
				% 	'method', 'wavelet');
				% sensors_values(:,4) = Helper.filter_noise(sensors_values(:,4), ...
				% 	'method', 'wavelet');

				% ecg_values = abs(ecg_values);
				% sensors_values = abs(sensors_values);
				% subplot(2, 2, 1)
				% plot(ecg_values(:, 1), ecg_values(:,2))
				% hold on
				% plot(sensors_values(:, 1), sensors_values(:,2)+0.3)
				% plot(sensors_values(:, 1), sensors_values(:,3))
				% plot(sensors_values(:, 1), sensors_values(:,4))
				% legend("ECG", "Acc.X", "Acc.Y", "Acc.Z")
				% hold off
				% pause
			end
   		end

   		function [scalogram, x] = sensors_to_scalogram(sensors_values, ecg_values, varargin)
   			% sensors_to_scalogram - Convert a sensors vector to a scalogram matrix
			% 	
			% Syntax:  [scalogram] = sensors_to_scalogram(sensors_values, opt)
			%
			% Inputs:
			%		sensors_values - The sensors values to be processed
			%  		opt - Optional name-value parameters
			%  			freq - The sampling frequency of the signal
			%      
			% Outputs:
			%  	 	scalogram - The output scalogram
			%------------- BEGIN CODE --------------

			p = inputParser;
			addRequired(p, 'sensors_values');
			addRequired(p, 'ecg_values');
			addParameter(p, 'freq', 100);
			addParameter(p, 'cwt_freq_limits', [2 50]);
			addParameter(p, 'cwt_voices_per_octave', 16);
			addParameter(p, 'cwt_time_bandwidth', 10);
			addParameter(p, 'signal_axis', 'pca');
			addParameter(p, 'denoising_method', 'wavelet');
			parse(p, sensors_values, ecg_values, varargin{:});

			sensors_values = p.Results.sensors_values;
			ecg_values = p.Results.ecg_values;
			freq = p.Results.freq;
			cwt_freq_limits = p.Results.cwt_freq_limits;
			cwt_voices_per_octave = p.Results.cwt_voices_per_octave;
			cwt_time_bandwidth = p.Results.cwt_time_bandwidth;
			signal_axis = p.Results.signal_axis;
			denoising_method = p.Results.denoising_method;

			% Denoise the signal
			% sensors_values = sensors_values ./ max(abs(sensors_values));
			sensors_values(:,1) = Helper.filter_noise(sensors_values(:,1), ...
				'method', denoising_method);
			sensors_values(:,2) = Helper.filter_noise(sensors_values(:,2), ...
				'method', denoising_method);
			sensors_values(:,3) = Helper.filter_noise(sensors_values(:,3), ...
				'method', denoising_method);

			% Normalize the input
			x = Helper.detrend(sensors_values(:,1));
			y = Helper.detrend(sensors_values(:,2));
			z = Helper.detrend(sensors_values(:,3));
			sensors_values = [x, y, z];
			%sensors_values = sensors_values - mean(sensors_values);
			% sensors_values = sensors_values ./ max(abs(sensors_values));

			% Signal combination
			sensors_values = sensors_values ./ max(abs(sensors_values));
			old_sensors_values = sensors_values;
			switch signal_axis
				case 'pca'
					[sensors_values, var_ret, U, S] = Helper.pca(sensors_values);
				case 'x'
					sensors_values = x;
				case 'y'
					sensors_values = y;
				case 'z'
					sensors_values = z;
			end

			global old_sensors_values_save;
			global sensors_values_save;
			global ecg_values_save;
			old_sensors_values_save = old_sensors_values;
			sensors_values_save = sensors_values;
			ecg_values_save = ecg_values;

			close all
			fig = figure('units','normalized','outerposition',[0 0 1 1]);
			n_rows = 5;
			n_cols = 1;
			id_tile = 1;
			t = (1:2000)/freq;
			positions = [2 0 -2];
			old_sensors_values = old_sensors_values(1:length(t), :);
			sensors_values = sensors_values(1:length(t), :);
			ecg = ecg_values(1:length(t));

			% subplot(n_rows, n_cols, id_tile)
			% plot(t, old_sensors_values+positions)
			% % hold on
			% % plot(t, -0.35 + abs(ecg) * 0.7 / max(ecg))
			% legend("X", "Y", "Z", "ECG")
			% xlabel("Time (s)")
			% ylabel("Amplitude")
			% title("Before preprocessing")
			% id_tile = id_tile + 1;

			% subplot(n_rows, n_cols, id_tile)
			% plot(t, sensors_values+positions)
			% % hold on
			% % plot(t, ecg * 10 / max(ecg))
			% legend("X", "Y", "Z", "ECG")
			% xlabel("Time (s)")
			% ylabel("Normalized amplitude")
			% title("After preprocessing")
			% id_tile = id_tile + 1;

			%% --------------------------
			subplot(n_rows, n_cols, id_tile)
			x = sensors_values(1:length(t), 1);
			% aaz = kurtosis(x)
			% x = old_sensors_values(1:length(t), 2);
			plot(t, x)
			id_tile = id_tile + 1;

			subplot(n_rows, n_cols, id_tile)
			[wt, f] = cwt(x, freq, ...
							'VoicesPerOctave', cwt_voices_per_octave, ...
						    'TimeBandWidth', cwt_time_bandwidth*4, ...
						    'FrequencyLimits', cwt_freq_limits);
			surface(t, f, abs(wt))
			axis tight
			shading flat
			xlabel('Time (s)')
			ylabel('Frequency (Hz)')
			set(gca, 'yscale', 'log')
			yticks([0 0.1 0.3 1 4 16 50])
			% global special_colormap;
			% colormap(special_colormap)
			drawnow
			hold on
			yyaxis right
			ecg_values = ecg_values(1:length(t));
			plot(t, ecg_values, 'r', 'LineWidth', 2)
			id_tile = id_tile + 1;

			subplot(n_rows, n_cols, id_tile)
			n_rep = 10;
			m = max(abs(wt));
			mm = repmat(m, n_rep, 1);
			surface(t, 1:n_rep, mm)
			axis tight
			shading flat
			drawnow
			id_tile = id_tile + 1;

			subplot(n_rows, n_cols, id_tile)
			ecg_values = ecg_values(1:length(t));
			plot(t, ecg_values)
			id_tile = id_tile + 1;

			subplot(n_rows, n_cols, id_tile)
			plot(t, m)
			id_tile = id_tile + 1;
			pause

			% Select one axis
			x = sensors_values(:,2);
			x = x - mean(x);
			x = abs(x);

			% Compute the Continuous Wavelet Transform
			[wt, f] = cwt(x, freq, ...
							'VoicesPerOctave', cwt_voices_per_octave, ...
						    'TimeBandWidth', cwt_time_bandwidth, ...
						    'FrequencyLimits', cwt_freq_limits);
			scalogram = abs(wt);
			scalogram = scalogram / max(max(scalogram));

			% Visualization
			% image(scalogram, 'CDataMapping','scaled')
   		end

		function [X, Y] = build_dataset(varargin)
			% build_dataset - Build the dataset that can be used to train the neural network
			% 	
			% Syntax:  [X, Y] = Handrate_scripts.build_dataset(opt)
			%------------- BEGIN CODE --------------

			% Useful variables
			directory = '../../measures/data/';
			metadata_file = 'metadata.csv';
			% metadata_file = 'metadata_files/metadata-usr.csv';
			% metadata_file = 'metadata_files/metadata-usr2.csv';
			% metadata_file = 'metadata_files/experimental-conditions/metadata.csv';
			% metadata_file = 'metadata_files/experimental-conditions/metadata-relaxed.csv';
			sensors_period = 0.0194;
			ecg_period = 1/250;
			resample_freq = 100;
			n_ones = 3;
			skip_delta = 2;
			item_duration_seconds = 3;
			item_stride_seconds = max(item_duration_seconds-1, 1);
			item_duration = item_duration_seconds * resample_freq;
			item_stride = item_stride_seconds * resample_freq;
			signal_axis = 'pca';

			% Read the data
			fprintf('Read the data...\n');
			[ecg_and_sensors_values, heart_rates, filenames] = Handrate_scripts.read_data_from_directory(directory, ...
									'metadata_file', metadata_file, ...
									'sensors_period', sensors_period, ...
									'ecg_period', ecg_period, ...
									'resample_freq', resample_freq, ...
									'skip_delta', skip_delta);


			% Actual building of the dataset
			fprintf('Actual building of the dataset...\n');
			X = [];
			Y = [];
			heartbeats_matrix = [];
			scalograms_matrix = [];
			n_items = 0;
			n_files = length(ecg_and_sensors_values);
			files_ids = 1:n_files;
			% files_ids = [49 52 53 102 144 191 193];
			% files_ids = [24 27 49 58 84 104 105 109 114 135 149];
			% files_ids = [36 52 90 105 106 129 132 183 185];
			max_peaks = [];
			ibi_stds = [];
			% fig = figure('units','normalized','outerposition',[0 1 1 0.5]);
			for idx=1:length(files_ids)
				k = files_ids(idx);
				fprintf('--- Processing file %d / %d: %.1f%%: %s\n', k, n_files, k*100/n_files, filenames{k});
				heart_rate = heart_rates{k};
				data = ecg_and_sensors_values{k};
				time = data(:,1);
				ecg_values = data(:,2);
				sensors_values = data(:,3:5);

				% Convert the ECG values to heartbeats vector (1 only where there is a peak)
				heartbeats = Handrate.ecg_to_heartbeats(ecg_values, ...
								'heart_rate', heart_rate, ...
								'n_ones', n_ones, ...
								'freq', resample_freq);

				% Convert the sensors values to time-frequency representation
				[scalogram, x] = Handrate_scripts.sensors_to_scalogram(sensors_values, ...
								ecg_values, ...
								'freq', resample_freq, ...
								'signal_axis', signal_axis);

				heartbeats_matrix{k} = heartbeats;
				scalograms_matrix{k} = scalogram;

				continue


				% sss = sensors_values - mean(sensors_values);
				% sss(:, 1) = Helper.detrend(sss(:, 1));
				% sss(:, 2) = Helper.detrend(sss(:, 2));
				% sss(:, 3) = Helper.detrend(sss(:, 3));
				% val_max = max(max(abs(sss)));
				% val_prctile = prctile(abs(sss(:)), 95);
				% if val_max > 1 || val_prctile > 0.35
				% 	continue
				% end
				% max_peaks(end+1, :) = [val_max, val_prctile];


				[pks, idx] = findpeaks(heartbeats);
				% plot(ecg_values)
				% hold on
				% plot(heartbeats)
				% plot(idx, pks, 'ro')
				% ylim([0 2])
				% hold off
				% pause

				ibis = 1000 * diff(idx) / resample_freq;
				ibi_std = std(ibis);
				if ibi_std > 165
					continue
				end
				ibi_stds(end+1) = ibi_std;
			end
			% X = max_peaks(:, 1);
			% Y = max_peaks(:, 2);

			X = ibi_stds;


			% Stop execution here
			return
			% [m, p] = Handrate_scripts.main(); close all; subplot(1, 2, 1); h1=histogram(m, 30); subplot(1, 2, 2); h2=histogram(p, 30)
			
			
			%% Split into train/dev/test sets
			fprintf('Split in to train/dev/test sets...\n');
			m = length(files_ids);
			m_train = floor(m * 0.6);
			m_dev = floor(m * 0.2);
			m_test = m - m_train - m_dev;
			
			perm = randperm(m);
			train_idx = perm(1:m_train);
			dev_idx = perm(m_train+1:m_train+m_dev);
			test_idx = perm(m_train+m_dev+1:end);

			X_train = [];
			Y_train = [];
			n_items = 0;
			for idx=1:length(train_idx)
				k = train_idx(idx);
				heartbeats = heartbeats_matrix{k};
				scalogram = scalograms_matrix{k};

				% Slice the (30s long) data into different items
				n_time_steps = min(length(heartbeats), size(scalogram, 2));
				for begin_time=1:item_stride:n_time_steps-item_duration+1
					n_items = n_items + 1;

					end_time = begin_time + item_duration - 1;
					item_scalogram = scalogram(:,begin_time:end_time);
					item_heartbeat = heartbeats(begin_time:end_time);

					X_train(n_items, :, :) = item_scalogram';
					Y_train(n_items, :) = item_heartbeat';
				end
			end

			X_dev = [];
			Y_dev = [];
			n_items = 0;
			for idx=1:length(dev_idx)
				k = dev_idx(idx);
				heartbeats = heartbeats_matrix{k};
				scalogram = scalograms_matrix{k};

				% Slice the (30s long) data into different items
				n_time_steps = min(length(heartbeats), size(scalogram, 2));
				for begin_time=1:item_stride:n_time_steps-item_duration+1
					n_items = n_items + 1;

					end_time = begin_time + item_duration - 1;
					item_scalogram = scalogram(:,begin_time:end_time);
					item_heartbeat = heartbeats(begin_time:end_time);

					X_dev(n_items, :, :) = item_scalogram';
					Y_dev(n_items, :) = item_heartbeat';
				end
			end

			X_test = [];
			Y_test = [];
			n_items = 0;
			for idx=1:length(test_idx)
				k = test_idx(idx);
				heartbeats = heartbeats_matrix{k};
				scalogram = scalograms_matrix{k};

				% Slice the (30s long) data into different items
				n_time_steps = min(length(heartbeats), size(scalogram, 2));
				for begin_time=1:item_stride:n_time_steps-item_duration+1
					n_items = n_items + 1;

					end_time = begin_time + item_duration - 1;
					item_scalogram = scalogram(:,begin_time:end_time);
					item_heartbeat = heartbeats(begin_time:end_time);

					X_test(n_items, :, :) = item_scalogram';
					Y_test(n_items, :) = item_heartbeat';
				end
			end

			% Save to file
			dataset_filename = sprintf('datasets/handrate-dataset-%ds-%dones-%s.mat', ...
				item_duration_seconds, n_ones, signal_axis);
			save(dataset_filename, 'X_train', 'Y_train', 'X_dev', 'Y_dev', 'X_test', 'Y_test')
			fprintf('Saved data to file %s\n', dataset_filename);

			% Save raw data
			heartbeats_matrix_train = heartbeats_matrix(train_idx);
			scalograms_matrix_train = scalograms_matrix(train_idx);
			filenames_train = filenames(train_idx);
			heartbeats_matrix_dev = heartbeats_matrix(dev_idx);
			scalograms_matrix_dev = scalograms_matrix(dev_idx);
			filenames_dev = filenames(dev_idx);
			heartbeats_matrix_test = heartbeats_matrix(test_idx);
			scalograms_matrix_test = scalograms_matrix(test_idx);
			filenames_test = filenames(test_idx);
			
			dataset_filename = sprintf('datasets/handrate-dataset-raw-%ds-%dones-%s.mat', ...
				item_duration_seconds, n_ones, signal_axis);
			save(dataset_filename, 'heartbeats_matrix_train', 'scalograms_matrix_train', 'filenames_train', ...
									'heartbeats_matrix_dev', 'scalograms_matrix_dev', 'filenames_dev', ...
									'heartbeats_matrix_test', 'scalograms_matrix_test', 'filenames_test')
			fprintf('Saved data to file %s\n', dataset_filename);
		end

		function [X, Y] = main(varargin)
			% main - Main function
			% 	
			% Syntax:  [] = Handrate_scripts.main(opt)
			%------------- BEGIN CODE --------------

			[X, Y] = Handrate_scripts.build_dataset();			
		end
   end
end