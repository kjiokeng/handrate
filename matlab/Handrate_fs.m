classdef Handrate_fs
% Handrate_fs - A class which implements Handrate_fs
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
   		function [ecg_values, sensors_values, heart_rates, filenames] = read_data_from_directory(directory, varargin)
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
			addParameter(p, 'resample_freq_ecg', 250);
			addParameter(p, 'skip_delta', 0);
			addParameter(p, 'keep_filenames', {});
			parse(p, directory, varargin{:});

			directory = p.Results.directory;
			metadata_file = p.Results.metadata_file;
			sensors_dir = p.Results.sensors_dir;
			ecg_file_name = p.Results.ecg_file_name;
			sensors_period = p.Results.sensors_period;
			ecg_period = p.Results.ecg_period;
			resample_freq = p.Results.resample_freq;
			resample_freq_ecg = p.Results.resample_freq_ecg;
			skip_delta = p.Results.skip_delta;
			keep_filenames = p.Results.keep_filenames;
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

			ecg_values = {};
			sensors_values = {};
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

				% if ecg_file_id > 413
				% 	fprintf("----------------\n")
				% 	continue
				% end

				if isempty(sensors_file) || isempty(check_time) || isempty(ecg_file_id) || isempty(time_length)
					fprintf('Skipping file number %d: missing information\n', k);
					continue
				end

				if has_should_ignore
					should_ignore = table2array(T(k, should_ignore_column_name));
					if should_ignore{1}
						fprintf('Skipping file number %d: should_ignore\n', k);
						continue
					end
				end

				if length(keep_filenames)>0
					if ~any(endsWith(keep_filenames, sensors_file))
						fprintf('Skipping file number %d: not in given filenames\n', k);
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

				[ecg_vals, sensors_vals] = Handrate.align_readings(sensors_file, ecg_file, check_time, ...
					'skip_delta', skip_delta, 'ecg_time_length', time_length, 'sensors_period', sensors_period, ...
					'realign_delta', realign_delta);


				% Resample the readings
				fs = 1/sensors_period;
				[b, a] = Helper.resample_readings(sensors_vals(:,2), fs, resample_freq);
				[c, a] = Helper.resample_readings(sensors_vals(:,3), fs, resample_freq);
				[d, a] = Helper.resample_readings(sensors_vals(:,4), fs, resample_freq);
				sensors_vals = [a; b; c; d]';

				fs = 1/ecg_period;
				[b, a] = Helper.resample_readings(ecg_vals(:,2), fs, resample_freq_ecg);
				ecg_vals = [a; b]';

				% Make sure that the dimensions match
				m = min(size(ecg_vals, 1)/resample_freq_ecg, size(sensors_vals, 1)/resample_freq);
				ecg_vals = ecg_vals(1:ceil(m*resample_freq_ecg), :);
				sensors_vals = sensors_vals(1:ceil(m*resample_freq), :);

				% Concatenate the results in a single matrix in the following format
				% Column 1: time
				% Column 2: ecg vals
				% Columns 3-5: sensors vals
				% res = [ecg_vals, sensors_vals(:, 2:end)];

				% Save the result
				ecg_values{end+1} = ecg_vals;
				sensors_values{end+1} = sensors_vals;
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

		function [X, Y] = build_dataset(varargin)
			% build_dataset - Build the dataset that can be used to train the neural network
			% 	
			% Syntax:  [X, Y] = Handrate_fs.build_dataset(opt)
			%------------- BEGIN CODE --------------

			% Useful variables
			directory = '../../measures/data/';
			metadata_file = 'metadata.csv';
			% metadata_file = 'metadata_files/metadata-usr.csv';
			% metadata_file = 'metadata_files/metadata-usr2.csv';
			sensors_period = 0.0194;
			ecg_period = 1/250;
			resample_freq = 100;
			resample_freq_ecg = 250;
			n_ones = 10;
			skip_delta = 2;
			item_duration_seconds = 3;
			item_stride_seconds = max(item_duration_seconds-1, 1);
			item_duration = item_duration_seconds * resample_freq;
			item_stride = item_stride_seconds * resample_freq;
			item_duration_ecg = item_duration_seconds * resample_freq_ecg;
			item_stride_ecg = item_stride_seconds * resample_freq_ecg;
			signal_axis = 'pca';
			cwt_freq_limits = [2, 50];
			denoising_method = 'wavelet';

			% Read the data
			fprintf('Read the data...\n');
			[ecg_values, sensors_values, heart_rates, filenames] = Handrate_fs.read_data_from_directory(directory, ...
									'metadata_file', metadata_file, ...
									'sensors_period', sensors_period, ...
									'ecg_period', ecg_period, ...
									'resample_freq', resample_freq, ...
									'resample_freq_ecg', resample_freq_ecg, ...
									'skip_delta', skip_delta);

			% save('data_from_directory_fs.mat', 'ecg_values', 'sensors_values', 'heart_rates', 'filenames')
			% load('data_from_directory_fs.mat')

			% Actual building of the dataset
			fprintf('Actual building of the dataset...\n');
			X = [];
			Y = [];
			heartbeats_matrix = {};
			scalograms_matrix = {};
			n_items = 0;
			n_files = length(ecg_values);
			files_ids = 1:n_files;
			for idx=1:length(files_ids)
				k = files_ids(idx);
				fprintf('--- Processing file %d / %d: %.1f%%: %s\n', k, n_files, k*100/n_files, filenames{k});
				heart_rate = heart_rates{k};
				ecg_vals = ecg_values{k};
				ecg_vals = ecg_vals(:, 2);
				sensors_vals = sensors_values{k};
				sensors_vals = sensors_vals(:, 2:end);

				% Convert the ECG values to heartbeats vector (1 only where there is a peak)
				heartbeats = Handrate.ecg_to_heartbeats(ecg_vals, ...
								'heart_rate', heart_rate, ...
								'n_ones', n_ones, ...
								'freq', resample_freq_ecg);

				% Convert the sensors values to time-frequency representation
				[scalogram, x] = Handrate.sensors_to_scalogram(sensors_vals, ...
								'freq', resample_freq, ...
								'signal_axis', signal_axis, ...
								'cwt_freq_limits', cwt_freq_limits, ...
								'denoising_method', denoising_method);

				heartbeats_matrix{k} = heartbeats;
				scalograms_matrix{k} = scalogram;
			end

			
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
			% save('perm_idx_fs.mat', 'perm', 'train_idx', 'dev_idx', 'test_idx')
			% load('perm_idx_fs.mat')

			fprintf('Bulding training set\n')
			X_train = [];
			Y_train = [];
			n_items = 0;
			for idx=1:length(train_idx)
				k = train_idx(idx);
				heartbeats = heartbeats_matrix{k};
				scalogram = scalograms_matrix{k};

				% Slice the (30s long) data into different items
				n_time_steps = min(length(heartbeats), size(scalogram, 2));
				begin_time_ecg = 1;
				for begin_time=1:item_stride:n_time_steps-item_duration+1
					n_items = n_items + 1;

					end_time = begin_time + item_duration - 1;
					item_scalogram = scalogram(:,begin_time:end_time);
					item_scalogram = item_scalogram / max(max(item_scalogram));
					X_train(n_items, :, :) = item_scalogram';

					end_time_ecg = begin_time_ecg + item_duration_ecg - 1;
					item_heartbeat = heartbeats(begin_time_ecg:end_time_ecg);
					Y_train(n_items, :) = item_heartbeat';
					begin_time_ecg = begin_time_ecg + item_stride_ecg;
				end

				rperm = randperm(n_items);
				X_train = X_train(rperm, :, :);
				Y_train = Y_train(rperm, :);
			end

			fprintf('Bulding dev set\n')
			X_dev = [];
			Y_dev = [];
			n_items = 0;
			for idx=1:length(dev_idx)
				k = dev_idx(idx);
				heartbeats = heartbeats_matrix{k};
				scalogram = scalograms_matrix{k};

				% Slice the (30s long) data into different items
				n_time_steps = min(length(heartbeats), size(scalogram, 2));
				begin_time_ecg = 1;
				for begin_time=1:item_stride:n_time_steps-item_duration+1
					n_items = n_items + 1;

					end_time = begin_time + item_duration - 1;
					item_scalogram = scalogram(:,begin_time:end_time);
					item_scalogram = item_scalogram / max(max(item_scalogram));
					X_dev(n_items, :, :) = item_scalogram';

					end_time_ecg = begin_time_ecg + item_duration_ecg - 1;
					item_heartbeat = heartbeats(begin_time_ecg:end_time_ecg);
					Y_dev(n_items, :) = item_heartbeat';
					begin_time_ecg = begin_time_ecg + item_stride_ecg;
				end

				rperm = randperm(n_items);
				X_dev = X_dev(rperm, :, :);
				Y_dev = Y_dev(rperm, :);
			end

			fprintf('Bulding test set\n')
			X_test = [];
			Y_test = [];
			n_items = 0;
			for idx=1:length(test_idx)
				k = test_idx(idx);
				heartbeats = heartbeats_matrix{k};
				scalogram = scalograms_matrix{k};

				% Slice the (30s long) data into different items
				n_time_steps = min(length(heartbeats), size(scalogram, 2));
				begin_time_ecg = 1;
				for begin_time=1:item_stride:n_time_steps-item_duration+1
					n_items = n_items + 1;

					end_time = begin_time + item_duration - 1;
					item_scalogram = scalogram(:,begin_time:end_time);
					item_scalogram = item_scalogram / max(max(item_scalogram));
					X_test(n_items, :, :) = item_scalogram';

					end_time_ecg = begin_time_ecg + item_duration_ecg - 1;
					item_heartbeat = heartbeats(begin_time_ecg:end_time_ecg);
					Y_test(n_items, :) = item_heartbeat';
					begin_time_ecg = begin_time_ecg + item_stride_ecg;
				end

				rperm = randperm(n_items);
				X_test = X_test(rperm, :, :);
				Y_test = Y_test(rperm, :);
			end

			% Save to file
			dataset_filename = sprintf('datasets/handrate-fs-dataset-%.1fs-%dones-%s.mat', ...
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
			
			dataset_filename = sprintf('datasets/handrate-fs-dataset-raw-%.1fs-%dones-%s.mat', ...
				item_duration_seconds, n_ones, signal_axis);
			save(dataset_filename, 'heartbeats_matrix_train', 'scalograms_matrix_train', 'filenames_train', ...
									'heartbeats_matrix_dev', 'scalograms_matrix_dev', 'filenames_dev', ...
									'heartbeats_matrix_test', 'scalograms_matrix_test', 'filenames_test')
			fprintf('Saved data to file %s\n', dataset_filename);
		end

		function [X, Y] = main(varargin)
			% main - Main function
			% 	
			% Syntax:  [] = Handrate.main(opt)
			%------------- BEGIN CODE --------------

			[X, Y] = Handrate_fs.build_dataset();			
		end
   end
end