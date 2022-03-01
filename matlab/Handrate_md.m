classdef Handrate_md
% Handrate_md - A class which implements Handrate_md
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
			% 	 		acc_period - The period of the sensors values 
			% 	 		ecg_period - The period of the ecg values
			% 	 		resample_freq - The frequency to which all the readings should be resampled
			%      
			% Outputs:
			%  	 	ecg_and_acc_values - A cell array containing the ecg and sensors values
			%  	 		for each (ecg, sensors) correspondance
			%  	 	heart_rates - The corresponding heart rates
			%------------- BEGIN CODE --------------

			p = inputParser;
			addRequired(p, 'directory');
			addParameter(p, 'metadata_file', 'metadata.csv');
			addParameter(p, 'sensors_dir', 'sensors');
			addParameter(p, 'ecg_file_name', 'ecg.dat');
			addParameter(p, 'acc_period', 0.0194);
			addParameter(p, 'gyr_period', 0.0025);
			addParameter(p, 'ecg_period', 1/250);
			addParameter(p, 'resample_freq', 100);
			addParameter(p, 'skip_delta', 0);
			addParameter(p, 'keep_filenames', {});
			parse(p, directory, varargin{:});

			directory = p.Results.directory;
			metadata_file = p.Results.metadata_file;
			sensors_dir = p.Results.sensors_dir;
			ecg_file_name = p.Results.ecg_file_name;
			acc_period = p.Results.acc_period;
			gyr_period = p.Results.gyr_period;
			ecg_period = p.Results.ecg_period;
			resample_freq = p.Results.resample_freq;
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

				if ecg_file_id > 413
					fprintf("----------------\n")
					continue
				end

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

				[ecg_values, acc_values, gyr_values] = Handrate_md.align_readings(sensors_file, ecg_file, check_time, ...
					'skip_delta', skip_delta, 'ecg_time_length', time_length, 'acc_period', acc_period, ...
					'gyr_period', gyr_period, 'realign_delta', realign_delta);


				% Resample the readings
				fs = 1/acc_period;
				[b, a] = Helper.resample_readings(acc_values(:,2), fs, resample_freq);
				[c, a] = Helper.resample_readings(acc_values(:,3), fs, resample_freq);
				[d, a] = Helper.resample_readings(acc_values(:,4), fs, resample_freq);
				acc_values = [a; b; c; d]';

				fs = 1/gyr_period;
				[b, a] = Helper.resample_readings(gyr_values(:,2), fs, resample_freq);
				[c, a] = Helper.resample_readings(gyr_values(:,3), fs, resample_freq);
				[d, a] = Helper.resample_readings(gyr_values(:,4), fs, resample_freq);
				gyr_values = [a; b; c; d]';

				fs = 1/ecg_period;
				[b, a] = Helper.resample_readings(ecg_values(:,2), fs, resample_freq);
				ecg_values = [a; b]';

				% Make sure that the dimensions match
				m = min(size(ecg_values, 1), size(acc_values, 1));
				ecg_values = ecg_values(1:m, :);
				acc_values = acc_values(1:m, :);
				gyr_values = gyr_values(1:m, :);

				% Concatenate the results in a single matrix in the following format
				% Column 1: time
				% Column 2: ecg values
				% Columns 3-5: acc values
				% Columns 6-8: gyr values
				res = [ecg_values, acc_values(:, 2:end), acc_values(:, 2:end)];

				% Save the result
				ecg_and_sensors_values{end+1} = res;
				heart_rates{end+1} = heart_rate;
				filenames{end+1} = sensors_file;	


				% ------------------- Tmp ---------------------
				% acc_values(:,2) = Helper.filter_noise(acc_values(:,2), ...
				% 	'method', 'wavelet');
				% acc_values(:,3) = Helper.filter_noise(acc_values(:,3), ...
				% 	'method', 'wavelet');
				% acc_values(:,4) = Helper.filter_noise(acc_values(:,4), ...
				% 	'method', 'wavelet');

				% ecg_values = abs(ecg_values);
				% acc_values = abs(sensors_values);
				% subplot(2, 2, 1)
				% plot(ecg_values(:, 1), ecg_values(:,2))
				% hold on
				% plot(acc_values(:, 1), acc_values(:,2)+0.3)
				% plot(acc_values(:, 1), acc_values(:,3))
				% plot(acc_values(:, 1), acc_values(:,4))
				% legend("ECG", "Acc.X", "Acc.Y", "Acc.Z")
				% hold off
				% pause
			end
   		end

   		function [ecg_values, acc_values, gyr_values] = align_readings(sensors_file, ecg_file, ecg_check_time, varargin)
   			% align_readings - Align readings from a sensors readings file to the ones of an ECG file
			% 	
			% Syntax:  [values, time] = Helper.align_readings(sensors_file, ecg_file, ecg_check_time, opt)
			%
			% Inputs:
			%		sensors_file - The sensors readings file
			%		ecg_file - The ecg readings file
			%		ecg_check_time - The time at which the ECG started to be recorded.
			%						 A string in the format 'YYYY-MM-DD HH:MM:SS'
			%  		opt - Optional name-value parameters
			%      
			% Outputs:
			%  	 	
			%------------- BEGIN CODE --------------

			p = inputParser;
			addRequired(p, 'sensors_file');
			addRequired(p, 'ecg_file');
			addRequired(p, 'ecg_check_time');
			addParameter(p, 'ecg_time_length', 30);
			addParameter(p, 'acc_period', 0.0194);
			addParameter(p, 'gyr_period', 0.0025);
			addParameter(p, 'skip_delta', 0);
			addParameter(p, 'realign_delta', 0);
			parse(p, sensors_file, ecg_file, ecg_check_time, varargin{:});

			sensors_file = p.Results.sensors_file;
			ecg_file = p.Results.ecg_file;
			ecg_check_time = p.Results.ecg_check_time;
			ecg_time_length = p.Results.ecg_time_length;
			acc_period = p.Results.acc_period;
			gyr_period = p.Results.gyr_period;
			skip_delta = p.Results.skip_delta;
			realign_delta = p.Results.realign_delta;

			check_time = datetime(ecg_check_time);
			tmp = strsplit(strrep(sensors_file, '.csv', ''), '_');
			sensors_end_time = datetime(tmp{end}, 'InputFormat', 'yyyy-MM-dd:HH:mm:ss');
			tmp = strsplit(sensors_file, 'yrs-');
			tmp = strsplit(tmp{2}, 's-');
			duration = str2num(tmp{1});
			sensors_start_time = sensors_end_time - seconds(duration);
			starts_delta = check_time - sensors_start_time;
			starts_delta = seconds(starts_delta);

			realign_delta_ecg = 0;
			realign_delta_sensors = 0;
			if realign_delta > 0
				realign_delta_sensors = realign_delta;
			else
				realign_delta_ecg = -realign_delta;
			end
			
			ecg_values = Helper.read_ecg_file(ecg_file, 'start_at', skip_delta + realign_delta_ecg);
			acc_values = extract_values(sensors_file, ...
				'period', acc_period, ...
				'start_at', starts_delta + skip_delta + realign_delta_sensors);
				% 'duration', ecg_time_length, ...
			acc_values(:,1) = acc_values(:,1) - acc_values(1,1); % Rebase the timestamps

			gyr_values = extract_values(sensors_file, ...
				'sensor', 'GYR', ...
				'period', gyr_period, ...
				'start_at', starts_delta + skip_delta + realign_delta_sensors);
				% 'duration', ecg_time_length, ...
			gyr_values(:,1) = gyr_values(:,1) - gyr_values(1,1); % Rebase the timestamps
   		end

   		function [heartbeats] = ecg_to_heartbeats(ecg_values, varargin)
   			% ecg_to_heartbeats - Convert an ecg vector to an output one
			% 	
			% Syntax:  [heartbeats] = ecg_to_heartbeats(ecg_values, opt)
			%
			% Inputs:
			%		ecg_values - The ecg readings vector
			%  		opt - Optional name-value parameters
			%  			freq - The sampling frequency of the signal
			%  			min_peak_height - The minimum height to consider a peak (relative to the highest peak)
			%  			heart_rate - The avg heart rate of this measurement
			%  			n_ones - The number of ones to set each time a heartbeat is identified
			%      
			% Outputs:
			%  	 	heartbeats - The output vector (1 at heartbeat time and 0 elsewhere)
			%------------- BEGIN CODE --------------

			p = inputParser;
			addRequired(p, 'ecg_values');
			addParameter(p, 'min_peak_height', 0.4);
			addParameter(p, 'freq', 100);
			addParameter(p, 'heart_rate', 70);
			addParameter(p, 'n_ones', 10);
			parse(p, ecg_values, varargin{:});

			ecg_values = p.Results.ecg_values;
			min_peak_height = p.Results.min_peak_height;
			heart_rate = p.Results.heart_rate;
			n_ones = p.Results.n_ones;
			freq = p.Results.freq;

			% Normalize the input
			ecg_values = ecg_values / max(ecg_values);

			% Find the heartbeat peaks
			min_peak_dist = 0.65 * heart_rate/60;
			[pks, locs] = findpeaks(ecg_values, ...
									'MinPeakHeight', min_peak_height, ...
									'MinPeakDistance', min_peak_dist*freq);

			% Build the output vector
			heartbeats = zeros(size(ecg_values));
			tmp = locs;
			for n=1:n_ones-1
				tmp = [tmp; locs+n];
			end
			locs = tmp;
			heartbeats(locs) = 1;

			% Visualization
			% plot(ecg_values)
			% hold on
			% plot(heartbeats)
			% hold off
			% legend("Original ECG", "Heartbeat instants")
			% yl = ylim();
			% yl(2) = 1.2;
			% ylim(yl);
   		end

   		function [scalograms, sensors_values] = sensors_to_scalogram(sensors_values, varargin)
   			% sensors_to_scalogram - Convert a sensors vector to a scalogram matrix
			% 	
			% Syntax:  [scalogram] = sensors_to_scalogram(sensors_values, opt)
			%
			% Inputs:
			%		acc_values - The sensors values to be processed
			%  		opt - Optional name-value parameters
			%  			freq - The sampling frequency of the signal
			%      
			% Outputs:
			%  	 	scalogram - The output scalogram
			%------------- BEGIN CODE --------------

			p = inputParser;
			addRequired(p, 'sensors_values');
			addParameter(p, 'freq', 100);
			addParameter(p, 'cwt_freq_limits', [2 50]);
			addParameter(p, 'cwt_voices_per_octave', 16);
			addParameter(p, 'cwt_time_bandwidth', 10);
			addParameter(p, 'signal_axis', 1:6);
			addParameter(p, 'denoising_method', 'wavelet');
			parse(p, sensors_values, varargin{:});

			sensors_values = p.Results.sensors_values;
			freq = p.Results.freq;
			cwt_freq_limits = p.Results.cwt_freq_limits;
			cwt_voices_per_octave = p.Results.cwt_voices_per_octave;
			cwt_time_bandwidth = p.Results.cwt_time_bandwidth;
			signal_axis = p.Results.signal_axis;
			denoising_method = p.Results.denoising_method;


			acc_values = sensors_values(:,1:3);
			gyr_values = sensors_values(:,4:6);

			% Detrend the signal
			x = Helper.detrend(acc_values(:,1));
			y = Helper.detrend(acc_values(:,2));
			z = Helper.detrend(acc_values(:,3));
			acc_values = [x, y, z];
			x = Helper.detrend(gyr_values(:,1));
			y = Helper.detrend(gyr_values(:,2));
			z = Helper.detrend(gyr_values(:,3));
			gyr_values = [x, y, z];

			% Denoise the signal
			x = Helper.filter_noise(acc_values(:,1), 'method', denoising_method);
			y = Helper.filter_noise(acc_values(:,2), 'method', denoising_method);
			z = Helper.filter_noise(acc_values(:,3), 'method', denoising_method);
			acc_values = [x, y, z];
			x = Helper.filter_noise(gyr_values(:,1), 'method', denoising_method);
			y = Helper.filter_noise(gyr_values(:,2), 'method', denoising_method);
			z = Helper.filter_noise(gyr_values(:,3), 'method', denoising_method);
			acc_values = [x, y, z];

			% Add PCA first components to the signals
			acc_values_pca = Helper.pca(acc_values);
			gyr_values_pca = Helper.pca(gyr_values);
			sensors_values = [acc_values, gyr_values, acc_values_pca(:,1), gyr_values_pca(:,1)];

			% Normalize the signals
			sensors_values = sensors_values ./ max(abs(sensors_values));

			% Signal axis selection
			sensors_values = sensors_values(:,signal_axis);

			% Continuous Wavelet Transform
			scalograms = [];
			for k=size(sensors_values, 2)
				x = sensors_values(:,k);
				x_noabs = x - mean(x);
				% x = abs(x_noabs);
				x = x_noabs;

				% Compute the Continuous Wavelet Transform
				[wt, f] = cwt(x, freq, ...
								'VoicesPerOctave', cwt_voices_per_octave, ...
							    'TimeBandWidth', cwt_time_bandwidth, ...
							    'FrequencyLimits', cwt_freq_limits);
				scalogram = abs(wt);
				scalogram = scalogram / max(max(scalogram));

				% Visualization
				% image(scalogram, 'CDataMapping','scaled')
				% pause

				scalograms(k, :, :) = scalogram;
			end
   		end

		function [X, Y] = build_dataset(varargin)
			% build_dataset - Build the dataset that can be used to train the neural network
			% 	
			% Syntax:  [X, Y] = Handrate_md.build_dataset(opt)
			%------------- BEGIN CODE --------------

			% Useful variables
			directory = '../../measures/data/';
			metadata_file = 'metadata.csv';
			% metadata_file = 'metadata_files/metadata-usr.csv';
			% metadata_file = 'metadata_files/metadata-usr2.csv';
			acc_period = 0.0194;
			gyr_period = 0.0025;
			ecg_period = 1/250;
			resample_freq = 200;
			n_ones = 5;
			skip_delta = 2;
			item_duration_seconds = 3;
			item_stride_seconds = max(item_duration_seconds-1, 1);
			item_duration = item_duration_seconds * resample_freq;
			item_stride = item_stride_seconds * resample_freq;
			signal_axis = 1:6;
			% signal_axis = 7:8;
			% signal_axis = 1:3;
			% signal_axis = 4:6;
			cwt_freq_limits = [4, 50];
			denoising_method = 'wavelet';

			% Read the data
			fprintf('Read the data...\n');
			% [ecg_and_sensors_values, heart_rates, filenames] = Handrate_md.read_data_from_directory(directory, ...
			% 						'metadata_file', metadata_file, ...
			% 						'acc_period', acc_period, ...
			% 						'gyr_period', gyr_period, ...
			% 						'ecg_period', ecg_period, ...
			% 						'resample_freq', resample_freq, ...
			% 						'skip_delta', skip_delta);

			% save('data_from_directory_md.mat', 'ecg_and_sensors_values', 'heart_rates', 'filenames')
			load('data_from_directory_md.mat')

			% Actual building of the dataset
			fprintf('Actual building of the dataset...\n');
			X = [];
			Y = [];
			heartbeats_matrix = {};
			scalograms_matrix = {};
			n_items = 0;
			n_files = length(ecg_and_sensors_values);
			files_ids = 1:n_files;
			for idx=1:length(files_ids)
				k = files_ids(idx);
				fprintf('--- Processing file %d / %d: %.1f%%: %s\n', k, n_files, k*100/n_files, filenames{k});
				heart_rate = heart_rates{k};
				data = ecg_and_sensors_values{k};
				time = data(:,1);
				ecg_values = data(:,2);
				sensors_values = data(:,3:end);

				% Convert the ECG values to heartbeats vector (1 only where there is a peak)
				heartbeats = Handrate_md.ecg_to_heartbeats(ecg_values, ...
								'heart_rate', heart_rate, ...
								'n_ones', n_ones, ...
								'freq', resample_freq);

				% Convert the sensors values to time-frequency representation
				[scalograms, x] = Handrate_md.sensors_to_scalogram(sensors_values, ...
								'freq', resample_freq, ...
								'signal_axis', signal_axis, ...
								'cwt_freq_limits', cwt_freq_limits, ...
								'denoising_method', denoising_method);


				heartbeats_matrix{k} = heartbeats;
				scalograms = permute(scalograms, [3 2 1]);
				scalograms_matrix{k} = scalograms;
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
			% save('perm_idx_md.mat', 'perm', 'train_idx', 'dev_idx', 'test_idx')
			% load('perm_idx_md.mat')

			fprintf('Bulding training set\n')
			X_train = [];
			Y_train = [];
			n_items = 0;
			for idx=1:length(train_idx)
				k = train_idx(idx);
				heartbeats = heartbeats_matrix{k};
				scalogram = scalograms_matrix{k};

				% Slice the (30s long) data into different items
				n_time_steps = min(length(heartbeats), size(scalogram, 1));
				for begin_time=1:item_stride:n_time_steps-item_duration+1
					n_items = n_items + 1;

					end_time = begin_time + item_duration - 1;
					item_scalogram = scalogram(begin_time:end_time, :, :);
					item_heartbeat = heartbeats(begin_time:end_time);

					X_train(n_items, :, :, :) = item_scalogram;
					Y_train(n_items, :) = item_heartbeat';
				end

				rperm = randperm(n_items);
				X_train = X_train(rperm, :, :, :);
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
				n_time_steps = min(length(heartbeats), size(scalogram, 1));
				for begin_time=1:item_stride:n_time_steps-item_duration+1
					n_items = n_items + 1;

					end_time = begin_time + item_duration - 1;
					item_scalogram = scalogram(begin_time:end_time, :, :);
					item_heartbeat = heartbeats(begin_time:end_time);

					X_dev(n_items, :, :, :) = item_scalogram;
					Y_dev(n_items, :) = item_heartbeat';
				end

				rperm = randperm(n_items);
				X_dev = X_dev(rperm, :, :, :);
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
				n_time_steps = min(length(heartbeats), size(scalogram, 1));
				for begin_time=1:item_stride:n_time_steps-item_duration+1
					n_items = n_items + 1;

					end_time = begin_time + item_duration - 1;
					item_scalogram = scalogram(begin_time:end_time, :, :);
					item_heartbeat = heartbeats(begin_time:end_time);

					X_test(n_items, :, :, :) = item_scalogram;
					Y_test(n_items, :) = item_heartbeat';
				end

				rperm = randperm(n_items);
				X_test = X_test(rperm, :, :, :);
				Y_test = Y_test(rperm, :);
			end

			% Save to file
			signal_axis_list = strrep(num2str(signal_axis), ' ', '');
			dataset_filename = sprintf('datasets/handrate-md-dataset-%ds-%dones-%s.mat', ...
				item_duration_seconds, n_ones, signal_axis_list);
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
			
			dataset_filename = sprintf('datasets/handrate-md-dataset-raw-%ds-%dones-%s.mat', ...
				item_duration_seconds, n_ones, signal_axis_list);
			save(dataset_filename, 'heartbeats_matrix_train', 'scalograms_matrix_train', 'filenames_train', ...
									'heartbeats_matrix_dev', 'scalograms_matrix_dev', 'filenames_dev', ...
									'heartbeats_matrix_test', 'scalograms_matrix_test', 'filenames_test')
			fprintf('Saved data to file %s\n', dataset_filename);
		end

		function [X, Y] = main(varargin)
			% main - Main function
			% 	
			% Syntax:  [] = Handrate_md.main(opt)
			%------------- BEGIN CODE --------------

			[X, Y] = Handrate_md.build_dataset();			
		end
   end
end