classdef Handrate_reg
% Handrate_reg - A class which implements Handrate_reg
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
   		function peak_instant = find_first_peak(heartbeats, varargin)
			% find_first_peak - Find the time instant of the first peak
			% 	
			% Syntax:  [] = Handrate_reg.find_first_peak(opt)
			%------------- BEGIN CODE --------------

			p = inputParser;
			addRequired(p, 'heartbeats');
			addParameter(p, 'sample_freq', 250);
			parse(p, heartbeats, varargin{:});

			heartbeats = p.Results.heartbeats;
			sample_freq = p.Results.sample_freq;

			
			peak_locs = find(heartbeats==1);
			% peak_instant = peak_locs(1) / sample_freq;
			if length(peak_locs) > 0
				peak_instant = peak_locs(1) / sample_freq;
			else
				peak_instant = -1;
			end
		end

   		function [X, Y] = build_dataset(varargin)
			% build_dataset - Build the dataset that can be used to train the neural network
			% 	
			% Syntax:  [X, Y] = Handrate_reg.build_dataset(opt)
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
			n_ones = 1;
			skip_delta = 2;
			item_duration_seconds = 1.5;
			item_stride_seconds = max(item_duration_seconds-0.5, 0.5);
			item_duration = item_duration_seconds * resample_freq;
			item_stride = item_stride_seconds * resample_freq;
			item_duration_ecg = item_duration_seconds * resample_freq_ecg;
			item_stride_ecg = item_stride_seconds * resample_freq_ecg;
			signal_axis = 'pca';
			cwt_freq_limits = [4, 50];
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
					Y_train(n_items) = Handrate_reg.find_first_peak(item_heartbeat, 'sample_freq', resample_freq_ecg);
					begin_time_ecg = begin_time_ecg + item_stride_ecg;
				end

				rperm = randperm(n_items);
				X_train = X_train(rperm, :, :);
				Y_train = Y_train(rperm);
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
					Y_dev(n_items) = Handrate_reg.find_first_peak(item_heartbeat, 'sample_freq', resample_freq_ecg);
					begin_time_ecg = begin_time_ecg + item_stride_ecg;
				end

				rperm = randperm(n_items);
				X_dev = X_dev(rperm, :, :);
				Y_dev = Y_dev(rperm);
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
					Y_test(n_items) = Handrate_reg.find_first_peak(item_heartbeat, 'sample_freq', resample_freq_ecg);
					begin_time_ecg = begin_time_ecg + item_stride_ecg;
				end

				rperm = randperm(n_items);
				X_test = X_test(rperm, :, :);
				Y_test = Y_test(rperm);
			end

			% Save to file
			dataset_filename = sprintf('datasets/handrate-reg-dataset-%.1fs-%dones-%s.mat', ...
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
			
			dataset_filename = sprintf('datasets/handrate-reg-dataset-raw-%.1fs-%dones-%s.mat', ...
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

			[X, Y] = Handrate_reg.build_dataset();			
		end
   end
end