classdef Handrate_nosplit
% Handrate - A class which implements Handrate
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
		function [X, Y] = build_dataset(varargin)
			% build_dataset - Build the dataset that can be used to train the neural network
			% 	
			% Syntax:  [X, Y] = Handrate.build_dataset(opt)
			%------------- BEGIN CODE --------------

			% Useful variables
			directory = '../../measures/data/';
			metadata_file = 'metadata.csv';
			% metadata_file = 'metadata_files/metadata-usr.csv';
			% metadata_file = 'metadata_files/metadata-usr2.csv';
			% metadata_file = 'metadata_files/experimental-conditions/metadata-vertical.csv';
			sensors_period = 0.0194;
			ecg_period = 1/250;
			resample_freq = 100;
			n_ones = 20;
			skip_delta = 2;
			item_duration_seconds = 3;
			item_stride_seconds = max(item_duration_seconds-1, 1);
			% item_stride_seconds = item_duration_seconds;
			item_duration = item_duration_seconds * resample_freq;
			item_stride = item_stride_seconds * resample_freq;
			signal_axis = 'pca';
			cwt_freq_limits = [2, 50];
			denoising_method = 'wavelet';

			% Read the data
			fprintf('Read the data...\n');
			% [ecg_and_sensors_values, heart_rates, filenames] = Handrate.read_data_from_directory(directory, ...
			% 						'metadata_file', metadata_file, ...
			% 						'sensors_period', sensors_period, ...
			% 						'ecg_period', ecg_period, ...
			% 						'resample_freq', resample_freq, ...
			% 						'skip_delta', skip_delta);

			% save('data_from_directory.mat', 'ecg_and_sensors_values', 'heart_rates', 'filenames')
			load('data_from_directory.mat')

			% hist(cell2mat(heart_rates))
			% pause

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
				sensors_values = data(:,3:5);

				% Convert the ECG values to heartbeats vector (1 only where there is a peak)
				heartbeats = Handrate.ecg_to_heartbeats(ecg_values, ...
								'heart_rate', heart_rate, ...
								'n_ones', n_ones, ...
								'freq', resample_freq);

				% plot(ecg_values)
				% hold on
				% plot(heartbeats/2)
				% hold off
				% pause

				% Convert the sensors values to time-frequency representation
				[scalogram, x] = Handrate.sensors_to_scalogram(sensors_values, ...
								'freq', resample_freq, ...
								'signal_axis', signal_axis, ...
								'cwt_freq_limits', cwt_freq_limits, ...
								'denoising_method', denoising_method);


				heartbeats_matrix{k} = heartbeats;
				scalograms_matrix{k} = scalogram;
			end


			%% Grouping filenames by users
			user_files_map = containers.Map();
			user_filenames_map = containers.Map();
			for file_idx=1:n_files
				filename = filenames{file_idx};
				tmp = strsplit(filename, '-');
				tmp = strsplit(tmp{1}, '/');
				user = tmp{end};

				if isKey(user_files_map, user)
					tmp = user_files_map(user);
					tmp{end+1} = file_idx;
					user_files_map(user) = tmp;

					tmp = user_filenames_map(user);
					tmp{end+1} = filename;
					user_filenames_map(user) = tmp;					
				else
					user_files_map(user) = {file_idx};
					user_filenames_map(user) = {filename};
				end 
			end

			X = {};
			Y = {};
			users = keys(user_files_map);
			for u=1:length(users)
				user = users{u};
				user_file_idx = user_files_map(user);

				user_X = [];
				user_Y = [];
				n_items = 0;
				for idx=1:length(user_file_idx)
					k = user_file_idx{idx};
					heartbeats = heartbeats_matrix{k};
					scalogram = scalograms_matrix{k};

					% Slice the (30s long) data into different items
					n_time_steps = min(length(heartbeats), size(scalogram, 2));
					for begin_time=1:item_stride:n_time_steps-item_duration+1
						n_items = n_items + 1;

						end_time = begin_time + item_duration - 1;
						item_scalogram = scalogram(:,begin_time:end_time);
						item_heartbeat = heartbeats(begin_time:end_time);

						item_scalogram = item_scalogram / max(max(item_scalogram));
						user_X(n_items, :, :) = item_scalogram';
						user_Y(n_items, :) = item_heartbeat';
					end

					rperm = randperm(n_items);
					user_X = user_X(rperm, :, :);
					user_Y = user_Y(rperm, :);

					X{u} = user_X;
					Y{u} = user_Y;
				end
			end

			% Save to file
			dataset_filename = sprintf('datasets/handrate-dataset-XY-%ds-%dones-%s.mat', ...
				item_duration_seconds, n_ones, signal_axis);
			save(dataset_filename, 'X', 'Y', 'users')
			fprintf('Saved data to file %s\n', dataset_filename);
			% pause


			% Save raw data
			dataset_filename = sprintf('datasets/handrate-dataset-XY-raw-%ds-%dones-%s.mat', ...
				item_duration_seconds, n_ones, signal_axis);
			save(dataset_filename, 'heartbeats_matrix', 'scalograms_matrix', 'filenames', ...
									'users', 'user_files_map', 'user_filenames_map')
			fprintf('Saved data to file %s\n', dataset_filename);
		end

		function [X, Y] = main(varargin)
			% main - Main function
			% 	
			% Syntax:  [] = Handrate.main(opt)
			%------------- BEGIN CODE --------------

			[X, Y] = Handrate_nosplit.build_dataset();			
		end
   end
end