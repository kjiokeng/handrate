classdef Handrate_users
% Handrate_users - A class which implements Handrate_users
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
			% Syntax:  [X, Y] = Handrate_users.build_dataset(opt)
			%------------- BEGIN CODE --------------

			% Useful variables
			directory = '../../measures/data/';
			metadata_file = 'metadata.csv';
			% metadata_file = 'metadata_files/metadata-usr.csv';
			% metadata_file = 'metadata_files/metadata-usr2.csv';
			sensors_period = 0.0194;
			ecg_period = 1/250;
			resample_freq = 100;
			n_ones = 10;
			skip_delta = 2;
			item_duration_seconds = 3;
			item_stride_seconds = max(item_duration_seconds-1, 1);
			% item_stride_seconds = item_duration_seconds;
			item_duration = item_duration_seconds * resample_freq;
			item_stride = item_stride_seconds * resample_freq;
			signal_axis = 'pca';
			cwt_freq_limits = [4, 50];
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

			% Split for different users
			users = {};
			ecg_and_sensors_values_users = {};
			heart_rates_users = {};
			filenames_users = {};

			last_user_name = '';
			current_ecg_and_sensors_values = {};
			current_heart_rates = {};
			current_filenames = {};
			for k=1:length(filenames)
				current_filename = filenames{k};
				tmp = strsplit(current_filename, '/');
				tmp = strsplit(tmp{end}, '-');
				user = tmp{1};

				if ~strcmp(user, last_user_name)
					% Save the previous user's info and pass to the next one
					if k~=1
						ecg_and_sensors_values_users{end+1} = current_ecg_and_sensors_values;
						heart_rates_users{end+1} = current_heart_rates;
						filenames_users{end+1} = current_filenames;
					end

					users{end+1} = user;
					current_ecg_and_sensors_values = {};
					current_heart_rates = {};
					current_filenames = {};
				end
				
				% Add this info to the ones of the current user
				current_ecg_and_sensors_values{end+1} = ecg_and_sensors_values{k};
				current_heart_rates{end+1} = heart_rates{k};
				current_filenames{end+1} = filenames{k};

				last_user_name = user;
			end


			% Actual building of the dataset
			fprintf('Actual building of the datasets...\n');

			n_users = length(users);

			for user_id = 1:n_users
				user_name = users{user_id};
				ecg_and_sensors_values = ecg_and_sensors_values_users{user_id};
				heart_rates = heart_rates_users{user_id};

				fprintf('--- Bulding dataset for user %d / %d: %.1f%%: %s\n', user_id, n_users, user_id*100/n_users, user_name);

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
				% save('perm_idx.mat', 'perm', 'train_idx', 'dev_idx', 'test_idx')
				% load('perm_idx.mat')

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
					for begin_time=1:item_stride:n_time_steps-item_duration+1
						n_items = n_items + 1;

						end_time = begin_time + item_duration - 1;
						item_scalogram = scalogram(:,begin_time:end_time);
						item_heartbeat = heartbeats(begin_time:end_time);

						item_scalogram = item_scalogram / max(max(item_scalogram));
						X_train(n_items, :, :) = item_scalogram';
						Y_train(n_items, :) = item_heartbeat';


						% close all
						% fig = figure('units','normalized','outerposition',[0 .35 1 .4]);
						% surface(item_scalogram)
						% axis tight
						% shading flat
						% set(gca, 'yscale', 'log')
						% drawnow

						% hold on
						% yyaxis right
						% plot(item_heartbeat, 'r', 'LineWidth', 2)
						% ylim([-0.05 1.05])
						% pause
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
					for begin_time=1:item_stride:n_time_steps-item_duration+1
						n_items = n_items + 1;

						end_time = begin_time + item_duration - 1;
						item_scalogram = scalogram(:,begin_time:end_time);
						item_heartbeat = heartbeats(begin_time:end_time);

						item_scalogram = item_scalogram / max(max(item_scalogram));
						X_dev(n_items, :, :) = item_scalogram';
						Y_dev(n_items, :) = item_heartbeat';
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
					for begin_time=1:item_stride:n_time_steps-item_duration+1
						n_items = n_items + 1;

						end_time = begin_time + item_duration - 1;
						item_scalogram = scalogram(:,begin_time:end_time);
						item_heartbeat = heartbeats(begin_time:end_time);

						item_scalogram = item_scalogram / max(max(item_scalogram));
						X_test(n_items, :, :) = item_scalogram';
						Y_test(n_items, :) = item_heartbeat';
					end

					rperm = randperm(n_items);
					X_test = X_test(rperm, :, :);
					Y_test = Y_test(rperm, :);
				end

				% Save to file
				dataset_filename = sprintf('datasets/users/handrate-dataset-%s-%ds-%dones-%s.mat', ...
					user_name, item_duration_seconds, n_ones, signal_axis);
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
				
				dataset_filename = sprintf('datasets/users/handrate-dataset-raw-%s-%ds-%dones-%s.mat', ...
					user_name, item_duration_seconds, n_ones, signal_axis);
				save(dataset_filename, 'heartbeats_matrix_train', 'scalograms_matrix_train', 'filenames_train', ...
										'heartbeats_matrix_dev', 'scalograms_matrix_dev', 'filenames_dev', ...
										'heartbeats_matrix_test', 'scalograms_matrix_test', 'filenames_test')
				fprintf('Saved data to file %s\n', dataset_filename);
			end
		end

		function [X, Y] = main(varargin)
			% main - Main function
			% 	
			% Syntax:  [] = Handrate_users.main(opt)
			%------------- BEGIN CODE --------------

			[X, Y] = Handrate_users.build_dataset();			
		end
   end
end