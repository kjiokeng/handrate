classdef Handrate_spect
% Handrate_spect - A class which implements Handrate_spect
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

   		function [x, x_noabs] = preprocess_sensors(sensors_values, varargin)
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
			addParameter(p, 'freq', 100);
			addParameter(p, 'cwt_freq_limits', [2 50]);
			addParameter(p, 'cwt_voices_per_octave', 16);
			addParameter(p, 'cwt_time_bandwidth', 10);
			addParameter(p, 'signal_axis', 'pca');
			addParameter(p, 'denoising_method', 'wavelet');
			parse(p, sensors_values, varargin{:});

			sensors_values = p.Results.sensors_values;
			freq = p.Results.freq;
			cwt_freq_limits = p.Results.cwt_freq_limits;
			cwt_voices_per_octave = p.Results.cwt_voices_per_octave;
			cwt_time_bandwidth = p.Results.cwt_time_bandwidth;
			signal_axis = p.Results.signal_axis;
			denoising_method = p.Results.denoising_method;

			% Normalize the input
			x = Helper.detrend(sensors_values(:,1));
			y = Helper.detrend(sensors_values(:,2));
			z = Helper.detrend(sensors_values(:,3));
			sensors_values = [x, y, z];

			% Denoise the signal
			sensors_values(:,1) = Helper.filter_noise(sensors_values(:,1), ...
				'method', denoising_method);
			sensors_values(:,2) = Helper.filter_noise(sensors_values(:,2), ...
				'method', denoising_method);
			sensors_values(:,3) = Helper.filter_noise(sensors_values(:,3), ...
				'method', denoising_method);

			sensors_values = sensors_values ./ max(abs(sensors_values));

			% Signal combination
			% old_sensors_values = sensors_values;
			switch signal_axis
				case 'pca'
					% [sensors_values, var_ret, U, S] = Helper.pca(sensors_values);
					[sensors_values, var_ret, U, S] = Helper.pca(abs(sensors_values));
				case 'x'
					sensors_values = x;
				case 'y'
					sensors_values = y;
				case 'z'
					sensors_values = z;
			end
			
			% positions = [4 0 -4];
			% subplot(2, 1, 1)
			% plot(old_sensors_values+positions)
			% legend("X", "Y", "Z")

			% subplot(2, 1, 2)
			% plot(sensors_values+positions)
			% legend("X", "Y", "Z")
			% pause

			% Select one axis
			x = sensors_values(:,1);
			x_noabs = x - mean(x);
			% x = abs(x_noabs);
			x = x_noabs;
   		end

   		function [spect, x] = sensors_to_spectrogram(x, varargin)
   			% sensors_to_scalogram - Convert a sensors vector to a scalogram matrix
			% 	
			% Syntax:  [scalogram] = sensors_to_scalogram(x, opt)
			%
			% Inputs:
			%		x - The sensors values to be processed
			%  		opt - Optional name-value parameters
			%  			freq - The sampling frequency of the signal
			%      
			% Outputs:
			%  	 	scalogram - The output scalogram
			%------------- BEGIN CODE --------------

			p = inputParser;
			addRequired(p, 'x');
			addParameter(p, 'freq', 100);
			addParameter(p, 'cwt_freq_limits', [2 50]);
			addParameter(p, 'cwt_voices_per_octave', 16);
			addParameter(p, 'cwt_time_bandwidth', 10);
			addParameter(p, 'signal_axis', 'pca');
			addParameter(p, 'denoising_method', 'wavelet');
			parse(p, x, varargin{:});

			x = p.Results.x;
			freq = p.Results.freq;
			cwt_freq_limits = p.Results.cwt_freq_limits;
			cwt_voices_per_octave = p.Results.cwt_voices_per_octave;
			cwt_time_bandwidth = p.Results.cwt_time_bandwidth;
			signal_axis = p.Results.signal_axis;
			denoising_method = p.Results.denoising_method;

			
			% Compute the Continuous Wavelet Transform
			[wt, f, t] = pspectrum(x, freq, 'spectrogram', ...
							'OverlapPercent', 99, ...
						    'leakage', 0.5, ...
						    'TimeResolution', 0.2);
			spect = abs(wt);
			spect = spect(f<25, :);
			spect = spect(1:2:size(spect, 1), :);
			spect = spect / max(max(spect));


			% vv = size(x)
			% rr = size(spect)

			% Visualization
			% image(spect, 'CDataMapping','scaled', 'XData', t, 'YData', f)
			% pause
   		end

		function [X, Y] = build_dataset(varargin)
			% build_dataset - Build the dataset that can be used to train the neural network
			% 	
			% Syntax:  [X, Y] = Handrate_spect.build_dataset(opt)
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
			cwt_freq_limits = [0, 50];
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
				% [scalogram, x] = Handrate_spect.sensors_to_spectrogram(sensors_vals, ...
				% 				'freq', resample_freq, ...
				% 				'signal_axis', signal_axis, ...
				% 				'cwt_freq_limits', cwt_freq_limits, ...
				% 				'denoising_method', denoising_method);

				heartbeats_matrix{k} = heartbeats;
				% scalograms_matrix{k} = scalogram;
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
				sensors_vals = sensors_values{k};
				fprintf('--- [TRAIN] Processing file %d / %d: %.1f%%: %s\n', idx, m_train, idx*100/m_train, filenames{k});

				sensors_vals = Handrate_spect.preprocess_sensors(sensors_vals, ...
								'freq', resample_freq, ...
								'signal_axis', signal_axis, ...
								'cwt_freq_limits', cwt_freq_limits, ...
								'denoising_method', denoising_method);

				% Slice the (30s long) data into different items
				n_time_steps = min(length(heartbeats), size(sensors_vals, 1));
				begin_time_ecg = 1;
				for begin_time=1:item_stride:n_time_steps-item_duration+1
					n_items = n_items + 1;

					end_time = begin_time + item_duration - 1;
					sens_vals = sensors_vals(begin_time:end_time, :);
					[scalogram, x] = Handrate_spect.sensors_to_spectrogram(sens_vals, ...
								'freq', resample_freq, ...
								'signal_axis', signal_axis, ...
								'cwt_freq_limits', cwt_freq_limits, ...
								'denoising_method', denoising_method);
					scalogram = scalogram / max(max(scalogram));
					X_train(n_items, :, :) = scalogram';


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
				sensors_vals = sensors_values{k};
				fprintf('--- [DEV] Processing file %d / %d: %.1f%%: %s\n', idx, m_dev, idx*100/m_dev, filenames{k});

				sensors_vals = Handrate_spect.preprocess_sensors(sensors_vals, ...
								'freq', resample_freq, ...
								'signal_axis', signal_axis, ...
								'cwt_freq_limits', cwt_freq_limits, ...
								'denoising_method', denoising_method);

				% Slice the (30s long) data into different items
				n_time_steps = min(length(heartbeats), size(sensors_vals, 1));
				begin_time_ecg = 1;
				for begin_time=1:item_stride:n_time_steps-item_duration+1
					n_items = n_items + 1;

					end_time = begin_time + item_duration - 1;
					sens_vals = sensors_vals(begin_time:end_time, :);
					[scalogram, x] = Handrate_spect.sensors_to_spectrogram(sens_vals, ...
								'freq', resample_freq, ...
								'signal_axis', signal_axis, ...
								'cwt_freq_limits', cwt_freq_limits, ...
								'denoising_method', denoising_method);
					scalogram = scalogram / max(max(scalogram));
					X_dev(n_items, :, :) = scalogram';

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
				sensors_vals = sensors_values{k};
				fprintf('--- [TEST] Processing file %d / %d: %.1f%%: %s\n', idx, m_test, idx*100/m_test, filenames{k});

				sensors_vals = Handrate_spect.preprocess_sensors(sensors_vals, ...
								'freq', resample_freq, ...
								'signal_axis', signal_axis, ...
								'cwt_freq_limits', cwt_freq_limits, ...
								'denoising_method', denoising_method);

				% Slice the (30s long) data into different items
				n_time_steps = min(length(heartbeats), size(sensors_vals, 1));
				begin_time_ecg = 1;
				for begin_time=1:item_stride:n_time_steps-item_duration+1
					n_items = n_items + 1;

					end_time = begin_time + item_duration - 1;
					sens_vals = sensors_vals(begin_time:end_time, :);
					[scalogram, x] = Handrate_spect.sensors_to_spectrogram(sens_vals, ...
								'freq', resample_freq, ...
								'signal_axis', signal_axis, ...
								'cwt_freq_limits', cwt_freq_limits, ...
								'denoising_method', denoising_method);
					scalogram = scalogram / max(max(scalogram));
					X_test(n_items, :, :) = scalogram';

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
			dataset_filename = sprintf('datasets/handrate-spect-dataset-%.1fs-%dones-%s.mat', ...
				item_duration_seconds, n_ones, signal_axis);
			save(dataset_filename, 'X_train', 'Y_train', 'X_dev', 'Y_dev', 'X_test', 'Y_test')
			fprintf('Saved data to file %s\n', dataset_filename);

			% % Save raw data
			% heartbeats_matrix_train = heartbeats_matrix(train_idx);
			% scalograms_matrix_train = scalograms_matrix(train_idx);
			% filenames_train = filenames(train_idx);
			% heartbeats_matrix_dev = heartbeats_matrix(dev_idx);
			% scalograms_matrix_dev = scalograms_matrix(dev_idx);
			% filenames_dev = filenames(dev_idx);
			% heartbeats_matrix_test = heartbeats_matrix(test_idx);
			% scalograms_matrix_test = scalograms_matrix(test_idx);
			% filenames_test = filenames(test_idx);
			
			% dataset_filename = sprintf('datasets/handrate-spect-dataset-raw-%.1fs-%dones-%s.mat', ...
			% 	item_duration_seconds, n_ones, signal_axis);
			% save(dataset_filename, 'heartbeats_matrix_train', 'scalograms_matrix_train', 'filenames_train', ...
			% 						'heartbeats_matrix_dev', 'scalograms_matrix_dev', 'filenames_dev', ...
			% 						'heartbeats_matrix_test', 'scalograms_matrix_test', 'filenames_test')
			% fprintf('Saved data to file %s\n', dataset_filename);
		end

		function [X, Y] = main(varargin)
			% main - Main function
			% 	
			% Syntax:  [] = Handrate.main(opt)
			%------------- BEGIN CODE --------------

			[X, Y] = Handrate_spect.build_dataset();			
		end
   end
end