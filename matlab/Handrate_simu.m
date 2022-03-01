classdef Handrate_simu
% Handrate_simu - A class which implements Handrate_simu
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
   		function [ecg_and_sensors] = generate_data(heart_rate, varargin)
   			p = inputParser;
			addRequired(p, 'heart_rate');
			addParameter(p, 'n_samples', 1);
			addParameter(p, 'snr', 5);
			addParameter(p, 'duration', 30);
			addParameter(p, 'sample_freq', 100);
			addParameter(p, 'f_oscil', 10);
			parse(p, heart_rate, varargin{:});

			heart_rate = p.Results.heart_rate;
			n_samples = p.Results.n_samples;
			snr = p.Results.snr;
			duration = p.Results.duration;
			sample_freq = p.Results.sample_freq;
			f_oscil = p.Results.f_oscil;


			% Actual signal generation
			t_sampling = 1/sample_freq;
			hr_per_second = heart_rate/60;
			cycle_duration_seconds = 1/hr_per_second;
			cycle_duration_units = round(cycle_duration_seconds*sample_freq);

			t_cycle_seconds = 0:t_sampling:cycle_duration_seconds;
			main_sine_wave = sin(2*pi*f_oscil*t_cycle_seconds);
			expo_decrease = exp(-20*t_cycle_seconds);
			signal_cycle = main_sine_wave .* expo_decrease;


			n_cyles = ceil(duration/cycle_duration_seconds);
			beats_random_shift = round(randn(n_samples*n_cyles, 1) * cycle_duration_units/20);
			global_cycle_idx = 1;
			ecg_and_sensors = [];
			for k=1:n_samples
				time = 0:t_sampling:duration-t_sampling;
				sample = zeros(1, length(time));
				ecg = zeros(1, length(time));
				random_begin = randi([1 cycle_duration_units-1], 1);

				for cycle_id=1:n_cyles
					random_shift = beats_random_shift(global_cycle_idx);
					global_cycle_idx = global_cycle_idx + 1;
					begin_idx = (cycle_id-1) * cycle_duration_units + random_begin + random_shift;
					begin_idx = max(begin_idx, 1);
					if begin_idx > length(time)
						continue
					end

					end_idx = min(begin_idx + cycle_duration_units - 1, length(time));
					current_cycle_length = end_idx - begin_idx + 1;

					sample(begin_idx:end_idx) = signal_cycle(1:current_cycle_length);
					ecg(begin_idx) = 1;
				end

				% Add noise to the signal sample
				noise = (randn(1, length(time)) - 0.5);
				noise = noise * max(abs(sample)) / (max(abs(noise)) * snr);
				sample = sample + noise;

				% Add to results: add the same signal as x, y and z
				ecg_and_sensors(k, :, :) = [time', ecg', sample', sample', sample'];
			end
   		end

		function [ecg_and_sensors_values, heart_rates, filenames] = generate_data_for_configs(configs, varargin)
			% configs should be an n x 3 matrix. Each row in the format heart_rate, n_samples, snr

			p = inputParser;
			addRequired(p, 'configs');
			addParameter(p, 'duration', 30);
			addParameter(p, 'sample_freq', 100);
			parse(p, configs, varargin{:});

			configs = p.Results.configs;
			duration = p.Results.duration;
			sample_freq = p.Results.sample_freq;

			if size(configs, 2) ~= 3
				error('Argument dimension mismatch: config must have 3 columns');
			end

			ecg_and_sensors_values = {};
			heart_rates = {};
			filenames = {};
			n_configs = size(configs, 1);
			for k=1:n_configs
				heart_rate = configs(k, 1);
				n_samples = configs(k, 2);
				snr = configs(k, 3);

				% Generate data for the given config
				ecg_and_sensors = Handrate_simu.generate_data(heart_rate, ...
					'n_samples', n_samples, ...
					'snr', snr, ...
					'duration', duration, ...
					'sample_freq', sample_freq);

				% Add the generated data to the result
				n_samples = size(ecg_and_sensors, 1);
				for sample=1:n_samples
					tmp = ecg_and_sensors(sample, :, :);
					ecg_and_sensors_values{end+1} = reshape(tmp, [size(tmp, 2), size(tmp, 3)]);
					heart_rates{end+1} = heart_rate;
					filenames{end+1} = 'GENERATED_DATA';
				end
			end
		end

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
			sensors_period = 0.0194;
			ecg_period = 1/250;
			resample_freq = 100;
			n_ones = 5;
			skip_delta = 2;
			item_duration_seconds = 1;
			item_stride_seconds = max(item_duration_seconds-1, 1);
			item_duration = item_duration_seconds * resample_freq;
			item_stride = item_stride_seconds * resample_freq;
			signal_axis = 'pca';
			cwt_freq_limits = [4, 50];
			denoising_method = 'none';

			% Read the data
			fprintf('Read the data...\n');
			% Config matrix. Each row in the format heart_rate, n_samples, snr
			configs = [ ...
				60, 10, 0.9 ...
				; 65, 25, 0.8 ...
				; 80, 10, 0.8 ...
				; 57, 10, 0.8 ...
				; 70, 15, 0.8 ...
				; 65, 10, 0.7 ...
				; 70, 10, 0.6 ...
				; 77, 10, 0.6 ...
				; 85, 15, 0.6 ...
				; 63, 10, 0.5 ...
				; 76, 15, 0.5 ...
				; 85, 10, 0.5 ...
			];
			% configs(:,3) = configs(:,3) + 0.2;
			% snrtype = 'goodsnr';

			configs(:,3) = configs(:,3) - 0.15;
			snrtype = 'lowsnr';
			% snrtype = 'lowsnrdenoised';

			[ecg_and_sensors_values, heart_rates, filenames] = Handrate_simu.generate_data_for_configs(configs);
			
			% X = ecg_and_sensors_values;
			% Y = filenames;
			% for k=1:length(ecg_and_sensors_values)
			% 	tmp = ecg_and_sensors_values{k};
			% 	time = tmp(:, 1);
			% 	ecg = tmp(:, 2);
			% 	x = tmp(:, 3);

			% 	subplot(2, 1, 1)
			% 	plot(time, x)
				
			% 	subplot(2, 1, 2)
			% 	plot(time, ecg)

			% 	pause
			% end
			% return

			% save('simulated-data.mat', 'ecg_and_sensors_values', 'heart_rates', 'filenames')
			% load('simulated-data.mat')

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

				% Convert the sensors values to time-frequency representation
				[scalogram, x] = Handrate.sensors_to_scalogram(sensors_values, ...
								'freq', resample_freq, ...
								'signal_axis', signal_axis, ...
								'cwt_freq_limits', cwt_freq_limits, ...
								'denoising_method', denoising_method);

				% [scalogram, x] = Handrate_scripts.sensors_to_scalogram(sensors_values, ...
				% 				ecg_values, ...
				% 				'freq', resample_freq, ...
				% 				'signal_axis', signal_axis, ...
				% 				'denoising_method', denoising_method);


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
			% save('perm_idx-simu.mat', 'perm', 'train_idx', 'dev_idx', 'test_idx')
			% load('perm_idx-simu.mat')

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
			dataset_filename = sprintf('datasets/handrate-simu-dataset-%ds-%dones-%s%s.mat', ...
				item_duration_seconds, n_ones, signal_axis, snrtype);
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
			
			dataset_filename = sprintf('datasets/handrate-simu-dataset-raw-%ds-%dones-%s%s.mat', ...
				item_duration_seconds, n_ones, signal_axis, snrtype);
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

			[X, Y] = Handrate_simu.build_dataset();			
		end
   end
end