classdef Handrate
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
   		function [ecg_and_sensors_values, heart_rates] = read_data_from_directory(directory, varargin)
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

			% Actual computation
			T = readtable(metadata_file, 'DatetimeType', 'text');
			[n_rows, n_cols] = size(T);

			ecg_and_sensors_values = {};
			heart_rates = {};
			for k=1:n_rows
				ecg_file_id = table2array(T(k, ecg_file_id_column_name));
				time_length = table2array(T(k, time_length_column_name));
				heart_rate = table2array(T(k, heart_rate_column_name));
				check_time = table2array(T(k, check_time_column_name));
				check_time = check_time{1};
				sensors_file = table2cell(T(k, sensors_file_column_name));
				sensors_file = sensors_file{1};

				if isempty(sensors_file) || isempty(check_time) || isempty(ecg_file_id) || isempty(time_length)
					continue
				end

				tmp = strsplit(check_time, ' ');
				ecg_date = tmp{1};
				ecg_file = strcat(directory, '/', ecg_date, '/', num2str(ecg_file_id), '/', ecg_file_name);
				sensors_file = strcat(sensors_dir, '/', sensors_file);

				[ecg_values, sensors_values] = Handrate.align_readings(sensors_file, ecg_file, check_time, ...
					'skip_delta', skip_delta, 'ecg_time_length', time_length, 'sensors_period', sensors_period);


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

   		function [ecg_values, sensors_values] = align_readings(sensors_file, ecg_file, ecg_check_time, varargin)
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
			addParameter(p, 'sensors_period', 0.0194);
			addParameter(p, 'skip_delta', 0);
			parse(p, sensors_file, ecg_file, ecg_check_time, varargin{:});

			sensors_file = p.Results.sensors_file;
			ecg_file = p.Results.ecg_file;
			ecg_check_time = p.Results.ecg_check_time;
			ecg_time_length = p.Results.ecg_time_length;
			sensors_period = p.Results.sensors_period;
			skip_delta = p.Results.skip_delta;

			check_time = datetime(ecg_check_time);
			tmp = strsplit(strrep(sensors_file, '.csv', ''), '_');
			sensors_end_time = datetime(tmp{end}, 'InputFormat', 'yyyy-MM-dd:HH:mm:ss');
			tmp = strsplit(sensors_file, 'yrs-');
			tmp = strsplit(tmp{2}, 's-');
			duration = str2num(tmp{1});
			sensors_start_time = sensors_end_time - seconds(duration);
			starts_delta = check_time - sensors_start_time;
			starts_delta = seconds(starts_delta);
			
			ecg_values = Helper.read_ecg_file(ecg_file, 'start_at', skip_delta);
			sensors_values = extract_values(sensors_file, ...
				'period', sensors_period, ...
				'start_at', starts_delta + skip_delta);
				% 'duration', ecg_time_length, ...
			sensors_values(:,1) = sensors_values(:,1) - sensors_values(1,1); % Rebase the timestamps
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
			min_peak_dist = 0.6 * heart_rate/60;
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

   		function [scalogram, x] = sensors_to_scalogram(sensors_values, varargin)
   			% sensors_to_scalogram - Convert an sensors vector to a scalogram matrix
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
			parse(p, sensors_values, varargin{:});

			sensors_values = p.Results.sensors_values;
			freq = p.Results.freq;
			cwt_freq_limits = p.Results.cwt_freq_limits;
			cwt_voices_per_octave = p.Results.cwt_voices_per_octave;
			cwt_time_bandwidth = p.Results.cwt_time_bandwidth;

			% Normalize the input
			x = Helper.detrend(sensors_values(:,1));
			y = Helper.detrend(sensors_values(:,2));
			z = Helper.detrend(sensors_values(:,3));
			sensors_values = [x, y, z];
			%sensors_values = sensors_values - mean(sensors_values);
			sensors_values = sensors_values ./ max(abs(sensors_values));

			% Denoise the signal
			sensors_values(:,1) = Helper.filter_noise(sensors_values(:,1), ...
				'method', 'wavelet');
			sensors_values(:,2) = Helper.filter_noise(sensors_values(:,2), ...
				'method', 'wavelet');
			sensors_values(:,3) = Helper.filter_noise(sensors_values(:,3), ...
				'method', 'wavelet');


			% Signal combination
			% old_sensors_values = sensors_values;
			[sensors_values, var_ret, U, S] = Helper.pca(sensors_values);
			
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
			% Syntax:  [X, Y] = Handrate.build_dataset(opt)
			%------------- BEGIN CODE --------------

			% Useful variables
			directory = '../../measures/data/';
			% metadata_file = 'metadata_files/metadata-user2.csv';
			metadata_file = 'metadata.csv';
			sensors_period = 0.0194;
			ecg_period = 1/250;
			resample_freq = 100;
			cwt_voices_per_octave = 16;
			n_ones = 10;
			skip_delta = 2;
			item_duration_seconds = 3;
			item_stride_seconds = 2;
			item_duration = item_duration_seconds * resample_freq;
			item_stride = item_duration_seconds * resample_freq;

			% Read the data
			fprintf('Read the data...\n');
			[ecg_and_sensors_values, heart_rates] = Handrate.read_data_from_directory(directory, ...
									'metadata_file', metadata_file, ...
									'sensors_period', sensors_period, ...
									'ecg_period', ecg_period, ...
									'resample_freq', resample_freq, ...
									'skip_delta', skip_delta);


			% Actual building of the dataset
			fprintf('Actual building of the dataset...\n');
			X = [];
			Y = [];
			n_items = 0;
			n_files = length(ecg_and_sensors_values);
			for k=1:n_files
				fprintf('--- Processing file %d / %d: %.1f%%\n', k, n_files, k*100/n_files);
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
								'cwt_voices_per_octave', cwt_voices_per_octave);

				% subplot(3, 1, 1)
				% plot(heartbeats)

				% subplot(3, 1, 2)
				% plot(x)

				% subplot(3, 1, 3)
				% image(scalogram, 'CDataMapping', 'scaled')

				% pause
				% continue

				% Slice the (30s long) data into different items
				n_time_steps = min(length(heartbeats), size(scalogram, 2));
				for begin_time=1:item_stride:n_time_steps-item_duration+1
					n_items = n_items + 1;

					end_time = begin_time + item_duration - 1;
					item_scalogram = scalogram(:,begin_time:end_time);
					item_heartbeat = heartbeats(begin_time:end_time);

					X(n_items, :, :) = item_scalogram';
					Y(n_items, :) = item_heartbeat';
				end
			end

			% Split into train/dev/test sets
			fprintf('Split in to train/dev/test sets...\n');
			m = n_items;
			m_train = floor(m * 0.7);
			m_dev = floor(m * 0.299);
			m_test = m - m_train - m_dev;
			
			perm = randperm(m);
			X = X(perm, :, :);
			Y = Y(perm, :);

			X_train = X(1:m_train, :, :);
			Y_train = Y(1:m_train, :);
			X_dev = X(m_train+1:m_train+m_dev, :, :);
			Y_dev = Y(m_train+1:m_train+m_dev, :);
			X_test = X(m_train+m_dev+1:end, :, :);
			Y_test = Y(m_train+m_dev+1:end, :);

			% Save to file
			save(strcat('handrate-dataset-pca-', num2str(n_ones), 'pts.mat'), ...
			 'X_train', 'Y_train', 'X_dev', 'Y_dev', 'X_test', 'Y_test')		
		end

		function [X, Y] = main(varargin)
			% main - Main function
			% 	
			% Syntax:  [] = Handrate.main(opt)
			%------------- BEGIN CODE --------------

			[X, Y] = Handrate.build_dataset();			
		end
   end
end