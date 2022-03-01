classdef Handrate_hrv
% Handrate_hrv - A class which implements Handrate_hrv
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

   		function [pred_heartbeats, gt_heartbeats] = read_heartbeats_from_files(predictions_directory, n_files, varargin)
   			p = inputParser;
			addRequired(p, 'predictions_directory');
			addRequired(p, 'n_files');
			parse(p, predictions_directory, n_files, varargin{:});

			predictions_directory = p.Results.predictions_directory;
			n_files = p.Results.n_files;

			pred_heartbeats = {};
			gt_heartbeats = {};
			for k=1:n_files
				if k==5
					continue
				end

				pred_file = sprintf('%s/pred%d.txt', predictions_directory, k-1)
				gt_file = sprintf('%s/gt%d.txt', predictions_directory, k-1);

				file_id = fopen(pred_file, 'r');
				preds = fscanf(file_id, '%f');
				fclose(file_id);

				file_id = fopen(gt_file, 'r');
				gts = fscanf(file_id, '%f');
				fclose(file_id);

				pred_heartbeats{end+1} = preds;
				gt_heartbeats{end+1} = gts;
			end
   		end

   		function [hrv_features] = compute_hrv_features(x, varargin)
   			p = inputParser;
			addRequired(p, 'x');
			addParameter(p, 'resample_freq', 100);
			addParameter(p, 'min_peak_distance', 0.4);
			addParameter(p, 'min_peak_height', 0.25);
			parse(p, x, varargin{:});

			x = p.Results.x;
			resample_freq = p.Results.resample_freq;
			min_peak_distance = p.Results.min_peak_distance;
			min_peak_height = p.Results.min_peak_height;


			hrv_features = [];
			[pks, idx] = findpeaks(x, 'MinPeakDistance', min_peak_distance*resample_freq, ...
									'MinPeakHeight', min_peak_height);
			% plot(idx, pks, 'ro')
			ibis = 1000 * diff(idx) / resample_freq;
			ibis_diff = diff(ibis);
			% [pks, idx]
			% plot(x)
			% pause

			% HR
			hr = 1000 * 60 / mean(ibis);
			hrv_features(end+1) = hr;

			% RMSSD
			rmssd = ibis_diff .* ibis_diff;
			rmssd = sqrt(mean(rmssd));
			hrv_features(end+1) = rmssd;

			% SDNN
			sdnn = std(ibis);
			hrv_features(end+1) = sdnn;

			% MRRI
			mrri = mean(ibis);
			hrv_features(end+1) = mrri;

			% SDSD
			sdsd = std(ibis_diff);
			hrv_features(end+1) = sdsd;

			% NN20
			nn20 = sum(ibis_diff>20);
			hrv_features(end+1) = nn20;

			% pNN20
			pnn20 = 100 * nn20 / length(ibis);
			hrv_features(end+1) = pnn20;

			% SD1
			sd1 = std(ibis) * sqrt(2) / 2;
			hrv_features(end+1) = sd1;

			% SD2
			sd2 = 2 * std(ibis .* ibis) - 0.5 * std(ibis_diff .* ibis_diff);
			sd2 = sqrt(sd2);
			hrv_features(end+1) = sd2;
   		end

   		function [hrv_features_matrix] = compute_hrv_features_batch(directory, varargin)
   			% Evaluate a given configuration (given by its different parameters values)

   			p = inputParser;
			addRequired(p, 'directory');
			addParameter(p, 'metadata_file', 'metadata.csv');
			addParameter(p, 'item_duration_seconds', 3);
			addParameter(p, 'n_ones', 5);
			addParameter(p, 'signal_axis', 'pca');
			addParameter(p, 'resample_freq', 100);
			addParameter(p, 'sensors_period', 0.0194);
			addParameter(p, 'ecg_period', 1/250);
			addParameter(p, 'skip_delta', 0);
			addParameter(p, 'denoising_method', 'wavelet');
			parse(p, directory, varargin{:});

			directory = p.Results.directory;
			metadata_file = p.Results.metadata_file;
			item_duration_seconds = p.Results.item_duration_seconds;
			n_ones = p.Results.n_ones;
			signal_axis = p.Results.signal_axis;   			
			resample_freq = p.Results.resample_freq;   			
			sensors_period = p.Results.sensors_period;   			
			ecg_period = p.Results.ecg_period;   			
			skip_delta = p.Results.skip_delta;   			
			denoising_method = p.Results.denoising_method;   			


   			dataset_filename = sprintf('datasets/handrate-dataset-raw-%ds-%dones-%s.mat', ...
				item_duration_seconds, n_ones, signal_axis);
			load(dataset_filename);
			filenames = filenames_test;

			% Read the data
			fprintf('Read the data...\n');
			[ecg_and_sensors_values, heart_rates, filenames] = Handrate.read_data_from_directory(directory, ...
									'metadata_file', metadata_file, ...
									'sensors_period', sensors_period, ...
									'ecg_period', ecg_period, ...
									'resample_freq', resample_freq, ...
									'skip_delta', skip_delta, ...
									'keep_filenames', filenames_test);

			hrv_features_matrix = [];

			n_files = length(ecg_and_sensors_values);
			files_ids = 1:n_files;
			% files_ids(5) = []
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

				% % Convert the sensors values to time-frequency representation
				% [~, x, x_noabs] = Handrate.sensors_to_scalogram(sensors_values, ...
				% 				'freq', resample_freq, ...
				% 				'signal_axis', signal_axis, ...
				% 				'denoising_method', denoising_method);

				hrv_features = Handrate_hrv.compute_hrv_features(heartbeats);
				hrv_features_matrix(k, :) = hrv_features;
				
			end
		end

		function [pred_hrv_features_matrix, gt_hrv_features_matrix] = compute_hrv_features_from_files(predictions_directory, n_files, varargin)
   			% Evaluate a given configuration (given by its different parameters values)

   			p = inputParser;
			addRequired(p, 'predictions_directory');
			addRequired(p, 'n_files');
			addParameter(p, 'resample_freq', 100);
			parse(p, predictions_directory, n_files, varargin{:});

			predictions_directory = p.Results.predictions_directory;
			n_files = p.Results.n_files;
			resample_freq = p.Results.resample_freq;  	


			gt_hrv_features_matrix = [];
			pred_hrv_features_matrix = [];
			[pred_heartbeats, gt_heartbeats] = Handrate_hrv.read_heartbeats_from_files(predictions_directory, n_files);

			files_ids = 1:length(pred_heartbeats);
			for idx=1:length(files_ids)
				k = files_ids(idx);
				fprintf('--- Processing file %d / %d: %.1f%%\n', k, n_files, k*100/n_files);
				
				preds = pred_heartbeats{k};
				gts = gt_heartbeats{k};

				% close all
				% fig = figure('units','normalized','outerposition',[0 1 1 0.5]);
				% plot(gts, 'b')
				% hold on
				% plot(preds, 'r')

				pred_hrv_features = Handrate_hrv.compute_hrv_features(preds, ...
										'resample_freq', resample_freq, ...
										'min_peak_distance', 0.35);
				gt_hrv_features = Handrate_hrv.compute_hrv_features(gts, ...
										'resample_freq', resample_freq, ...
										'min_peak_distance', 0.4);

				gt_hr = gt_hrv_features(1);
				pred_hr = pred_hrv_features(1);
				if gt_hr < 40
					continue
				end

				% if abs(pred_hr-gt_hr) > 15
				% 	continue
				% end

				pred_hrv_features_matrix(end+1, :) = pred_hrv_features;
				gt_hrv_features_matrix(end+1, :) = gt_hrv_features;
			end

			figure
			hr_errs = abs(pred_hrv_features_matrix(:,1)-gt_hrv_features_matrix(:, 1))
			hr_errs_stats = [median(hr_errs), mean(hr_errs), prctile(hr_errs, 90)]
			ecdf(hr_errs)

			figure
			scatter(gt_hrv_features_matrix(:, 1), pred_hrv_features_matrix(:,1))
			hold on
			plot(50:90, 50:90)
			hold off
		end

		function [X, Y, correlations] = compute()
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
			signal_axis = 'pca';
			denoising_method = 'wavelet';

			X = [];
			Y = [];

			% X = Handrate_hrv.compute_hrv_features_batch(directory, ...
			% 							  'metadata_file', metadata_file, ...
			% 							  'item_duration_seconds', item_duration_seconds, ...
			% 							  'n_ones', n_ones, ...
			% 							  'signal_axis', signal_axis, ...
			% 							  'resample_freq', resample_freq, ...
			% 							  'sensors_period', sensors_period, ...
			% 							  'ecg_period', ecg_period, ...
			% 							  'skip_delta', skip_delta, ...
			% 							  'denoising_method', denoising_method);



			% predictions_directory = '../python/predictions/';
			% predictions_directory = '../python/predictions/handrate.rnn.conv.3s.5ones.128units.h5/';
			% predictions_directory = sprintf('../python/predictions/handrate.rnn.conv.3s.%dones.128units.h5/', n_ones);
			predictions_directory = sprintf('../python/predictions/handrate.rnn.3s.5ones.256units.overfit.h5/', n_ones);
			% predictions_directory = '../python/predictions/handrate.rnn.conv.3s.5ones.128units.overfit.h5/';
			n_files = 131;
			resample_freq = 100;
			[X, Y] = Handrate_hrv.compute_hrv_features_from_files(predictions_directory, ...
										  n_files, ...
										  'resample_freq', resample_freq);

			correlations = [];
			for k=1:size(X, 2)
				cor = corrcoef(X(:,k), Y(:, k));
				cor = cor(1, 2);
				correlations(end+1) = cor;
			end
			correlations

		end

		function [X, Y] = main(varargin)
			% main - Main function
			% 	
			% Syntax:  [] = Handrate_hrv.main(opt)
			%------------- BEGIN CODE --------------

			[X, Y] = Handrate_hrv.compute();			
		end
   end
end