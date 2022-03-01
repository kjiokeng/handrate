classdef Handrate_combi_qkurt
% Handrate_combi_qkurt - A class which implements Handrate_combi_qkurt
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
   		function [ibi, hr] = compute_ibi_hr(x, varargin)
   			p = inputParser;
			addRequired(p, 'x');
			addParameter(p, 'resample_freq', 100);
			parse(p, x, varargin{:});

			x = p.Results.x;
			resample_freq = p.Results.resample_freq;

   			[pks, locs] = findpeaks(x, 'MinPeakDist', 0.55 * resample_freq);
			ibi = mean(diff(locs)) / resample_freq;

			hr = 60/ibi;
			ibi = ibi * 1000;
   		end

   		function [ibi, hr] = compute_ibi_hr_fft(x, varargin)
   			p = inputParser;
			addRequired(p, 'x');
			addParameter(p, 'resample_freq', 100);
			parse(p, x, varargin{:});

			x = p.Results.x;
			resample_freq = p.Results.resample_freq;

   			% [z, f] = Helper.to_frequential(x, 'period', 1/resample_freq);
   			[z, f] = Helper.to_frequential(x, 'period', 1/resample_freq);
			heart_activity_inds = f>0.7 & f<2;
			f = f(heart_activity_inds);
			z = z(heart_activity_inds);
			[m, m_ind] = max(z);
			hr = 60*f(m_ind);

			ibi = 1e4;
   		end

   		function [ibi, hr] = compute_ibi_hr_xcorr(x, varargin)
   			p = inputParser;
			addRequired(p, 'x');
			addParameter(p, 'resample_freq', 100);
			parse(p, x, varargin{:});

			x = p.Results.x;
			resample_freq = p.Results.resample_freq;

   			one_second = resample_freq;
			xt = x(one_second:2*one_second);
			[autocor, lags] = xcorr(xt, x);

			[ibi, hr] = Handrate_combi_qkurt.compute_ibi_hr(autocor, ...
						'resample_freq', resample_freq);
   		end

   		function [ibi, hr] = compute_ibi_hr_wavelet(x, varargin)
   			p = inputParser;
			addRequired(p, 'x');
			addParameter(p, 'resample_freq', 100);
			parse(p, x, varargin{:});

			x = p.Results.x;
			resample_freq = p.Results.resample_freq;

			[wt, f] = cwt(x, resample_freq, ...
					'VoicesPerOctave', 16, ...
				    'TimeBandWidth', 10);
			wt = max(abs(wt));

			[ibi, hr] = Handrate_combi_qkurt.compute_ibi_hr(wt, ...
						'resample_freq', resample_freq);
   		end

   		function [] = evaluate_config(directory, varargin)
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
			% filenames = filenames_train;

			% Read the data
			fprintf('Read the data...\n');
			[ecg_and_sensors_values, heart_rates, filenames] = Handrate.read_data_from_directory(directory, ...
									'metadata_file', metadata_file, ...
									'sensors_period', sensors_period, ...
									'ecg_period', ecg_period, ...
									'resample_freq', resample_freq, ...
									'skip_delta', skip_delta, ...
									'keep_filenames', filenames);

			% Actual building of the dataset
			fprintf('Actual evaluation...\n');
			ibi_gts = [];
			ibi_ffts = [];
			ibi_xcorrs = [];
			ibi_wavelets = [];

			hr_gts = [];
			hr_ffts = [];
			hr_xcorrs = [];
			hr_wavelets = [];

			n_files = length(ecg_and_sensors_values);
			% files_ids = 1:n_files;
			files_ids = 1:n_files-1;
			q_kurts = [];
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
				[~, x, x_noabs] = Handrate.sensors_to_scalogram(sensors_values, ...
								'freq', resample_freq, ...
								'signal_axis', signal_axis, ...
								'denoising_method', denoising_method);


				q_kurt = Helper.compute_q_kurt(x_noabs, 'period', 1/resample_freq)
				q_kurts(end+1) = q_kurt;
				continue

				% Ground truth
				[ibi_gt, hr_gt] = Handrate_combi_qkurt.compute_ibi_hr(heartbeats, ...
								'resample_freq', resample_freq);
				ibi_gts(end+1) = ibi_gt;
				hr_gts(end+1) = hr_gt;

				% FFT
				[ibi_fft, hr_fft] = Handrate_combi_qkurt.compute_ibi_hr_fft(x_noabs, ...
								'resample_freq', resample_freq);
				ibi_ffts(end+1) = ibi_fft;
				hr_ffts(end+1) = hr_fft;

				% xcorr
				[ibi_xcorr, hr_xcorr] = Handrate_combi_qkurt.compute_ibi_hr_xcorr(x_noabs, ...
								'resample_freq', resample_freq);
				ibi_xcorrs(end+1) = ibi_xcorr;
				hr_xcorrs(end+1) = hr_xcorr;

				% wavelet
				[ibi_wavelet, hr_wavelet] = Handrate_combi_qkurt.compute_ibi_hr_wavelet(x_noabs, ...
								'resample_freq', resample_freq);
				ibi_wavelets(end+1) = ibi_wavelet;
				hr_wavelets(end+1) = hr_wavelet;
			end

			q_kurts'

			% Statistics
			% fprintf("FFT\tXCORR\tWAVELETS\n");
			% ibi_err_ffts = abs(ibi_ffts - ibi_gts);
			% ibi_err_xcorrs = abs(ibi_xcorrs - ibi_gts);
			% ibi_err_wavelets = abs(ibi_wavelets - ibi_gts);
			% ibi_errs = [ibi_err_ffts', ibi_err_xcorrs', ibi_err_wavelets'];
			% mean_ibi_err = mean(ibi_errs)
			% median_ibi_err = median(ibi_errs)
			% perc90_ibi_err = prctile(ibi_errs, 90)

			% hr_err_ffts = abs(hr_ffts - hr_gts);
			% hr_err_xcorrs = abs(hr_xcorrs - hr_gts);
			% hr_err_wavelets = abs(hr_wavelets - hr_gts);
			% hr_errs = [hr_err_ffts', hr_err_xcorrs', hr_err_wavelets'];
			% mean_hr_err = mean(hr_errs)
			% median_hr_err = median(hr_errs)
			% perc90_hr_err = prctile(hr_errs, 90)

			% ibi_errs
			% hr_errs
			% ecdf(hr_errs(:,1))
			% hold on
			% ecdf(hr_errs(:,2))
			% ecdf(hr_errs(:,3))
			% legend("FFT", "XCORR", "WAVELETS")
		end

		function [] = evaluate()
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
			signal_axis = 'pca3';
			denoising_method = 'wavelet';

			Handrate_combi_qkurt.evaluate_config(directory, ...
										  'metadata_file', metadata_file, ...
										  'item_duration_seconds', item_duration_seconds, ...
										  'n_ones', n_ones, ...
										  'signal_axis', signal_axis, ...
										  'resample_freq', resample_freq, ...
										  'sensors_period', sensors_period, ...
										  'ecg_period', ecg_period, ...
										  'skip_delta', skip_delta, ...
										  'denoising_method', denoising_method);
		end

		function [X, Y] = main(varargin)
			% main - Main function
			% 	
			% Syntax:  [] = Handrate_combi_qkurt.main(opt)
			%------------- BEGIN CODE --------------

			Handrate_combi_qkurt.evaluate();			
		end
   end
end