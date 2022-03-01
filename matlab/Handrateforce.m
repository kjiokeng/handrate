classdef Handrateforce
% Handrateforce - A class which implements Handrate brute force method
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
      	
   end

   methods (Static)
   		function [out] = preprocessing_remove_noise(in, varargin)
			
			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'in');
			addParameter(p, 'method', 'none');
			addParameter(p, 'period', 0.0194);
			parse(p, in, varargin{:});

			in = p.Results.in;
			method = p.Results.method;
			period = p.Results.period;
			sampling_freq = 1/period;

			% Useful variables
			n_users = size(in, 1);
			n_dims = size(in, 2);
			n_points = size(in, 3);

			% Actual computation
			out = in;
			switch method
				case 'none'
					out = in;
				case 'smooth'
					n_points = 5;
					y1 = movmean(in(1, 1, :), n_points);
					out = zeros(n_users, n_dims, length(y1));
					for user=1:n_users
						for dim=1:n_dims
							x = in(user, dim, :);
							x = squeeze(x);
							y = movmean(x, n_points);
							out(user, dim, :) = y;
						end
					end
				case 'bandpass'
					for user=1:n_users
						for dim=1:n_dims
							x = in(user, dim, :);
							x = squeeze(x);
							y = bandpass(x, [0.5 2.1], sampling_freq, 'StopbandAttenuation', 60);
							out(user, dim, :) = y;
						end
					end
				case 'wavelet'
					for user=1:n_users
						for dim=1:n_dims
							x = in(user, dim, :);
							x = squeeze(x);
							% y = Helper.filter_noise(x, 'method', 'wavelet');
							y = wdenoise(x,7, ...
								    'Wavelet', 'sym4', ...
								    'DenoisingMethod', 'Bayes', ...
								    'ThresholdRule', 'Median', ...
								    'NoiseEstimate', 'LevelIndependent');

							out(user, dim, :) = y;
						end
					end
			end
		end

		function [out] = preprocessing_combination(in, varargin)
			
			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'in');
			addParameter(p, 'method', 'none');
			parse(p, in, varargin{:});

			in = p.Results.in;
			method = p.Results.method;

			% Useful variables
			n_users = size(in, 1);
			n_dims = size(in, 2);
			n_points = size(in, 3);

			% Actual computation
			out = in;
			switch method
				case 'none'
					out = in;
				case 'pca'
					for user=1:n_users
						x = in(user, :, :);
						x = squeeze(x);
						y = Helper.pca(x');
						out(user,:,:) = y';
					end
				case 'ica'
					for user=1:n_users
						x = in(user, :, :);
						x = squeeze(x);
						y = Helper.ica(x');
						out(user,:,:) = y';
					end
			end
		end

		function [out] = preprocessing_selection(in, varargin)
			
			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'in');
			addParameter(p, 'method', 'xyz');
			parse(p, in, varargin{:});

			in = p.Results.in;
			method = p.Results.method;

			% Actual computation
			dims = 1:3;
			switch method
				case 'x'
					dims = 1;
				case 'y'
					dims = 2;
				case 'z'
					dims = 3;
				case 'xy'
					dims = [1, 2];
				case 'yz'
					dims = [2, 3];
				case 'xz'
					dims = [1, 3];
				case 'xyz'
					dims = 1:3;
			end

			out = in(:, dims, :);
		end

		function [out] = preprocessing_transformation(in, varargin)
			
			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'in');
			addParameter(p, 'method', 'none');
			parse(p, in, varargin{:});

			in = p.Results.in;
			method = p.Results.method;

			% Actual computation
			out = in;
			switch method
				case 'none'
					out = in;
				case 'abs'
					out = abs(in);
			end
		end

		function [out] = processing_core(in, varargin)
			
			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'in');
			addParameter(p, 'method', 'fft');
			addParameter(p, 'period', 0.0194);
			parse(p, in, varargin{:});

			in = p.Results.in;
			method = p.Results.method;
			period = p.Results.period;

			% Useful variables
			n_users = size(in, 1);
			n_dims = size(in, 2);
			n_points = size(in, 3);

			% Actual computation
			out = zeros(n_users, n_dims, 1);
			for user=1:n_users
				for dim=1:n_dims
					x = in(user, dim, :);
					x = squeeze(x);
					y = Helper.compute_heart_rate(x, 'period', period, 'method', method);
					out(user, dim, 1) = y;
				end
			end
		end

		function [out] = postprocessing_combination(in, varargin)
			
			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'in');
			parse(p, in, varargin{:});

			in = p.Results.in;

			% Useful variables
			n_users = size(in, 1);
			n_dims = size(in, 2);

			% Actual computation
			out = zeros(n_users, 1);
			for user=1:n_users
				x = in(user, :, :);
				x = squeeze(x);
				x = reshape(x, numel(x), 1);
				out(user, 1) = nanmean(x);
				% out(user, 2) = median(x);
				% out(user, 3) = harmmean(x);
			end
		end

		function [out] = read_values_from_dir(directory, varargin)
			
			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'directory');
			addParameter(p, 'period', 0.0194);
			addParameter(p, 'start_at', 0);
			addParameter(p, 'duration', Inf);
			parse(p, directory, varargin{:});

			directory = p.Results.directory;
			period = p.Results.period;
			start_at = p.Results.start_at;
			duration = p.Results.duration;

			heartbeat_files = dir(directory);
			heartbeat_files = heartbeat_files(3:end);
			num_files = length(heartbeat_files);

			filenames = {};
			values = {};
			n_points = Inf;
			for k=1:num_files
				filenames{k} = strcat(heartbeat_files(k).folder, '/', heartbeat_files(k).name);
				values{k} = extract_values(filenames{k}, 'start_at', start_at, 'duration', duration, ...
				'period', period, 'sensor', 'ACC');
				n_points = min(n_points, size(values{k}, 1));
			end

			out = zeros(num_files, 3, n_points);
			for k=1:num_files
				tmp = values{k};
				out(k, :, :) = tmp(1:n_points,2:4)';
			end
		end

		function [out, gtt] = read_values_from_metadata_file(directory, varargin)
			
			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'directory');
			addParameter(p, 'metadata_file', 'metadata.csv');
			addParameter(p, 'sensors_dir', 'sensors');
			addParameter(p, 'ecg_file_name', 'ecg_file_name');
			addParameter(p, 'period', 0.0194);
			addParameter(p, 'start_at', 0);
			addParameter(p, 'duration', Inf);
			parse(p, directory, varargin{:});

			directory = p.Results.directory;
			metadata_file = p.Results.metadata_file;
			sensors_dir = p.Results.sensors_dir;
			ecg_file_name = p.Results.ecg_file_name;
			period = p.Results.period;
			start_at = p.Results.start_at;
			duration = p.Results.duration;
			metadata_file = strcat(directory, '/', metadata_file);
			sensors_dir = strcat(directory, '/', sensors_dir);

			% Useful variables
			ecg_file_id_column_name = 'ecg_file_id';
			time_length_column_name = 'time_length';
			check_time_column_name = 'check_time';
			heart_rate_column_name = 'heart_rate';
			sensors_file_column_name = 'sensors_file';
			values = {};
			heart_rates = {};
			num_files = 0;
			n_points = Inf;

			% Actual computation
			T = readtable(metadata_file, 'DatetimeType', 'text');
			[n_rows, n_cols] = size(T);

			for k=1:n_rows
				ecg_file_id = table2array(T(k, ecg_file_id_column_name));
				time_length = table2array(T(k, time_length_column_name));
				heart_rate = table2array(T(k, heart_rate_column_name));
				check_time = table2array(T(k, check_time_column_name));
				check_time = check_time{1};
				sensors_file = table2cell(T(k, sensors_file_column_name));
				sensors_file = sensors_file{1};

				if isempty(sensors_file)
					continue
				end
				num_files = num_files + 1;

				tmp = strsplit(check_time, ' ');
				ecg_date = tmp{1};
				ecg_file = strcat(directory, '/', ecg_date, '/', num2str(ecg_file_id), '/', ecg_file_name);
				sensors_file = strcat(sensors_dir, '/', sensors_file)

				heart_rates{num_files} = heart_rate;
				values{num_files} = extract_values(sensors_file, 'start_at', start_at, 'duration', duration, ...
				'period', period, 'sensor', 'ACC');
				n_points = min(n_points, size(values{num_files}, 1));
			end

			out = zeros(num_files, 3, n_points);
			gtt = zeros(num_files, 1);
			num_files
			for k=1:num_files
				tmp = values{k};
				out(k, :, :) = tmp(1:n_points,2:4)';
				gtt(k) = heart_rates{k};
			end

		end

		function [accuracy, results, expe_ids] = run(varargin)
			% run - Function which actually runs the brute force process
			% 	
			% Syntax:  [] = Handrateforce.run(opt)
			%------------- BEGIN CODE --------------


			% Parse the input parameters
			p = inputParser;
			addParameter(p, 'directory', '../../measures/data/');
			addParameter(p, 'metadata_file', 'metadata.csv');
			addParameter(p, 'output_dir', './results/hand/pixel/hr/brute-force/');
			addParameter(p, 'start_at', 10);
			addParameter(p, 'duration', 30);
			addParameter(p, 'period', 0.0194);
			parse(p, varargin{:});

			directory = p.Results.directory;
			metadata_file = p.Results.metadata_file;
			output_dir = p.Results.output_dir;
			start_at = p.Results.start_at;
			duration = p.Results.duration;
			period = p.Results.period;

			
			fprintf("---------- Reading the files ----------\n")
			% directory = '../../measures/hand/pixel/hr/25-02-2020/';
			% gtt = [71, 63, 71, 56, 83, 54, 59, 61, 67, 72, 70, 52]';
			% in = Handrateforce.read_values_from_dir(directory, ...
				% 'period', period, 'start_at', start_at, 'duration', duration);

			[in, gtt] = Handrateforce.read_values_from_metadata_file(directory, ...
				'metadata_file', metadata_file, ...
				'period', period, 'start_at', start_at, 'duration', duration);

			fprintf("---------- Actual computation ----------\n")
			noise_removal_methods = {'none', 'smooth', 'bandpass', 'wavelet'};
			transformation_methods = {'none', 'abs'};
			combination_methods = {'none', 'pca', 'ica'};
			selection_methods = {'x', 'y', 'z', 'xy', 'yz', 'xz', 'xyz'};
			noise_removal2_methods = {'none', 'smooth', 'bandpass', 'wavelet'};
			transformation2_methods = {'none', 'abs'};
			core_methods = {'fft', 'corr', 'wavelet'};

			expe_ids = {};
			results = {};
			config_id = 0;
			n_configs = length(noise_removal_methods) * length(transformation_methods) ...
								* length(combination_methods) * length(selection_methods) ...
								* length(noise_removal2_methods) * length(transformation2_methods) ...
								* length(core_methods);

			for noise_removal_method_id=1:length(noise_removal_methods)
				noise_removal_method = noise_removal_methods{noise_removal_method_id};
				expe_id_noise_removal = noise_removal_method;
				out_noise_removal = Handrateforce.preprocessing_remove_noise(in, 'method', noise_removal_method, ...
					'period', period);

				for transformation_method_id=1:length(transformation_methods)
					transformation_method = transformation_methods{transformation_method_id};
					expe_id_transformation = strcat(expe_id_noise_removal, ',', transformation_method);
					out_transformation = Handrateforce.preprocessing_transformation(out_noise_removal, 'method', transformation_method);

					for combination_method_id=1:length(combination_methods)
						combination_method = combination_methods{combination_method_id};
						expe_id_combination = strcat(expe_id_transformation, ',', combination_method);
						out_combination = Handrateforce.preprocessing_combination(out_transformation, 'method', combination_method);

						for selection_method_id=1:length(selection_methods)
							selection_method = selection_methods{selection_method_id};
							expe_id_selection = strcat(expe_id_combination, ',', selection_method);
							out_selection = Handrateforce.preprocessing_selection(out_combination, 'method', selection_method);

							for noise_removal2_method_id=1:length(noise_removal2_methods)
								noise_removal2_method = noise_removal2_methods{noise_removal2_method_id};
								expe_id_noise_removal2 = strcat(expe_id_selection, ',', noise_removal2_method);
								out_noise_removal2 = Handrateforce.preprocessing_remove_noise(out_selection, 'method', noise_removal2_method);

								for transformation2_method_id=1:length(transformation2_methods)
									transformation2_method = transformation2_methods{transformation2_method_id};
									expe_id_transformation2 = strcat(expe_id_noise_removal2, ',', transformation2_method);
									out_transformation2 = Handrateforce.preprocessing_transformation(out_selection, 'method', transformation2_method);

									for core_method_id=1:length(core_methods)
										config_id = config_id + 1;

										core_method = core_methods{core_method_id};
										expe_id_core = strcat(expe_id_transformation2, ',', core_method);
										out_core = Handrateforce.processing_core(out_transformation2, 'method', core_method);

										out_postprocessing_combination = Handrateforce.postprocessing_combination(out_core);

										% Save the results
										expe_id = expe_id_core;
										out = out_postprocessing_combination;
										expe_ids{config_id} = expe_id;
										results{config_id} = out;

										% Clear stateless temporary variables
										clear x y

										fprintf('--- Configuration %d / %d: %.1f%%: %s\n', config_id, n_configs, 100*config_id/n_configs, expe_id);
									end
								end
							end
						end
					end
				end
			end

			% Compute the accuracy
			accuracy = zeros(n_configs, 7);
			for config_id=1:n_configs
				out = results{config_id};
				err = abs(out-gtt);
				err_min = nanmin(err);
				err_mean = nanmean(err);
				err_median = nanmedian(err);
				err_75prctile = prctile(err, 75);
				err_90prctile = prctile(err, 90);
				err_max = nanmax(err);
				rmse = sqrt(nanmean(err .* err));

				out = [out, gtt, err];
				results{config_id} = out;

				accuracy(config_id, :) = [err_min, err_mean, err_median, err_75prctile, err_90prctile, err_max, rmse];
			end

			% Save the results to a file
			metadata_file = strsplit(metadata_file, '/');
			metadata_file = metadata_file{end};
			metadata_file = strrep(metadata_file, '.csv', '');
			filename = sprintf('bruteforce-%s-%s', metadata_file, datetime());
			save(filename, 'accuracy', 'results', 'expe_ids')

			% Print and save as CSV
			filename = strcat(output_dir, '/', filename,'.csv');
			fid = fopen(filename, 'w');
			tmp_str = ['config_id,noise_removal1,transformation1,combination,selection,noise_removal2,' ...
				'transformation2,core,err_min,err_mean,err_median,err_75prctile,err_90prctile,err_max,rmse\n'];
			fprintf(fid, tmp_str);
			fprintf(tmp_str);
			for config_id=1:n_configs
				tmp_str = sprintf('%d,%s,%f,%f,%f,%f,%f,%f,%f\n', config_id, expe_ids{config_id}, ...
					accuracy(config_id, 1), ...
					accuracy(config_id, 2), accuracy(config_id, 3), accuracy(config_id, 4), ...
					accuracy(config_id, 5), accuracy(config_id, 6), accuracy(config_id, 7));
				fprintf(fid, tmp_str);
				fprintf(tmp_str);
			end
			fclose(fid);
		end

		function [accuracy, results, expe_ids] = recompute_stats(varargin)
			% recompute_stats - recompute the stats based on saved results
			% 	
			% Syntax:  [] = Handrateforce.recompute_stats(opt)
			%------------- BEGIN CODE --------------


			% Parse the input parameters
			p = inputParser;
			addParameter(p, 'directory', '../../measures/data/');
			addParameter(p, 'metadata_file', 'metadata.csv');
			addParameter(p, 'output_dir', './results/hand/pixel/hr/brute-force/');
			addParameter(p, 'start_at', 10);
			addParameter(p, 'duration', 30);
			addParameter(p, 'period', 0.0194);
			parse(p, varargin{:});

			directory = p.Results.directory;
			metadata_file = p.Results.metadata_file;
			output_dir = p.Results.output_dir;
			start_at = p.Results.start_at;
			duration = p.Results.duration;
			period = p.Results.period;

			
			fprintf("---------- Loading the results ----------\n")
			metadata_file = strsplit(metadata_file, '/');
			metadata_file = metadata_file{end};
			metadata_file = strrep(metadata_file, '.csv', '');
			search_query = strcat('bruteforce-', metadata_file, '-1*.mat')
			res_file = dir(search_query);
			res_file = res_file(1)
			res_file = res_file.name
			load(res_file)

			fprintf("---------- Actual recomputation ----------\n")
			n_configs = length(results);
			for config_id=1:n_configs
				out = results{config_id};
				err = out(:,3);
				gtt = out(:,2);
				out = out(:,1);

				err_min = nanmin(err);
				err_mean = nanmean(err);
				err_median = nanmedian(err);
				err_75prctile = prctile(err, 75);
				err_90prctile = prctile(err, 90);
				err_max = nanmax(err);
				rmse = sqrt(nanmean(err .* err));

				out = [out, gtt, err];
				results{config_id} = out;

				accuracy(config_id, :) = [err_min, err_mean, err_median, err_75prctile, err_90prctile, err_max, rmse];
			end

			% Save the results to a file
			filename = sprintf('recomputed/bruteforce-%s-%s', metadata_file, datetime());
			save(filename, 'accuracy', 'results', 'expe_ids')

			% Print and save as CSV
			filename = strcat(output_dir, '/', filename,'.csv');
			fid = fopen(filename, 'w');
			tmp_str = ['config_id,noise_removal1,transformation1,combination,selection,noise_removal2,' ...
				'transformation2,core,err_min,err_mean,err_median,err_75prctile,err_90prctile,err_max,rmse\n'];
			fprintf(fid, tmp_str);
			fprintf(tmp_str);
			for config_id=1:n_configs
				tmp_str = sprintf('%d,%s,%f,%f,%f,%f,%f,%f,%f\n', config_id, expe_ids{config_id}, ...
					accuracy(config_id, 1), ...
					accuracy(config_id, 2), accuracy(config_id, 3), accuracy(config_id, 4), ...
					accuracy(config_id, 5), accuracy(config_id, 6), accuracy(config_id, 7));
				fprintf(fid, tmp_str);
				fprintf(tmp_str);
			end
			fclose(fid);
		end

		function [accuracy, results, expe_ids] = main(varargin)
			% Execution parameters
			directory = '../../measures/data/';
			metadata_file = 'metadata.csv';
			output_dir = './results/hand/pixel/hr/brute-force/';
			start_at = 10;
			duration = 30;
			period = 0.0194; % Pixel 2
			gyr_period = 0.0025; % Pixel 2

			% metadata_files = {
			% 	'metadata.csv'
			% };

			metadata_files = {
				'metadata_files/metadata-user1.csv',
				'metadata_files/metadata-user2.csv',
				'metadata_files/metadata-user3.csv',
				'metadata_files/metadata-user4.csv',
				'metadata_files/metadata-user5.csv',
				'metadata_files/metadata-user6.csv',
				'metadata_files/metadata-user7.csv',
				'metadata_files/metadata-user8.csv',
				'metadata_files/metadata-user9.csv',
				'metadata_files/metadata-user.csv',
				'metadata_files/metadata-demo.csv',
				'metadata_files/metadata-userdemo.csv',
				'metadata.csv'
			};

			for k=1:length(metadata_files)
			% parfor k=1:length(metadata_files)
				metadata_file = metadata_files{k};
				% [accuracy, results, expe_ids] = Handrateforce.run('directory', directory, ...
				% 								'metadata_file', metadata_file, ...
				% 								'output_dir', output_dir, ...
				% 								'start_at', start_at, ...
				% 								'duration', duration, ...
				% 								'period', period);

				[accuracy, results, expe_ids] = Handrateforce.recompute_stats('directory', directory, ...
												'metadata_file', metadata_file, ...
												'output_dir', output_dir, ...
												'start_at', start_at, ...
												'duration', duration, ...
												'period', period);

			end
		end
   end
end