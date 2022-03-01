classdef Ubicomp18
% Ubicomp18 - A class which implements Ubicomp'18 paper on heartbeat-based authentication
% Reference: Unlock with Your Heart: Heartbeat-based Authentication on Commercial Mobile Phones
% Link: https://dl.acm.org/citation.cfm?id=3264950
%
% Author: Kevin Jiokeng
% email: kevin.jiokeng@enseeiht.fr
% November 2019; Last revision: 20-November-2019
%------------- BEGIN CODE --------------

   % Constants
   properties (Constant)
      normalized_cycle_length = 256;
      % normalized_features_length = 263;
      normalized_features_length = 335;
   end

   properties
      dir_path = '.';
      filenames = {};
      user_ids = {};
      cycles = {};
      fine_templates = {};
      features = {};
      svm_models = {};
   end

   methods
      	function obj = build(obj)
	      	% Initializes an object (compute filenames, user_ids and extract cycles)

	        heartbeat_files = dir(obj.dir_path);
			heartbeat_files = heartbeat_files(3:end);
			num_files = length(heartbeat_files);

			obj.filenames = {};
			obj.user_ids = {};
			obj.cycles = {};
			obj.fine_templates = {};
			for k=1:num_files
				obj.filenames{k} = strcat(heartbeat_files(k).folder, '/', heartbeat_files(k).name);

				% Extract the user id from the filename
				tmp = strsplit(obj.filenames{k}, '/');
				tmp = tmp{end};
				tmp = strsplit(tmp, '-');
				obj.user_ids{k} = tmp{1};

				% Extract cycles from the current file
				obj.fine_templates{k} = obj.generate_fine_template(obj.filenames{k});
				obj.cycles{k} = obj.extract_cycles(obj.filenames{k}, obj.fine_templates{k});
			end

			% Feature extraction
			for k=1:length(obj.user_ids)
				cycles_k = obj.cycles{k};

				n_cycles = size(cycles_k, 1);
				features_k = zeros(n_cycles, Ubicomp18.normalized_features_length);

				for l=1:n_cycles
					cycle = cycles_k(l,:);
					if sum(cycle)==0
						continue
					end

					features_k(l,:) = Ubicomp18.extract_features(cycle);
				end
				obj.features{k} = features_k;
			end
      	end

      	function obj = train(obj, varargin)
      		% Train the SVM classifiers
			% 	
			% Inputs:
			%		obj - A reference to the current object
			%  		opt - Optional name-value parameters
			%			cross_validate - Boolean. If true, perform cross validation on each built model. Default: true
			%------------- BEGIN CODE --------------

      		% Parse the input parameters
			p = inputParser;
			addRequired(p, 'obj');
			addParameter(p, 'cross_validate', true);
			parse(p, obj, varargin{:});

			obj = p.Results.obj;
			cross_validate = p.Results.cross_validate;

			addpath('./libsvm/matlab')

	      	n_users = length(obj.user_ids);
	      	obj.svm_models = {};
	      	generalization_error = [];
	      	for k=1:n_users
	      		features_k = obj.features{k};
	      		n_cycles = size(features_k, 1);

	      		X = zeros(2*n_cycles, Ubicomp18.normalized_features_length); %For positive and negative samples
	      		Y = zeros(2*n_cycles, 1);
	      		
	      		% Positive examples
	      		X(1:n_cycles,:) = features_k;
	      		Y(1:n_cycles) = 1;

	      		% Negative examples
	      		other_users = mod(k:k+n_users-2, n_users)+1;
	      		for l=1:n_cycles
	      			% Pick n_cycles users randomly (with possible repetitions)
	      			other_users_to_select = randsample(other_users, n_cycles, true);

	      			% For each user, pick a cycle
	      			for m=1:length(other_users_to_select)
	      				user_idx = other_users_to_select(m);
	      				cycle_idx = randsample(size(obj.features{user_idx}, 1),1);
	      				cycle = obj.features{user_idx}(cycle_idx,:);
	      			end
	      			
	      			X(n_cycles+l,:) = cycle;
	      		end

	      		% Random permutation of the cycles
	      		perm = randperm(2*n_cycles);
	      		X = X(perm,:);
	      		Y = Y(perm);
	      		% pause

	      		% Actual training of the SVM model
	      		if false
		      		kf = 'polynomial';
		      		standardize = true;
			      	% SVMModel = fitcsvm(X, Y, 'KernelFunction', kf, 'Standardize', standardize);
			      	SVMModel = fitckernel(X, Y, 'CrossVal', 'on', ...
			      		'ScoreTransform', 'logit');
			    	obj.svm_models{k} = compact(SVMModel); % Reduces the size (in bytes) of the model

			    	% Cross validation
			    	if cross_validate
				    	CVSVMModel = crossval(SVMModel);
				    	generalization_error(k) = kfoldLoss(CVSVMModel);
				    end
			    end

			    % libsvm
			    if true
			    	obj.svm_models{k} = svmtrain(Y, X, '-s 1 -b 1 -n 0.3 -g 0.5');
		    	end
	      	end

	      	% fprintf("\n--- CROSS VALIDATION RESULTS ---\n")
	      	% for k=1:length(generalization_error)
	      	% 	fprintf("--- User: %s, Generalization error: %.2f%%\n", obj.user_ids{k}, generalization_error(k)*100);
	      	% end
	      	% fprintf("--- AVERAGE GENERALIZATION ERROR: %.2f%%\n", mean(generalization_error)*100)
      	end

      	function save(obj)
      		% Save the current object to a file

	      	tmp = strsplit(obj.dir_path, '/');
	      	target_filename = tmp{end};
	      	save(target_filename, 'obj')
      	end
   end

   methods (Static)
   		function [ct, ctb, cte] = extract_coarse_template(values, varargin)
			% extract_coarse_template - Extract coarse template
			% 	
			% Syntax:  [ct, ctb, cte] = Ubicomp18.extract_coarse_template(values, opt)
			%
			% Inputs:
			%		values - The time domain values
			%  		opt - Optional name-value parameters
			%			period - The period of time domain values (time interval between two consecutive values)
			%      
			% Outputs:
			%  	 	ct  - The coarse template values
			%  	 	ctb - The coarse template begin index
			%  	 	cte - The coarse template end index
			%------------- BEGIN CODE --------------

			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'values');
			addParameter(p, 'period', 0.005);
			parse(p, values, varargin{:});

			values = p.Results.values;
			period = p.Results.period;

			
			% Actual computation
			two_secs_inds = 1:floor(2/period); % Indices for a 2-seconds long slice
			v = values(two_secs_inds,3); % Take only the y axis as actual values
			t = values(two_secs_inds,1); % Timestamps
			% v = v / max(abs(v));

			% Peak finding and pruning algorithm
			pks_start = ceil(0.5/period); % Start looking for peaks at time 0.5s
			t_threshold = 200 * 1e-3; % 200ms in ns
			d_threshold = t_threshold/period;
			[pks, locs] = findpeaks(v(pks_start:end), 'SortStr','descend', ...
			'MinPeakDistance', d_threshold, 'MinPeakHeight', 0.1);
			locs = locs + pks_start - 1;
			
			candidate_peaks = pks;
			candidate_peaks_locs = locs;
			candidate_peaks_timestamps = t(locs);

			% Choosing the two closest peaks
			n_cand = length(candidate_peaks);
			ind1 = 1;
			ind2 = 1;
			min_d = Inf;
			for k=1:n_cand
				for l=k+1:n_cand
					d = abs(candidate_peaks_timestamps(l) - candidate_peaks_timestamps(k));
					if d < min_d
						min_d = d;
						ind1 = k;
						ind2 = l;
					end
				end
			end
			ind1 = candidate_peaks_locs(ind1);
			ind2 = candidate_peaks_locs(ind2);

			% Reordering the found peaks (permutation)
			if ind2 < ind1
				tmp = ind1;
				ind1 = ind2;
				ind2 = tmp;
			end

			% Resulting coarse template
			ctb = ind1 - (ind2 - ind1) / 2;
			ctb = max(1, ceil(ctb));
			cte = ind2;
			ct = v(ctb:cte);
		end

		function [fine_template, bpm, cycles] = generate_fine_template(data, varargin)
			% generate_fine_template - Generate fine template
			% 	
			% Syntax:  fine_template = Ubicomp18.generate_fine_template(data, opt)
			%
			% Inputs:
			%		data - Either a values matrix or a filename from which to fetch the values.
			%  		opt  - Optional name-value parameters
			%			should_plot     - Boolean. If false the result is not plotted. Default: false
			%			period - The period of time domain values (time interval between two consecutive values)
			%			heart_rate_only - Boolean. If true, only return the heart rate. Useful to avoid code duplication. Default: false
			%			start_at        - Time at which to start the extraction (in s). Default 20
			%			duration        - The duration (in s) of the heartbeat slice to extract and perform computation on. Default: 10
			%      
			% Outputs:
			%  	 	fine_template - The fine template
			%------------- BEGIN CODE --------------
			

			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'data');
			addParameter(p, 'should_plot', false);
			addParameter(p, 'period', 0.005);
			addParameter(p, 'heart_rate_only', false);
			addParameter(p, 'start_at', 30);
			addParameter(p, 'duration', 10);
			parse(p, data, varargin{:});

			data = p.Results.data;
			should_plot = p.Results.should_plot;
			period = p.Results.period;
			heart_rate_only = p.Results.heart_rate_only;
			start_at = p.Results.start_at;
			duration = p.Results.duration;

			% Actual computation
			if isa(data, 'char') || isa(data, 'string')
				values = extract_values_single(data, 'start_at', start_at, 'duration', duration);
			else
				values = data;
			end
			v = values(:,3);
			t = values(:,1);

			% Compute coarse template
			[ct, ctb, cte] = Ubicomp18.extract_coarse_template(values);
			t2 = t(ctb:cte);
			
			% Cross-correlation between coarse template and the input signal
			correlation = xcorr(ct, v);

			if should_plot
				plot(t, v)
				hold on
				plot(t2, ct)
				xlabel("Time (s)")
				ylabel("Acceleration (m/s^2)")
				legend("All", "Coarse template")
				title("Raw heartbeats")
				xlim([t(1) t(end)])

				figure
				% plot(t, correlation(1:length(t)));
				plot(correlation(1:length(t)));
				xlabel("Time (s)")
				title("Correlation with coarse template")
				% xlim([t(1) t(end)])
			end

			% Find the mean interval between the correlation peaks
			min_peak_height = max(correlation) * 0.25;
			min_peak_dist = 500 * 1e-3/period; % 500ms converted to distance between indices
			[pks, locs] = findpeaks(correlation, 'MinPeakHeight', min_peak_height, 'MinPeakDistance', min_peak_dist);
			locs = min(locs, length(t));
			beat_times = t(locs);
			beat_interval = mean(diff(beat_times));
			bpm = 60/beat_interval; % Number of beats per minute
			
			fine_template = [];
			if heart_rate_only
				return
			end

			% Extract heartbeat cycles (normal, as described in the paper)
			n_cycles = length(locs) - 1;
			max_cycle_length = max(diff(locs));
			cycles = zeros(n_cycles, max_cycle_length);
			for k=1:n_cycles
				cycle = v(locs(k):locs(k+1)-1);
				cycles(k,1:length(cycle)) = cycle;
			end


			% ----- Optimized alignement process (kjiokeng's addition) -----
			% Principle: Align the cycles based on their highest peaks
			% n_cycles = length(locs) - 1;
			% max_cycle_length = max(diff(locs));
			% second_dim = 3*max_cycle_length;
			% cycles = zeros(n_cycles, second_dim);

			% % Place the first cycle at the center
			% cycle = v(locs(1):locs(2)-1);
			% cycle_length = length(cycle);
			% cycle_start_idx = floor(second_dim/2 - cycle_length/2 + 1);
			% cycle_end_idx = cycle_start_idx + cycle_length - 1;
			% cycles(1,cycle_start_idx:cycle_end_idx) = cycle;

			% % Get the location of the peak to align (the highest peaks)
			% [~, cycle_locs] = findpeaks(cycles(1,:), 'SortStr','descend');
			% anchor_loc = cycle_locs(1);
			% if cycle_locs(1) < cycle_locs(2)
			% 	anchor_loc = cycle_locs(2);
			% end

			% for k=2:n_cycles
			% 	cycle = v(locs(k):locs(k+1)-1);

			% 	min_peak_dist = 400 * 1e-3/period; % 500ms converted to index distance
			% 	[~, cycle_locs] = findpeaks(cycle, 'SortStr','descend', 'MinPeakDistance', min_peak_dist);
			% 	loc_to_align = cycle_locs(1);
			% 	if cycle_locs(1) < cycle_locs(2)
			% 		loc_to_align = cycle_locs(2);
			% 	end

			% 	cycle_length = length(cycle);
			% 	cycle_start_idx = anchor_loc - loc_to_align;
			% 	cycle_end_idx = cycle_start_idx + cycle_length - 1;
			% 	cycles(k,cycle_start_idx:cycle_end_idx) = cycle;
			% end
			% ----- End of optimized alignment process -----

			% Average all the cycles and remove trailing zeros
			fine_template = mean(cycles);

			% close all
			if should_plot
				figure
				plot(cycles')
				for k=1:n_cycles
					plot(cycles(k,:))
					hold on
				end
				hold on
				plot(fine_template, 'r', 'LineWidth',2)
			end

			% Compute the derivative
			m = 8; % 40ms at 200Hz
			derivative = zeros(length(fine_template)-m, 1);
			for k=1:length(fine_template)-m
				derivative(k) = fine_template(k+m)-fine_template(k);
			end

			% Find the first peak of the derivative
			[pks, locs] = findpeaks(derivative, 'MinPeakHeight', 0.025);
			ft_start = locs(1);
			fine_template(1:ft_start-1) = 0;
			% fine_template(ft_start+ceil(beat_interval/period):end) = 0;

			if should_plot
				plot(fine_template, 'k', 'LineWidth',3)
				% plot(derivative, 'c--', 'LineWidth',2)
			end

			% Remove leading and trailing zeros
			fine_template = fine_template(find(fine_template,1,'first'):find(fine_template,1,'last'));
		end
		
		function bpm = compute_heart_rate(data, varargin)
			% compute_heart_rate - Compute heart_rate
			% 	
			% Syntax:  bpm = Ubicomp18.compute_heart_rate(values, opt)
			%
			% Inputs:
			%		data - Either a values matrix or a filename from which to fetch the values.
			%  		opt  - Optional name-value parameters
			%			should_plot  - Boolean. If false the result is not plotted. Default: false
			%      
			% Outputs:
			%  	 	bmp  - The heart rate (expressed in beats per minute)
			%------------- BEGIN CODE --------------

			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'data');
			addParameter(p, 'should_plot', false);
			parse(p, data, varargin{:});
			
			data = p.Results.data;
			should_plot = p.Results.should_plot;

			[~, bpm] = Ubicomp18.generate_fine_template(data, 'heart_rate_only', true, 'should_plot', should_plot);
		end

		function cycles = extract_cycles(data, template, varargin)
			% extract_cycles - Extract cycles based on the given cycle (fine) template
			% 	
			% Syntax:  cycles = Ubicomp18.extract_cycles(data, template, opt)
			%
			% Inputs:
			%		data     - Either a values matrix or a filename from which to fetch the values.
			%		template - The (fine) template with which to compare the input data
			%  		opt      - Optional name-value parameters
			%			should_plot - Boolean. If false the result is not plotted. Default: false
			%			period      - The period of time domain values (time interval between two consecutive values)
			%			start_at    - Time at which to start the extraction (in s). Default 20
			%			duration    - The duration (in s) of the heartbeat slice to extract and perform computation on. Default: 10
			%      
			% Outputs:
			%  	 	cycles - Cell array where each cell corresponds to a cycle
			%------------- BEGIN CODE --------------

			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'data');
			addRequired(p, 'template');
			addParameter(p, 'should_plot', false);
			addParameter(p, 'period', 0.005);
			addParameter(p, 'start_at', 1);
			addParameter(p, 'duration', 55);
			parse(p, data, template, varargin{:});

			data = p.Results.data;
			should_plot = p.Results.should_plot;
			period = p.Results.period;
			start_at = p.Results.start_at;
			duration = p.Results.duration;

			% Actual computation
			if isa(data, 'char') || isa(data, 'string')
				values = extract_values_single(data, 'start_at', start_at, 'duration', duration);
			else
				values = data;
			end
			
			% Old implem
			% [~, ~, cycles] = Ubicomp18.generate_fine_template(data, 'should_plot', should_plot);
			% % Remove leading and trailing zeros
			% avg_abs_cycle = sum(abs(cycles));
			% real_start = find(avg_abs_cycle,1,'first');
			% real_end = find(avg_abs_cycle,1,'last');
			% cycles = cycles(:,real_start:real_end);

			% New implem
			v = values(:,3);
			t = values(:,1);
			
			% Cross-correlation between coarse template and the input signal
			correlation = xcorr(template, v);

			if should_plot
				plot(t, v)
				hold on
				plot(t(1:length(template)), template)
				xlabel("Time (s)")
				ylabel("Acceleration (m/s^2)")
				legend("All", "Fine template")
				title("Raw heartbeats")
				xlim([t(1) t(end)])

				figure
				% plot(t, correlation(1:length(t)));
				plot(correlation(1:length(t)));
				xlabel("Time (s)")
				title("Correlation with fine template")
				% xlim([t(1) t(end)])
			end

			% Find the mean interval between the correlation peaks
			min_peak_height = max(correlation) * 0.25;
			min_peak_dist = 500 * 1e-3/period; % 500ms converted to distance between indices
			[pks, locs] = findpeaks(correlation, 'MinPeakHeight', min_peak_height, 'MinPeakDistance', min_peak_dist);
			locs = min(locs, length(t));			

			% Extract heartbeat cycles (normal, as described in the paper)
			n_cycles = length(locs) - 1;
			max_cycle_length = max(diff(locs));
			cycles = zeros(n_cycles, max_cycle_length);
			for k=1:n_cycles
				cycle = v(locs(k):locs(k+1)-1);
				cycles(k,1:length(cycle)) = cycle;
			end

			% ----- Optimized alignement process (kjiokeng's addition) -----
			% Principle: Align the cycles based on their highest peaks
			% n_cycles = length(locs) - 1;
			% max_cycle_length = max(diff(locs));
			% second_dim = 3*max_cycle_length;
			% cycles = zeros(n_cycles, second_dim);

			% % Place the first cycle at the center
			% cycle = v(locs(1):locs(2)-1);
			% cycle_length = length(cycle);
			% cycle_start_idx = floor(second_dim/2 - cycle_length/2 + 1);
			% cycle_end_idx = cycle_start_idx + cycle_length - 1;
			% cycles(1,cycle_start_idx:cycle_end_idx) = cycle;

			% % Get the location of the peak to align (the highest peaks)
			% [~, cycle_locs] = findpeaks(cycles(1,:), 'SortStr','descend');
			% anchor_loc = cycle_locs(1);
			% if cycle_locs(1) < cycle_locs(2)
			% 	anchor_loc = cycle_locs(2);
			% end

			% for k=2:n_cycles
			% 	cycle = v(locs(k):locs(k+1)-1);

			% 	min_peak_dist = 200 * 1e-3/period; % 500ms converted to index distance
			% 	[~, cycle_locs] = findpeaks(cycle, 'SortStr','descend', 'MinPeakDistance', min_peak_dist);
			% 	loc_to_align = cycle_locs(1);
			% 	if cycle_locs(1) < cycle_locs(2)
			% 		loc_to_align = cycle_locs(2);
			% 	end

			% 	cycle_length = length(cycle);
			% 	cycle_start_idx = anchor_loc - loc_to_align;
			% 	cycle_end_idx = cycle_start_idx + cycle_length - 1;
			% 	cycles(k,cycle_start_idx:cycle_end_idx) = cycle;
			% end
			% ----- End of optimized alignment process -----

			if should_plot
				figure
				plot(cycles')
			end
		end

		function features = extract_features(cycle, varargin)
			% extract_features - Extract (DWT) features from a given (temporal) cycle
			% 	
			% Syntax:  features = Ubicomp18.extract_features(cycle, opt)
			%
			% Inputs:
			%		cycle - The segmented cycle data.
			%  		opt  - Optional name-value parameters
			%      
			% Outputs:
			%  	 	features - Array of extracted features (of length Ubicomp18.normalized_features_length)
			%------------- BEGIN CODE --------------

			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'cycle');
			parse(p, cycle, varargin{:});

			cycle = p.Results.cycle;

			% Normalization
			cycle = cycle ./ max(cycle);
			cycle_norm = zeros(1, Ubicomp18.normalized_cycle_length); 
			m = min(Ubicomp18.normalized_cycle_length, length(cycle));
			cycle_norm(1:m) = cycle(1:m);
			
			% DWT features extraction
			features = Helper.dwt(cycle_norm);
			% features = abs(fft(cycle_norm, Ubicomp18.normalized_features_length));
		end

		function [is_user, prob] = predict(cycle, svm_model, varargin)
			% predict - Predict if a given cycle is from a given user
			% 	
			% Syntax:  is_user = Ubicomp18.predict(cycle, svm_model, opt)
			%
			% Inputs:
			%		cycle - The segmented cycle data
			%		svm_model - The SVM model corresponding to the user
			%  		opt  - Optional name-value parameters
			%  			compute_features  - Boolean. Indicates if we should compute features from the first input
			% 								or consider it as features already. Default: true
			%      
			% Outputs:
			%  	 	is_user - Boolean indicating if the given cycle is from the given user
			%------------- BEGIN CODE --------------

			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'cycle');
			addRequired(p, 'svm_model');
			addParameter(p, 'compute_features', true);
			parse(p, cycle, svm_model, varargin{:});

			cycle = p.Results.cycle;
			svm_model = p.Results.svm_model;
			compute_features = p.Results.compute_features;

			if compute_features
				features = Ubicomp18.extract_features(cycle);
			else
				features = cycle;
			end

			% [label, score] = predict(svm_model, features);
			% is_user = logical(label);
			% prob = score(label+1);

			[predict_label, accuracy, prob_estimates] = ...
			svmpredict(0, features, svm_model, '-b 1');
			is_user = logical(predict_label);
			prob = prob_estimates;
		end

		function obj = load(source_file, varargin)
			% load - Load a Ubicomp18 object from the given file
			% 	
			% Syntax:  obj = Ubicomp18.load(source_file, opt)
			%
			% Inputs:
			%		source_file - The file from which to load the object
			%  		opt  - Optional name-value parameters
			%      
			% Outputs:
			%  	 	obj - The loaded Ubicomp18 object
			%------------- BEGIN CODE --------------

			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'source_file');
			parse(p, source_file, varargin{:});

			source_file = p.Results.source_file;

			% Load the file content into a structure
			S = load(source_file);
			obj = S.obj;
		end

		function [] = test(varargin)
			% test - Helper function (script) to test different functionnalities
			% 	
			% Syntax:  [] = Ubicomp18.test(opt)
			%------------- BEGIN CODE --------------

			% filename = "../../measures/chest/KJIOKENG-MALE-24yrs-60s-STANDING-NORMAL_2019-11-06:14:31:00.csv";
			filename = "../../measures/chest/GENTIAN-MALE-40yrs-60s-STANDING-NORMAL_2019-11-08:10:41:52.csv";
			% filename = "../../measures/chest/ALB-MALE-52yrs-60s-STANDING-NORMAL_2019-11-06:16:37:15.csv";
			% filename = "../../measures/chest/GUILLAUME-MALE-30yrs-60s-STANDING-NORMAL_2019-11-06:16:28:25.csv";
			% filename = "../../measures/chest/MOHAMED-MALE-24yrs-60s-STANDING-NORMAL_2019-11-08:09:52:45.csv";
			
			% bpm = Ubicomp18.compute_heart_rate(filename, 'should_plot', true)
			% [fine_template, bpm] = Ubicomp18.generate_fine_template(filename, 'should_plot', true);
			% cycles = Ubicomp18.extract_cycles(filename, fine_template, 'should_plot', true);

			
			obj = Ubicomp18.load("comp-12-11-19-unique");
			n_users = length(obj.user_ids);
			features = {};
			n_features = 0;
			n_cycles_per_user = 10;
			for l=1:n_users
				features_l = obj.features{l};
				n_cycles = size(features_l, 1);
				for m=1:n_cycles_per_user
					n_features = n_features + 1;
					features{n_features} = features_l(m, :);
				end
			end

			comp = zeros(n_features, n_features);
			for k1=1:n_features
				x1 = features{k1};
				for k2=1:n_features
					x2 = features{k2};

					d = 1/(abs(Helper.correlation(x1, x2)) + eps);
					comp(k1, k2) = d;				
				end
			end

			image(comp(end:-1:1,:), 'CDataMapping','scaled')
			colorbar
			xticks((1:n_users) * n_cycles_per_user)
			yticks((1:n_users) * n_cycles_per_user)
			title("Ground truth")
			grid on
		end

		function [predictions, ground_truth, comparison] = main(varargin)
			% main - Main function
			% 	
			% Syntax:  [] = Ubicomp18.main(opt)
			%------------- BEGIN CODE --------------

			obj = Ubicomp18;
			obj.dir_path = "../../measures/chest/comp-12-11-19-unique";
			obj = obj.build();
			obj = obj.train();
			obj.save();

			obj = Ubicomp18.load("comp-12-11-19-unique");
			Ubicomp18.predict(obj.fine_templates{1}, obj.svm_models{1})

			% Prediction
			n_users = length(obj.user_ids);
			predictions = zeros(n_users, 1);
			ground_truth = zeros(n_users, 1);
			n_cycles_per_user = zeros(n_users, 1);
			
			% Make predictions with each of the svm models
			for k=1:n_users
				svm_model = obj.svm_models{k};

				% Loop over all the cycles of the other users
				cycle_idx = 1;
				for l=1:n_users
					features_l = obj.features{l};
					n_cycles = size(features_l, 1);
					n_cycles_per_user(l) = n_cycles;
					for m=1:n_cycles
						prediction = Ubicomp18.predict(features_l(m,:), svm_model, 'compute_features', true);
						predictions(k, cycle_idx) = prediction;
						ground_truth(k, cycle_idx) = (k==l);
						cycle_idx = cycle_idx + 1;
					end
				end
			end

			% Useful for plotting
			x_ticks = zeros(1, n_users);
			x_ticks(1) = n_cycles_per_user(1);
			for k=2:n_users
				x_ticks(k) = x_ticks(k-1) + n_cycles_per_user(k);
			end

			% Plot the ground truth
			image(ground_truth(end:-1:1,:), 'CDataMapping','scaled')
			colorbar
			xticks(x_ticks)
			xticklabels(obj.user_ids)
			xtickangle(45)
			xlabel("Heatbeat cycles")
			yticks(1:n_users)
			yticklabels(obj.user_ids(end:-1:1))
			ytickangle(45)
			ylabel("Classifiers (is it user x ?)")
			title("Ground truth")
			grid on

			% Plot the predictions
			figure
			image(predictions(end:-1:1,:), 'CDataMapping','scaled')
			colorbar
			xticks(x_ticks)
			xticklabels(obj.user_ids)
			xtickangle(45)
			xlabel("Heatbeat cycles")
			yticks(1:n_users)
			yticklabels(obj.user_ids(end:-1:1))
			ytickangle(45)
			ylabel("Classifiers (is it user x ?)")
			title("Predictions")
			grid on

			% Comparison
			comparison = abs(predictions - ground_truth);
			figure
			image(comparison(end:-1:1,:), 'CDataMapping','scaled')
			colorbar
			xticks(x_ticks)
			xticklabels(obj.user_ids)
			xtickangle(45)
			xlabel("Heatbeat cycles")
			yticks(1:n_users)
			yticklabels(obj.user_ids(end:-1:1))
			ytickangle(45)
			ylabel("Classifiers (is it user x ?)")
			title("Comparison")
			grid on
		end
   end
end