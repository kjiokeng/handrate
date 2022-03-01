classdef Helper
% Helper - A class to gather helper functions
%
% Author: Kevin Jiokeng
% email: kevin.jiokeng@enseeiht.fr
% November 2019; Last revision: 03-Mars-2020
%------------- BEGIN CODE --------------

   methods (Static)
   		function [z, f] = to_frequential(values, varargin)
			% to_frequential - Convert readings to frequential domain
			% 	
			% Syntax:  [z, f] = Helper.to_frequential(values, opt)
			%
			% Inputs:
			%		values - The time domain values
			%  		opt - Optional name-value parameters
			%			period - The period of time domain values (time interval between two consecutive values)
			%			fft_size - The fft size
			%      
			% Outputs:
			%  	 	z - The amplitude in the frequency domain
			%  	 	f - The corresponding frequencies
			%------------- BEGIN CODE --------------

			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'values');
			addParameter(p, 'period', 0.005);
			addParameter(p, 'fft_size', 1024);
			parse(p, values, varargin{:});

			values = p.Results.values;
			period = p.Results.period;
			fft_size = max(p.Results.fft_size, length(values));

			
			% Actual computation
			z = abs(fftshift(fft(values, fft_size))).^2/fft_size;
			fs = 1/period;
			f = (-fft_size/2:fft_size/2-1)*fs/fft_size;
		end

		function [detailed_coefs_array, approx_coefs_array, detailed_coefs_cell] = ...
		dwt(values, varargin)
			% dwt - performs discrete wavelet transform
			% 	
			% Syntax:  [z, f] = Helper.dwt(values, opt)
			%
			% Inputs:
			%		values - The time domain values
			%  		opt - Optional name-value parameters
			%			method - The wavelet method. Default: 'db2'
			%			n_levels - The number of decomposition levels. Default: 6
			%			levels - The levels to output (filtering). Default: 3:5
			%      
			% Outputs:
			%  	 	detailed_coefs_array - Array of concatenated (output) detailed coefficients
			%  	 	approx_coefs_array - Array of approximation coefficients
			%  	 	detailed_coefs_cell - Cell array of (output) detailed coefficients, each entry corresponding to one level
			%------------- BEGIN CODE --------------

			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'values');
			addParameter(p, 'method', 'dmey', @ischar);
			addParameter(p, 'n_levels', 6);
			addParameter(p, 'levels', 3:5);
			parse(p, values, varargin{:});

			values = p.Results.values;
			method = p.Results.method;
			n_levels = p.Results.n_levels;
			levels = p.Results.levels;
			% levels = 1:n_levels;
			
			% Actual computation
			[c,l] = wavedec(values,n_levels,method);
			approx_coefs_array = appcoef(c,l,method);
			detailed_coefs_cell = detcoef(c,l,levels);
			detailed_coefs_array = [];
			for k=1:length(detailed_coefs_cell)
				tmp = detailed_coefs_cell{k};
				detailed_coefs_array = [detailed_coefs_array reshape(tmp, 1, length(tmp))];
			end
			% tmp = approx_coefs_array;
			% detailed_coefs_array = [reshape(tmp, 1, length(tmp)) detailed_coefs_array];
		end

		function y = filter_noise(x, varargin)
			% filter_noise - Filter the input
			% 	
			% Syntax:  y = Helper.filter_noise(x, opt)
			%
			% Inputs:
			%		x - The non-filtered time domain values
			%  		opt - Optional name-value parameters
			%			method - The preprocessing method used
			%			freq_range - The 'allowed' frequency range
			%			sampling_freq - The sampling frequency of the input signal
			%			n_points - The number of points over which to perform the filtering
			%      
			% Outputs:
			%  	 	y - The filtered values
			%------------- BEGIN CODE --------------

			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'x');
			% addParameter(p, 'freq_range', [0.5 50]);
			% addParameter(p, 'sampling_freq', 200);
			addParameter(p, 'freq_range', [0.5 25]);
			addParameter(p, 'sampling_freq', 100);
			addParameter(p, 'n_points', 3);
			addParameter(p, 'method', 'movmean');
			parse(p, x, varargin{:});

			x = p.Results.x;
			freq_range = p.Results.freq_range;
			sampling_freq = p.Results.sampling_freq;
			n_points = p.Results.n_points;
			method = p.Results.method;
			
			% Actual computation
			% y = bandpass(x, freq_range, sampling_freq);

			% Actual filtering
			switch method
				case 'movmean'
					y = movmean(x, n_points);
				case 'smooth'
					y = smoothdata(x,'movmean', 'SmoothingFactor',0.7);
				case 'medfilt'
					y = medfilt1(x, n_points);
				case 'bandpass'
					y = bandpass(x,freq_range,sampling_freq,'Steepness',0.85,'StopbandAttenuation',60);
				case 'wavelet'
					[y, tmp] = wdenoise(x,7, ...
								    'Wavelet', 'sym4', ...
								    'DenoisingMethod', 'Bayes', ...
								    'ThresholdRule', 'Median', ...
								    'NoiseEstimate', 'LevelIndependent');
				case 'wavelet2'
					[y, tmp] = wdenoise(x,11, ...
								    'Wavelet', 'sym8', ...
								    'DenoisingMethod', 'SURE', ...
								    'ThresholdRule', 'Soft', ...
								    'NoiseEstimate', 'LevelDependent');

					% Center at zero by removing the approximation coefficients
					% approx = tmp{end};
					% la = length(approx);
					% lx = length(x);
					% approx_idx = linspace(1, lx, la);
					% approx = interp1(approx_idx, approx, 1:lx);
					% y = y - approx';

				case 'none'
					y = x;
			end
		end

		function y = detrend(x, varargin)
			% detrend - Remove polynomial trend from a signal
			% 	
			% Syntax:  y = Helper.detrend(x, opt)
			%
			% Inputs:
			%		x - The signal from which to remove trend
			%  		opt - Optional name-value parameters
			%			polynomial_order - The order of the polynomial
			%      
			% Outputs:
			%  	 	y - The detrended signal
			%------------- BEGIN CODE --------------

			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'x');
			addParameter(p, 'polynomial_order', 3);
			parse(p, x, varargin{:});

			x = p.Results.x;
			polynomial_order = p.Results.polynomial_order;
			
			% Actual computation
			t = (1:length(x))';
			[p, s, mu] = polyfit(t, x, polynomial_order);
			f_y = polyval(p, t, [], mu);
			y = x - f_y;
		end

		function r = correlation(x1, x2, varargin)
			% correlation - Compute the normalized correlation
			% 	
			% Syntax:  r = Helper.correlation(x1, x2, opt)
			%
			% Inputs:
			%		x1, x2 - The input vectors
			%  		opt - Optional name-value parameters
			%      
			% Outputs:
			%  	 	r - The correlation coefficient between the two inputs
			%------------- BEGIN CODE --------------

			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'x1');
			addRequired(p, 'x2');
			parse(p, x1, x2, varargin{:});

			x1 = p.Results.x1;
			x2 = p.Results.x2;


			% Make sure the inputs have the same length
			l = min(length(x1), length(x2));
			y1 = x1(1:l);
			y2 = x2(1:l);

			% Actual computation
			% n1 = norm(x1);
			% n2 = norm(x2);
			% r = dot(y1, y2)/(n1*n2);

			% r = corrcoef(y1, y2);
			% r = abs(r(1,2));

			r = max(xcorr(x1, x2));
			r = r / (norm(x1) * norm(x2));
		end

		function [r, lags] = xcorr_2d(x1, x2, varargin)
			% xcorr_2d - Compute the normalized correlation
			% 	
			% Syntax:  r = Helper.xcorr_2d(x1, x2, opt)
			%
			% Inputs:
			%		x1, x2 - The input matrices
			%  		opt - Optional name-value parameters
			%  			  should_normalize - Boolean. If true, normalize the results by
			%  			  					 the product of matrices energy. Default: true.
			%      
			% Outputs:
			%  	 	r - The cross-correlation coefficients between the two inputs
			%------------- BEGIN CODE --------------

			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'x1');
			addRequired(p, 'x2');
			addParameter(p, 'should_normalize', true);
			parse(p, x1, x2, varargin{:});

			x1 = p.Results.x1;
			x2 = p.Results.x2;
			should_normalize = p.Results.should_normalize;


			% Make sure the two inputs have same size by padding with zeros
			[m1, n1] = size(x1);
			[m2, n2] = size(x2);
			M = max(m1, m2);
			N = max(n1, n2);
			z1 = zeros(M, N);
			z2 = zeros(M, N);
			z1(1:m1,1:n1) = x1;
			z2(1:m2,1:n2) = x2;

			% Actual computation
			r = zeros(1, 2*N-1);
			lags = -N+1:N-1;

			% Negative lags
			for l=1:N-1
				s1 = z1(:, 1:l);
				s2 = z2(:, N-l+1:N);
				r(l) = sum(sum(s1 .* s2));
			end

			% Positive lags
			for l=1:N-1
				s1 = z1(:, l+1:N);
				s2 = z2(:, 1:N-l);
				r(N+l) = sum(sum(s1 .* s2));
			end
			
			% Lag = 0
			r(N) = sum(sum(z1 .* z2));
			r(N) = r(N);

			if should_normalize
				denom = sqrt(sum(sum(x1 .* x1)) * sum(sum(x2 .* x2)));
				r = r / denom;
			end
		end

		function [r, lags] = sliding_euclidian(x1, x2, varargin)
			% sliding_euclidian - Compute the normalized sliding euclidian distance
			% 	
			% Syntax:  r = Helper.sliding_euclidian(x1, x2, opt)
			%
			% Inputs:
			%		x1, x2 - The input matrices
			%  		opt - Optional name-value parameters
			%  			  should_normalize - Boolean. If true, normalize the results by
			%  			  					 the product of matrices energy. Default: true.
			%      
			% Outputs:
			%  	 	r - The cross-correlation coefficients between the two inputs
			%------------- BEGIN CODE --------------

			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'x1');
			addRequired(p, 'x2');
			addParameter(p, 'should_normalize', true);
			parse(p, x1, x2, varargin{:});

			x1 = p.Results.x1;
			x2 = p.Results.x2;
			should_normalize = p.Results.should_normalize;


			% Make sure the two inputs have same size by padding with zeros
			[m1, n1] = size(x1);
			[m2, n2] = size(x2);
			M = max(m1, m2);
			N = max(n1, n2);
			z1 = zeros(M, N);
			z2 = zeros(M, N);
			z1(1:m1,1:n1) = x1;
			z2(1:m2,1:n2) = x2;

			% Actual computation
			r = zeros(1, 2*N-1);
			denoms = zeros(1, 2*N-1);
			lags = -N+1:N-1;

			% Negative lags
			for l=1:N-1
				s1 = z1(:, 1:l);
				s2 = z2(:, N-l+1:N);
				r(l) = norm(s1 - s2);
				denoms(l) = numel(s1) * numel(s2);
			end

			% Positive lags
			for l=1:N-1
				s1 = z1(:, l+1:N);
				s2 = z2(:, 1:N-l);
				r(N+l) = norm(s1 - s2);
				denoms(N+l) = numel(s1) * numel(s2);
			end
			
			% Lag = 0
			r(N) = sum(sum(z1 .* z2));
			r(N) = r(N);
			denoms(N) = numel(z1) * numel(z2);

			if should_normalize
				r = r ./ denoms;
			end
		end

		function hr = compute_heart_rate(x, varargin)
			% compute_heart_rate - Compute heart rate based on the measurements vector x
			% 	
			% Syntax:  hr = Helper.compute_heart_rate(x, opt)
			%
			% Inputs:
			%		x - The time domain measurements
			%  		opt - Optional name-value parameters
			%			period - The period of time domain values (time interval between two consecutive values)
			%			method - The estimation method, correlation-based or fft-based. Default: fft-based
			%      
			% Outputs:
			%  	 	hr - The estimated heart rate
			%------------- BEGIN CODE --------------

			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'x');
			addParameter(p, 'period', 0.005);
			addParameter(p, 'method', 'fft');
			parse(p, x, varargin{:});

			x = p.Results.x;
			period = p.Results.period;
			method = p.Results.method;

			hr = 0;
			switch method
				case 'fft'
					[z, f] = Helper.to_frequential(x, 'period', period);
					heart_activity_inds = f>0.9 & f<2;
					f = f(heart_activity_inds);
					z = z(heart_activity_inds);
					[m, m_ind] = max(z);
					hr = 60*f(m_ind);

				case 'corr'
					one_second = ceil(1/period);
					xt = x(one_second:2*one_second);
					[autocor, lags] = xcorr(xt, x);
					[pks, locs] = findpeaks(autocor, 'MinPeakDist', 0.45/period);
					p = mean(diff(locs)) * period;
					hr = 60/p;

				case 'wavelet'
					sampling_freq = 1/period;
					[wt, f] = cwt(x, sampling_freq, ...
							'VoicesPerOctave', 16, ...
						    'TimeBandWidth', 10);
					wt = max(abs(wt));
					[pks, locs] = findpeaks(wt, 'MinPeakDist', 0.45/period);
					p = mean(diff(locs)) * period;
					hr = 60/p;
			end
		end

		function [X_proj, var_ret, U, S] = pca(X)
			% pca - Perform Principal Components Analysis on the input matrix X
			% 	
			% Syntax:  [X_proj, var_ret, U, S] = Helper.pca(X, opt)
			%
			% Inputs:
			%		X - The input matrix
			%  		opt - Optional name-value parameters
			%			
			%      
			% Outputs:
			%  	 	X_proj - The data projected in the new basis
			%  	 	var_ret - The amount (%) of variance retained
			%  	 	U - The projection matrix
			%  	 	S - The matrix S from SVD
			%------------- BEGIN CODE --------------

			% Normalization
			X_norm = X - mean(X);
			X_norm = X_norm ./ std(X_norm);

			% Singular Value Decomposition
			[m, n] = size(X);
			Sigma = X' * X / m;
			[U, S, V] = svd(Sigma);

			% Projection
			X_proj = X_norm * U;

			% Variance retained
			var_ret = diag(S) / sum(sum(S));
		end

		function [X_proj] = ica(X)
			% ica - Perform Independent Components Analysis on the input matrix X
			% 	
			% Syntax:  [X_proj] = Helper.ica(X, opt)
			%
			% Inputs:
			%		X - The input matrix
			%  		opt - Optional name-value parameters
			%			
			%      
			% Outputs:
			%  	 	X_proj - The data projected in the new basis
			%------------- BEGIN CODE --------------

			mdl = rica(X, 3);
			X_proj = transform(mdl, X);
		end

		function euler_angles = rotm2euler(R)
			% rotm2euler - Compute euler angles from rotation matrix R
			% 	
			% Syntax:  euler_angles = Helper.rotm2euler(R, opt)
			%
			% Inputs:
			%		R - The input rotation matrix
			%  		opt - Optional name-value parameters
			%			
			%      
			% Outputs:
			%  	 	euler_angles - The computed euler angles
			%------------- BEGIN CODE --------------

			theta_x = atan2(R(3, 2), R(3, 3));
			theta_y = atan2(-R(3, 1), sqrt(R(3, 2)*R(3, 2) + R(3, 3)*R(3, 3)));
			theta_z = atan2(R(2, 1), R(1, 1));


			euler_angles = [theta_x theta_y theta_z];
		end

		function [values] = read_ecg_file(filename, varargin)
			% read_ecg_file - Read ECG measures
			% 	
			% Syntax:  [values] = Helper.read_ecg_file(filename, opt)
			%
			% Inputs:
			%		filename - The name of the file to be read
			%  		opt - Optional name-value parameters
			%  	 		base_value - Value corresponding to zero
			%  	 		gain - Scaling coefficients (in units per mV)
			%  	 		time_res - The time resolution (time separating to consecutive readings)
			%			
			%      
			% Outputs:
			%  	 	values - A 2-column array with:
			%  	 			first column: the time (in s), relative to the first measure
			%  	 			second column: the read values, expressed in mV
			%------------- BEGIN CODE --------------


			% Default values are the one observed empirically on the "ECG Palmar PM10" device
			p = inputParser;
			addRequired(p, 'filename');
			addParameter(p, 'base_value', 2048);
			addParameter(p, 'gain', 50);
			addParameter(p, 'time_res', 1/250);
			addParameter(p, 'start_at', 0);
			addParameter(p, 'duration', Inf);
			parse(p, filename, varargin{:});

			filename = p.Results.filename;
			base_value = p.Results.base_value;
			gain = p.Results.gain;
			time_res = p.Results.time_res;
			start_at = p.Results.start_at;
			duration = p.Results.duration;

			% Actually reading the file
			fid = fopen(filename, 'r');
			[data, count] = fread(fid, Inf, 'int16');
			fclose(fid);

			data = (data - base_value) / gain;
			time = (0:count-1) * time_res;

			start_idx = 1 + max(ceil(start_at/time_res), 0);
			end_idx = min(floor(start_idx-1 + duration/time_res), count);
			selected_idx = start_idx:end_idx;
			data = data(selected_idx);
			time = time(selected_idx);

			% Make sure the data is 'positively oriented'
			[~, max_idx] = max(abs(data));
			data = data * sign(data(max_idx));

			values = zeros(length(selected_idx), 2);
			values(:, 1) = time;
			values(:, 2) = data;
		end

		function [out, out_time] = resample_readings(in, in_freq, out_freq, varargin)
			% resample_reading - Resample readings to match a desired sampling frequency
			% 	
			% Syntax:  [out, out_time] = Helper.resample_reading(in, in_frequency, out_frequency, opt)
			%
			% Inputs:
			%		in - The input data to resample
			%		in_freq - The sampling frequency of the input data
			%		out_freq - The desired sampling frequency for the output data
			%  		opt - Optional name-value parameters
			%			
			%      
			% Outputs:
			%  	 	out - The resampled data
			%  	 	out_time - The corresponding time values (in s), relative to the first measure
			%------------- BEGIN CODE --------------


			% Default values are the one observed empirically on the "ECG Palmar PM10" device
			p = inputParser;
			addRequired(p, 'in');
			addRequired(p, 'in_freq');
			addRequired(p, 'out_freq');
			parse(p, in, in_freq, out_freq, varargin{:});

			in = p.Results.in;
			in_freq = p.Results.in_freq;
			out_freq = p.Results.out_freq;

			% Actual resampling
			in_time = (0:length(in)-1) / in_freq;
			out_time = 0:1/out_freq:in_time(end);
			out = interp1(in_time, in, out_time);
		end

		function [] = print_matrix(mat, varargin)
			% print_matrix - Print a given matrix (without the default Matlab multiplication factor)
			% 	
			% Syntax:  [] = Helper.print_matrix(mat, opt)
			%
			% Inputs:
			%		mat - The matrix to print
			%  		opt - Optional name-value parameters
			%			
			%      
			% Outputs:
			%------------- BEGIN CODE --------------


			p = inputParser;
			addRequired(p, 'mat');
			parse(p, mat, varargin{:});

			mat = p.Results.mat;

			for i=1:size(mat, 1)
				for j=1:size(mat, 2)
					fprintf("%.6f ", mat(i, j))
				end
				fprintf("\n")
			end
		end

		function [q_kurt] = compute_q_kurt(values, varargin)
			% compute_q_kurt - Compute Q_kurt
			% 	
			% Syntax:  [q_kurt] = Helper.compute_q_kurt(values, opt)
			%
			% Inputs:
			%		values - The time domain values
			%  		opt - Optional name-value parameters
			%			period - The period of time domain values (time interval between two consecutive values)
			%			fft_size - The fft size
			%      
			% Outputs:
			%  	 	q_kurt - The result
			%------------- BEGIN CODE --------------

			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'values');
			addParameter(p, 'period', 0.005);
			addParameter(p, 'fft_size', 1024);
			parse(p, values, varargin{:});

			values = p.Results.values;
			period = p.Results.period;
			fft_size = max(p.Results.fft_size, length(values));

			
			% Actual computation
			[z, f] = Helper.to_frequential(values, 'period', period, 'fft_size', fft_size);
			z = z(f>=0);
			f = f(f>=0);
			heart_activity_inds = f>0.5 & f<2;
			f_ha = f(heart_activity_inds);
			z_ha = z(heart_activity_inds);
			z_ha = z_ha * 1000;
			f_ha = f_ha' * 60;
			f = f_ha;
			z = z_ha;
			[m, m_ind] = max(z_ha);
			indices = 1:length(z);
			% indices = indices(heart_activity_inds);
			% m_ind = m_ind + indices(1) - 1;
			stem(f, z)
			hold on
			stem(f(m_ind), z(m_ind), 'ro')
			hold off
			hr = 60*f(m_ind);
			k_real = kurtosis(z);

			fft_perfect = zeros(size(z));
			fft_perfect(m_ind) = z(m_ind);
			k_perfect = kurtosis(fft_perfect);

			q_kurt = k_real / k_perfect;
			% pause
		end

		function [integ] = integrate(values, varargin)
			% Integrate - Compute the integral of a signal
			% 	
			% Syntax:  [integ] = Helper.integrate(values, opt)
			%
			% Inputs:
			%		values - The values to integrate
			%  		opt - Optional name-value parameters
			%			period - The period of time domain values (time interval between two consecutive values)
			%      
			% Outputs:
			%  	 	integ - The result
			%------------- BEGIN CODE --------------

			% Parse the input parameters
			p = inputParser;
			addRequired(p, 'values');
			addParameter(p, 'period', 0.005);
			parse(p, values, varargin{:});

			values = p.Results.values;
			period = p.Results.period;


			n = length(values);
			integ = zeros(1, n);
			for k=2:n
				integ(k) = (k-1) * period * (values(k) + values(1))/2;
			end
		end

   end
end