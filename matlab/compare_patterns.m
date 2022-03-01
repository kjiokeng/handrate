function dist = compare_patterns(values, ids, varargin)
% compare_patterns - Compare heartbeat patterns
% 
% Syntax:  dist = compare_patterns(values, ids, opt)
%
% Inputs:
%    values - Cell array containing the measures corresponding to each pattern
%    ids - Cell array containing the ids of each user
%    opt - Optional name-value parameters
%		   method - The comparison method. Either 'euclidian', 'dtw', 'fft_euclidian', 'fft_dtw'. Default: 'fft_euclidian'
%		   should_plot - Boolean. If false the result is not plotted. Default: true
%		   change_labels - Boolean. If false the ids are not plotted as x- and y-labels. Default: true
%
% Outputs:
%    dist - A 2D matrix containing the distances between each pair of elements
%
% MAT-files required: extract_values.m
%
% See also: extract_values
% Author: Kevin Jiokeng
% email: kevin.jiokeng@enseeiht.fr
% November 2019; Last revision: 06-November-2019
%------------- BEGIN CODE --------------

% Parse the input parameters
p = inputParser;
valid_char = @(x) isa(x, 'char');
addRequired(p, 'values');
addRequired(p, 'ids');
addParameter(p, 'method', 'fft_euclidian', valid_char);
addParameter(p, 'should_plot', true);
addParameter(p, 'change_labels', true);
parse(p, values, ids, varargin{:});

values = p.Results.values;
ids = p.Results.ids;
method = p.Results.method;
should_plot = p.Results.should_plot;
change_labels = p.Results.change_labels;


n = length(values);
dist = zeros(n, n);

% Selection of the comparison function
comp_function = select_comp_function(method);

% Actual comparison
for k=1:n
% parfor k=1:n
	fprintf("\t--- Method: %s, Pattern %d/%d (%.1f%%)\n", method, k, n, (k-1)*100/n);
	for l=1:n
		x1 = values{k};
		x2 = values{l};

		m = min(length(x1), length(x2));
		x1 = x1(1:m);
		x2 = x2(1:m);

		d = comp_function(x1, x2);
		dist(k, l) = d;
	end
end

% Plot
if should_plot
	image(rot90(dist), 'CDataMapping','scaled')
	% image(rot90(dist))
	colormap default
	colorbar
	grid on

	if change_labels
		n_repets = n/length(ids);
		xticks((0:length(ids)-1)*n_repets+0.5)
		xticklabels(ids)
		xtickangle(45)
		yticks((1:length(ids))*n_repets)
		yticklabels(ids(end:-1:1))
		ytickangle(45)
	end

	title(strcat("Distance with method ", upper(strrep(method, '_', '\_'))))
end

end

%% Helper functions
function comp_function = select_comp_function(method)
	comp_function = @f_otherwise;
	switch method
		case 'euclidian'
			comp_function = @f_euclidian;
		case 'dtw'
			comp_function = @f_dtw;
		case 'fft_euclidian'
			comp_function = @f_fft_euclidian;
		case 'fft_dtw'
			comp_function = @f_fft_dtw;
		case 'corrcoef'
			comp_function = @f_corrcoef;
		case 'fft_corrcoef'
			comp_function = @f_fft_corrcoef;
		case 'cos'
			comp_function = @f_cos;
		case 'fft_cos'
			comp_function = @f_fft_cos;
		case 'dwt_euclidian'
			comp_function = @f_dwt_euclidian;
		case 'dwt_corrcoef'
			comp_function = @f_dwt_corrcoef;
		case 'dwt_cos'
			comp_function = @f_dwt_cos;
		case 'dwt_dtw'
			comp_function = @f_dwt_dtw;
		case 'cceps_euclidian'
			comp_function = @f_cceps_euclidian;
		case 'cceps_corrcoef'
			comp_function = @f_cceps_corrcoef;
		case 'stft_euclidian'
			comp_function = @stft_euclidian;
		case 'stft_corrcoef'
			comp_function = @stft_corrcoef;
		case 'cwt_euclidian'
			comp_function = @cwt_euclidian;
		case 'cwt_corrcoef'
			comp_function = @cwt_corrcoef;
		otherwise
			comp_function = @f_otherwise;
	end
end

function d = f_euclidian(x1, x2)
		d = norm(x1 - x2);
end

function d = f_dtw(x1, x2)
	d = dtw(x1, x2);
end

function d = f_fft_euclidian(x1, x2)
	f1 = Helper.to_frequential(x1);
	f2 = Helper.to_frequential(x2);
	d = norm(f1 - f2);
end

function d = f_fft_dtw(x1, x2)
	f1 = Helper.to_frequential(x1);
	f2 = Helper.to_frequential(x2);
	d = dtw(f1, f2);
end

function d = f_corrcoef(x1, x2)
	d = 1/(abs(Helper.correlation(x1, x2)) + eps);
end

function d = f_fft_corrcoef(x1, x2)
	f1 = Helper.to_frequential(x1);
	f2 = Helper.to_frequential(x2);
	d = 1/(abs(Helper.correlation(f1, f2)) + eps);
end

function d = f_cos(x1, x2)
	d = dot(x1, x2);
	d = d / (norm(x1) * norm(x2));
	d = abs(1/d);
end

function d = f_fft_cos(x1, x2)
	f1 = Helper.to_frequential(x1);
	f2 = Helper.to_frequential(x2);
	d = dot(f1, f2);
	d = d / (norm(f1) * norm(f2));
	d = abs(1/d);
end

function d = f_dwt_euclidian(x1, x2)
	f1 = Helper.dwt(x1);
	f2 = Helper.dwt(x2);
	d = norm(f1 - f2);
end

function d = f_dwt_corrcoef(x1, x2)
	f1 = Helper.dwt(x1);
	f2 = Helper.dwt(x2);
	d = 1/(abs(Helper.correlation(f1, f2)) + eps);
end

function d = f_dwt_cos(x1, x2)
	f1 = Helper.dwt(x1);
	f2 = Helper.dwt(x2);
	d = dot(f1, f2);
	d = d / (norm(f1) * norm(f2));
	d = abs(1/d);
end

function d = f_dwt_dtw(x1, x2)
	f1 = Helper.dwt(x1);
	f2 = Helper.dwt(x2);
	d = dtw(f1, f2);
end

function d = f_cceps_euclidian(x1, x2)
	f1 = cceps(x1);
	f2 = cceps(x2);
	d = norm(f1 - f2);
end

function d = f_cceps_corrcoef(x1, x2)
	f1 = cceps(x1);
	f2 = cceps(x2);
	d = 1/(abs(Helper.correlation(f1, f2)) + eps);
end

function d = stft_euclidian(x1, x2)
	period = 0.01; % S8
	f1 = abs(spectrogram(x1, 100, 80, 2048, 1/period, 'yaxis', 'power'));
	f2 = abs(spectrogram(x2, 100, 80, 2048, 1/period, 'yaxis', 'power'));
	f1 = f1(floor(end/2):end,:);
	f2 = f2(floor(end/2):end,:);
	% d = norm(f1 - f2);
	d = min(Helper.sliding_euclidian(f1, f2));
end

function d = stft_corrcoef(x1, x2)
	period = 0.01; % S8
	f1 = abs(spectrogram(x1, 100, 80, 512, 1/period, 'yaxis', 'power'));
	f2 = abs(spectrogram(x2, 100, 80, 512, 1/period, 'yaxis', 'power'));
	f1 = f1(floor(end/2):end,:);
	f2 = f2(floor(end/2):end,:);
	d = 1/(max(Helper.xcorr_2d(f1, f2)) + eps);
end

function d = cwt_euclidian(x1, x2)
	period = 0.01; % S8
	f1 = abs(cwt(x1));
	f2 = abs(cwt(x2));
	f1 = f1(floor(end/2):end,:);
	f2 = f2(floor(end/2):end,:);
	% d = norm(f1 - f2);
	d = min(Helper.sliding_euclidian(f1, f2));
end

function d = cwt_corrcoef(x1, x2)
	period = 0.01; % S8
	f1 = abs(cwt(x1));
	f2 = abs(cwt(x2));
	f1 = f1(floor(end/2):end,:);
	f2 = f2(floor(end/2):end,:);
	d = 1/(max(Helper.xcorr_2d(f1, f2)) + eps);
end

function d = f_otherwise(x1, x2)
	d = Inf;
end