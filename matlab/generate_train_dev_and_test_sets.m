% generate_train_dev_and_tes_sets.m - Prepare data as input to the neural net
%
% MAT-files required: extract_values.m
%
% See also: extract_values
% Author: Kevin Jiokeng
% email: kevin.jiokeng@enseeiht.fr
% January 2020; Last revision: 20-January 2020
%------------- BEGIN CODE --------------


% directory = '../../measures/chest/comp-08-11-19';
directory = '../../measures/chest/comp-12-11-19-unique';
directory = '../../measures/hand/pixel/';
directory = '../../measures/hand/nexus/gyro/';
directory = '../../measures/hand/s8/analysis/';
directory = '../../measures/hand/s8/analysis2/';
directory = '../../measures/hand/s8/all/';
% directory = '../../measures/tmp';

start_ats = 0:2:20;
duration = 4;
% period = 0.0194; % Pixel 2
% period = 0.005; % Nexus 5
period = 0.01; % S8
gyr_period = 0.002;

heartbeat_files = dir(directory);
heartbeat_files = heartbeat_files(3:end);
num_files = length(heartbeat_files);

filenames = {};
ids = {};
ages = [];
genders = []; % true for MALE
for k=1:num_files
	filenames{k} = strcat(heartbeat_files(k).folder, '/', heartbeat_files(k).name);

	% Extract the user id from the filename
	tmp = strsplit(filenames{k}, '/');
	tmp = tmp{end};
	tmp = strsplit(tmp, '-');
	ids{k} = tmp{1};
	% ids{k} = strcat(tmp{1}, '-', tmp{5}); % Take the id and the position

	genders(k) = strcmp(tmp{2}, 'MALE');
	ages(k) = str2num(strrep(tmp{3}, 'yrs', ''));
end

fft_size = 256;
% X = zeros(num_files * length(start_ats), fft_size);
X = zeros(num_files * length(start_ats), 99, fft_size/2+1);
Y = zeros(num_files * length(start_ats), 1);
idx = 1;
for k=1:num_files
	fprintf("Current file: %s\n", heartbeat_files(k).name)

	for s=1:length(start_ats)
		start_at = start_ats(s);
		% Acc
		values = extract_values(filenames{k}, 'start_at', start_at, 'duration', duration, ...
			'period', period, 'sensor', 'ACC');
		t = values(:,1);
		
		% PCA
		[vals, vr, U] = Helper.pca(values(:,2:end));
		v = vals(:, 1);
		% v = Helper.filter_noise(v);

		% z = Helper.to_frequential(v, 'period', period, 'fft_size', fft_size);
		z = abs(spectrogram(v, 8, [], fft_size, 1/period, 'power'));

		% Save the result
		% X(idx, :) = z;
		X(idx, :, :) = rot90(z);
		Y(idx) = k-1;
		idx = idx + 1;	
	end
end


% x_mu = mean(X);
% x_std = std(X);
% X = (X - x_mu) / x_std;

% Shuffle the arrays
% m = length(Y);
% perm = randperm(m);
% X = X(perm, :);
% Y = Y(perm, :);

% % Split in 60% train, 20% dev, 20% test
% m_train = floor(m * 0.6);
% m_dev = floor(m * 0.2);
% m_test = m - m_train - m_dev;

% X_train = X(1:m_train, :);
% Y_train = Y(1:m_train);

% X_dev = X(m_train+1:m_train+m_dev, :);
% Y_dev = Y(m_train+1:m_train+m_dev);

% X_test = X(m_train+m_dev+1:end, :);
% Y_test = Y(m_train+m_dev+1:end);


% Spectrograms
m = length(Y);
perm = randperm(m);
X = X(perm, :, :);
Y = Y(perm, :, :);

% Split in 60% train, 20% dev, 20% test
m_train = floor(m * 0.6);
m_dev = floor(m * 0.2);
m_test = m - m_train - m_dev;

X_train = X(1:m_train, :, :);
Y_train = Y(1:m_train);

X_dev = X(m_train+1:m_train+m_dev, :, :);
Y_dev = Y(m_train+1:m_train+m_dev);

X_test = X(m_train+m_dev+1:end, :, :);
Y_test = Y(m_train+m_dev+1:end);

% Save to file
% save('heartbeatdata.mat', 'X_train', 'Y_train', 'X_dev', 'Y_dev', 'X_test', 'Y_test')
% save('heartbeatdata-all.mat', 'X_train', 'Y_train', 'X_dev', 'Y_dev', 'X_test', 'Y_test')
save('heartbeatdata-spectrograms.mat', 'X_train', 'Y_train', 'X_dev', 'Y_dev', 'X_test', 'Y_test')