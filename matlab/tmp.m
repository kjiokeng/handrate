start_at = 10;
duration = 30;
% period = 0.005; % Nexus 5
% period = 0.01; % S8
% gyr_period = 0.002; % S8

n_rows = 2;
n_cols = 2;

% Pixel
period1 = 0.0194; % Pixel 2
gyr_period1 = 0.0025; % Pixel 2
% period2 = 0.01; % S8
% gyr_period2 = 0.002; % S8
% filename1 = "../../measures/hand/pixel/hr/HANDSTATIC-MALE-24yrs-60s-STATIC-NORMAL_2020-02-19:10:57:26.csv"
filename1 = "../../measures/hand/pixel/hr/HANDSTATIC-MALE-24yrs-60s-STATIC-NORMAL_2020-02-19:10:59:01.csv"
% filename1 = "../../measures/hand/pixel/hr/FK-MALE-22yrs-60s-STATIC-NORMAL_2020-02-19:17:13:52.csv"

values1 = extract_values(filename1, 'start_at', start_at, 'duration', duration, ...
	'period', period1, 'sensor', 'ACC');
	% 'period', gyr_period1, 'sensor', 'GYR');
t1 = values1(:,1);
v1 = values1(:, 2:end);
% v1(:,1) = Helper.filter_noise(v1(:, 1), 'n_points', 3);
% v1(:,2) = Helper.filter_noise(v1(:, 2), 'n_points', 3);
% v1(:,3) = Helper.filter_noise(v1(:, 3), 'n_points', 3);
v1 = v1 - mean(v1);
v1 = v1 ./ max(abs(v1));
% v1 = abs(v1);
subplot(n_rows, n_cols, 1)
plot(t1-start_at, v1)
xlim([0 duration])
legend("Dim 1", "Dim 2", "Dim 3")
xlabel("Time (s)")
ylabel("Amplitude")

% figure
% v1 = Helper.pca(v1);
% instfreq(v1, 1/period1, 'FrequencyLimits', [0.8 2.5])
% ylim([0 5])
% title("toto toto")
% pause

[z11, f11] = Helper.to_frequential(v1(:, 1), 'period', period1, 'fft_size', 8*4096);
[z12, f12] = Helper.to_frequential(v1(:, 2), 'period', period1, 'fft_size', 8*4096);
[z13, f13] = Helper.to_frequential(v1(:, 3), 'period', period1, 'fft_size', 8*4096);

subplot(n_rows, n_cols, 2)
hold on
plot(f11*60, z13)
plot(f12*60, z12)
plot(f13*60, z13)
xlim([50 130])
legend("Dim 1", "Dim 2", "Dim 3")
xlabel("Heart rate in bpm (= 60 * Freq)")
ylabel("Amplitude")



%% PCA
[v2, var_ret, U, S] = Helper.pca((v1));
% v2 = abs(v1);
[z21, f21] = Helper.to_frequential(v2(:, 1), 'period', period1, 'fft_size', 8*4096);
[z22, f22] = Helper.to_frequential(v2(:, 2), 'period', period1, 'fft_size', 8*4096);
[z23, f23] = Helper.to_frequential(v2(:, 3), 'period', period1, 'fft_size', 8*4096);

subplot(n_rows, n_cols, 3)
plot(t1-start_at, v2)
legend("Dim 1", "Dim 2", "Dim 3")
xlabel("Time (s)")
ylabel("Amplitude")

subplot(n_rows, n_cols, 4)
hold on
plot(f21*60, z23)
plot(f22*60, z22)
plot(f23*60, z23)
xlim([50 130])
legend("Dim 1", "Dim 2", "Dim 3")
xlabel("Heart rate in bpm (= 60 * Freq)")
ylabel("Amplitude")


v1a = abs(v1);