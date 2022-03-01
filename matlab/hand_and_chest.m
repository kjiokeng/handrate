start_at = 0;
duration = 10;
time_delta = 0.17;
% period = 0.005; % Nexus 5
% period = 0.01; % S8
% gyr_period = 0.002; % S8

n_rows = 2;
n_cols = 1;

% Pixel
period1 = 0.0194; % Pixel 2
gyr_period1 = 0.0025; % Pixel 2
% filename1 = "../../measures/hand/pixel/hr/KJ-HAND-OTHER-ON-CHEST-MALE-24yrs-60s-STATIC-NORMAL_2020-02-03:11:00:04.csv";
% filename1 = "../../measures/hand/pixel/hr/KJ-RELAXEDHAND-OTHER-ON-CHEST-MALE-24yrs-60s-STATIC-NORMAL_2020-02-03:11:04:52.csv";
% filename1 = "../../measures/hand/pixel/hr/KJ-POCKET-OTHER-ON-CHEST-MALE-24yrs-60s-STATIC-NORMAL_2020-02-03:11:02:31.csv";
% filename1 = "../../measures/hand/pixel/hr/KJ-RUN2-MALE-24yrs-60s-STATIC-NORMAL_2020-02-04:11:51:03.csv";
% filename1 = "../../measures/hand/pixel/hr/KJ-RUN3-MALE-24yrs-60s-STATIC-NORMAL_2020-02-04:19:18:52.csv";
% filename1 = "../../measures/hand/pixel/hr/KJ-RUN4-MALE-24yrs-60s-STATIC-NORMAL_2020-02-04:19:20:35.csv"
% filename1 = "../../measures/hand/pixel/hr/KJ-RUN5-MALE-24yrs-60s-STATIC-NORMAL_2020-02-04:19:22:13.csv"
% filename1 = "../../measures/hand/pixel/hr/KJ-RUN6-MALE-24yrs-60s-STATIC-NORMAL_2020-02-04:19:24:12.csv"
% filename1 = "../../measures/hand/pixel/hr/KJ-RUN7-MALE-24yrs-60s-STATIC-NORMAL_2020-02-04:19:26:32.csv"
filename1 = "../../measures/hand/pixel/hr/KJ-RUN8-MALE-24yrs-60s-STATIC-NORMAL_2020-02-04:19:28:13.csv"
% filename1 = "../../measures/hand/pixel/hr/KJ-RUN9-MALE-24yrs-60s-STATIC-NORMAL_2020-02-04:19:31:01.csv"

values1 = extract_values(filename1, 'start_at', start_at, 'duration', duration, ...
	'period', period1, 'sensor', 'ACC');
	% 'period', gyr_period1, 'sensor', 'GYR');
t1 = values1(:,1);
v1 = values1(:, 2:end);
% v1(:,1) = Helper.filter_noise(v1(:, 1), 'n_points', 30);
% v1(:,2) = Helper.filter_noise(v1(:, 2), 'n_points', 30);
% v1(:,3) = Helper.filter_noise(v1(:, 3), 'n_points', 30);
v1 = v1 - mean(v1);
v1 = v1 ./ max(abs(v1));
v1a = abs(v1);
% subplot(n_rows, n_cols, 1)
% plot(t1-start_at, v1)
% xlim([0 duration])

% figure
% v1 = Helper.pca(v1);
% instfreq(v1, 1/period1, 'FrequencyLimits', [0.8 2.5])
% ylim([0 5])
% title("toto toto")
% pause

% S8
period2 = 0.01; % S8
gyr_period2 = 0.002; % S8
% filename2 = "../../measures/hand/s8/hr/KJ-CHEST-OTHER-IN-HAND-MALE-24yrs-60s-STATIC-NORMAL_2020-02-03:11:01:06.csv";
% filename2 = "../../measures/hand/s8/hr/KJ-CHEST-OTHER-IN-RELAXEDHAND-MALE-24yrs-60s-STATIC-NORMAL_2020-02-03:11:05:55.csv";
% filename2 = "../../measures/hand/s8/hr/KJ-CHEST-OTHER-IN-POCKET-MALE-24yrs-60s-STATIC-NORMAL_2020-02-03:11:03:33.csv";
% filename2 = "../../measures/hand/s8/hr/KJ-CHEST-RUN2-MALE-24yrs-60s-STATIC-NORMAL_2020-02-04:11:52:06.csv";
% filename2 = "../../measures/hand/s8/hr/KJ-CHEST-RUN3-MALE-24yrs-60s-STATIC-NORMAL_2020-02-04:19:19:59.csv";
% filename2 = "../../measures/hand/s8/hr/KJ-CHEST-RUN4-MALE-24yrs-60s-STATIC-NORMAL_2020-02-04:19:21:42.csv"
% filename2 = "../../measures/hand/s8/hr/KJ-CHEST-RUN5-MALE-24yrs-60s-STATIC-NORMAL_2020-02-04:19:23:21.csv"
% filename2 = "../../measures/hand/s8/hr/KJ-CHEST-RUN6-MALE-24yrs-60s-STATIC-NORMAL_2020-02-04:19:25:20.csv"
% filename2 = "../../measures/hand/s8/hr/KJ-CHEST-RUN7-MALE-24yrs-60s-STATIC-NORMAL_2020-02-04:19:27:40.csv"
filename2 = "../../measures/hand/s8/hr/KJ-CHEST-RUN8-MALE-24yrs-60s-STATIC-NORMAL_2020-02-04:19:29:20.csv"
% filename2 = "../../measures/hand/s8/hr/KJ-CHEST-RUN9-MALE-24yrs-60s-STATIC-NORMAL_2020-02-04:19:32:08.csv"



values2 = extract_values(filename2, 'start_at', start_at+time_delta, 'duration', duration, ...
	'period', period2, 'sensor', 'ACC');
t2 = values2(:,1);
v2 = values2(:, 2:end);
% v2(:,1) = Helper.filter_noise(v2(:, 1), 'n_points', 20);
% v2(:,2) = Helper.filter_noise(v2(:, 2), 'n_points', 20);
% v2(:,3) = Helper.filter_noise(v2(:, 3), 'n_points', 20);
v2 = v2 - mean(v2);
v2 = v2 ./ max(abs(v2));
v2a = abs(v2);
% subplot(n_rows, n_cols, 2)
% plot(t2-start_at, v2)
% xlim([0 duration])


% Combination
% idx = abs(v1(:,3)) < 0.4;
% v1(idx,3) = 0;
% idx = abs(v2(:,2)) < 0.4;
% v2(idx,2) = 0;
subplot(n_rows, n_cols, 1)
plot(t1-start_at, abs(v1(:,3)))
hold on
plot(t2-start_at, abs(v2(:,2)))
xlim([0 duration])
xlabel("Time (s)")
ylabel("Normalized amplitude")
legend("In hand", "On the chest")
title("KJ-RUN6")
% pause

v4 = Helper.pca(abs(v1));
[z11, f11] = Helper.to_frequential(v1(:, 1), 'period', period1, 'fft_size', 8*4096);
[z12, f12] = Helper.to_frequential(v1(:, 2), 'period', period1, 'fft_size', 8*4096);
[z13, f13] = Helper.to_frequential(v1(:, 3), 'period', period1, 'fft_size', 8*4096);
[z21, f21] = Helper.to_frequential(v2(:, 1), 'period', period2, 'fft_size', 8*4096);
[z22, f22] = Helper.to_frequential(v2(:, 2), 'period', period2, 'fft_size', 8*4096);
[z23, f23] = Helper.to_frequential(v2(:, 3), 'period', period2, 'fft_size', 8*4096);

subplot(n_rows, n_cols, 2)
% plot(f1'*60, z1'/max(z1'))
hold on
plot(f11*60, z13)
% plot(f12*60, z12)
% plot(f13*60, z13)
% plot(f21*60, z21)
plot(f22*60, z22)
% plot(f23*60, z23)
xlim([50 130])
legend("In hand", "On the chest")
% legend("In hand, dim 1", "In hand, dim 2", "In hand, dim 3", "On the chest, dim 1", "On the chest, dim 2", "On the chest, dim 3")
xlabel("Heart rate in bpm (= 60 * Freq)")
ylabel("Amplitude")


v1a = abs(v1);
v2a = abs(v2);



s1 = v1(:,3);
s2 = v2(:,2);
t1 = t1 - t1(1);
t2 = t2 - t2(1);
t1 = t1(t1<=duration);
t2 = t2(t2<=duration);
s1 = s1(1:length(t1));
s2 = s2(1:length(t2));
[s1, t1] = Helper.resample_readings(s1, 1/period1, 100);
[s2, t2] = Helper.resample_readings(s2, 1/period2, 100);
m = min(length(t1), length(t2));
t1 = t1(1:m);
s1 = s1(1:m);
t2 = t2(1:m);
s2 = s2(1:m);
close all
plot(t1, s1)
hold on
plot(t2, s2)
legend("Hand", "Chest")

[pks1, locs1] = findpeaks(s1, 'MinPeakDistance', 0.5*100)
[pks2, locs2] = findpeaks(s2, 'MinPeakDistance', 0.62*100)
locs1 = locs1(1:end-1);
pks1 = pks1(1:end-1);
locs2 = locs2(1:end-1);
pks2 = pks2(1:end-1);
locs1(end-1) = 842;
pks1 = s1(locs1);
pks2 = s2(locs2);
figure
plot(s1)
hold on
plot(s2)
plot(locs1, pks1, 'bx')
plot(locs2, pks2, 'ro')
delta = locs1 - locs2
mean(delta)
std(delta)
res = [t1', s1', t2', s2']
locs1 = locs1-1;
locs2 = locs2-1;
[locs1'/100, pks1', locs2'/100, pks2']