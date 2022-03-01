% n = 100
% tmp = 100 * ones(n, n)
% for base=1:10:n-9
% 	for k=base:base+9
% 		for l=base:base+9
% 			tmp(k,l)=0;
% 		end
% 	end
% end
% image(rot90(tmp), 'CDataMapping','scaled')
% colorbar
% xticks(0:10:n)
% xticklabels(0:10:n)
% yticks(0:10:n)
% yticklabels(n:-10:10)


% filename = "../../measures/KJIOKENG-MALE-24yrs-60s-STANDING-NORMAL_2019-11-06:14:31:00.csv";
% filename = "../../measures/MOHAMED-MALE-24yrs-60s-STANDING-NORMAL_2019-11-08:09:52:45.csv";
% start_at = 10;
% duration = 5;
% v5 = extract_values(filename, 'start_at', start_at, 'duration', duration);
% v5 = v5(:,3);

% % v1 = wdenoise(v,7, ...
% %     'Wavelet', 'sym4', ...
% %     'DenoisingMethod', 'Bayes', ...
% %     'ThresholdRule', 'Median', ...
% %     'NoiseEstimate', 'LevelIndependent');

%  Fstop1 = 1e-06;  % First Stopband Frequency
%     Fpass1 = 2;    % First Passband Frequency
%     Fpass2 = 5;     % Second Passband Frequency
%     Fstop2 = 100;    % Second Stopband Frequency
%     Astop1 = 60;     % First Stopband Attenuation (dB)
%     Apass  = 1;      % Passband Ripple (dB)
%     Astop2 = 60;     % Second Stopband Attenuation (dB)
%     Fs     = 200;    % Sampling Frequency
    
%     h = fdesign.bandpass('fst1,fp1,fp2,fst2,ast1,ap,ast2', Fstop1, Fpass1, ...
%         Fpass2, Fstop2, Astop1, Apass, Astop2, Fs);
    
%     Hd = design(h, 'butter', ...
%         'MatchExactly', 'stopband', ...
%         'SOSScaleNorm', 'Linf');

%     y = filter(Hd,v);

%     plot(v)
%     hold on
%     plot(y)
%     legend("Original", "Filtered")



% return
% n = length(v);
% period = 0.005;
% fs = 1/period;
% t = (0:n) * period;
% % s = spectrogram(v, fs*10, fs/2, 20000, fs, 'yaxis')
% s = stft(v, fs)
% image(abs(s))
% colorbar


% SVM Parameters
% C = 1; sigma = 0.1;

% % We set the tolerance and max_passes lower here so that the code will run
% % faster. However, in practice, you will want to run the training to
% % convergence.
% model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 

% or fitckernel




% period = 0.0194; % Pixel 2
% filename = "../../measures/hand/pixel/KJIOKENG-MALE-0yrs-30s-STANDING-NORMAL_2019-11-26:17:29:32.csv";
% filename = "../../measures/hand/pixel/GUILLAUME-MALE-0yrs-30s-STANDING-NORMAL_2019-11-26:17:32:58.csv";
% % filename = "../../measures/hand/pixel/MOHAMED-MALE-0yrs-30s-STANDING-NORMAL_2019-11-26:17:34:18.csv";

% period = 0.005; % Nexus 5
% % filename = "../../measures/hand/nexus/KJIOKENG-MALE-24yrs-30s-STANDING-NORMAL_2019-11-29:17:23:09.csv";
% filename = "../../measures/hand/nexus/KJIOKENG-MALE-24yrs-30s-STANDING-NORMAL_2019-11-29:17:23:46.csv";
% filename = "../../measures/hand/nexus/KJIOKENG-MALE-24yrs-30s-STANDING-NORMAL_2019-11-29:17:30:21.csv";

% % On the chest
% % filename = "../../measures/hand/nexus/KJIOKENG-MALE-24yrs-30s-STANDING-NORMAL_2019-11-29:17:31:13.csv";

% start_at = 10;
% duration = 10;

% % Read the values
% v = extract_values(filename, 'start_at', start_at, 'duration', duration, 'period', period);
% t = v(:, 1);
% v = v(:, 2:4);

% % Normalization
% v_norm = v - mean(v);
% v_norm = v_norm ./ std(v_norm);
% subplot(5, 1, 1)
% plot(t, v(:,1), 'LineWidth', 2)
% xlabel("Time (s)")
% xlim([min(t) max(t)])
% title(filename)
% legend("x", "y", "z")

% f = floor(1/period)
% template = v(1:2*f, 1);
% correlation = xcorr(template, v(:,1));
% subplot(5, 1, 2)
% plot(t, correlation(1:length(t)));
% xlim([min(t) max(t)])

% template = v(1:2*f, 2);
% correlation = xcorr(template, v(:,2));
% subplot(5, 1, 3)
% plot(t, correlation(1:length(t)));
% xlim([min(t) max(t)])

% template = v(1:2*f, 3);
% correlation = xcorr(template, v(:,3));
% subplot(5, 1, 4)
% plot(t, correlation(1:length(t)));
% xlim([min(t) max(t)])


% % PCA and projection
% [U, S] = pca(v_norm);
% v_proj = v_norm * U(:, 1:3);
% template = v_proj(1:2*f, 1);
% correlation = xcorr(template, v_proj(:,1));
% subplot(5, 1, 5)
% plot(t, correlation(1:length(t)), 'LineWidth', 2)
% xlabel("Time (s)")
% xlim([min(t) max(t)])
% title("PCA")
% legend("u1", "u2", "u3")

% % Variance retained
% n = size(S, 1);
% s = sum(sum(S));
% sk = zeros(n, 1);
% sk(1) = S(1);
% for k=2:n
% 	sk(k) = sk(k-1) + S(k,k);
% end
% sk = sk * 100 / s


Fs = 100;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 1000;             % Length of signal
t = (0:L-1)*T;
% X = sin(2*pi*50*t) + 0.5 * randn(size(t));
frequencies = [1 4 7 11 12];
amp = [1 0.75 1 2 1];
X = zeros(1, L);
for k=1:length(frequencies)
	X = X + amp(k) * sin(2*pi*frequencies(k)*t);
end
X = 0.07 * X / max(X);

subplot(3, 1, 1)
plot(1000*t,X)
title('Original signal')
xlabel('t (milliseconds)')
ylabel('X(t)')

X = X + 0.03 * randn(size(t));
subplot(3, 1, 2)
plot(1000*t,X)
title('Signal Corrupted with Zero-Mean Random Noise')
xlabel('t (milliseconds)')
ylabel('X(t)')

% Autocorr
X = X - mean(X);
[autocor,lags] = xcorr(X,'coeff');

subplot(3, 1, 3)
plot(lags,autocor)
xlabel('Lag')
ylabel('Autocorrelation')

[pksh,lcsh] = findpeaks(autocor, 'MinPeakDistance', 1/frequencies(1)*Fs/2);
period = 1000 * mean(diff(lcsh))/Fs;

hold on
pks = plot(lags(lcsh),pksh,'or');
hold off
legend(pks,['Period: ', num2str(period,0), ' ms'])