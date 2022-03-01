fs = 100;
ts = 1/fs;
hr = 80;
f_hr = hr/60;
t_cycle = 1/f_hr;
f_oscil = 20;

t1 = 0:ts:t_cycle;
s1 = sin(2*pi*f_oscil*t1);
e1 = exp(-8*t1);
w1 = s1 .* e1;
plot(t1, s1)
hold on
plot(t1, e1, 'LineWidth', 2)
plot(t1, w1, 'LineWidth', 2)
legend(sprintf("s1 = sin(2 x pi x %d x t)", f_oscil), "s2 = exp(-8 x t)", "s = s1 x s2")
figure





ltb = length(tb);
t = [tb, t1, t_cycle+ts+t1, 2*t_cycle+ts+t1, 3*t_cycle+ts+t1(1:end-ltb)];
w = [w1(end-ltb+1:end), w1, w1, w1, w1(1:end-ltb)];

plot(t, w, 'LineWidth', 1.2)
xlim([min(t) max(t)])
legend("Perfect signal")

snr = 2;
% n = (randn(1, length(t)) - 0.5);
n = (randn(1, length(t)) - 0.5);
n = n * max(abs(w)) / (max(abs(n)) * snr);
wn = w + n;
figure
plot(t, wn, 'r', 'LineWidth', 1.2)
xlim([min(t) max(t)])
legend("Signal with noise")

s = sin(2*pi*f_oscil*t);
sa = abs(s);
wa = abs(w);
wna = abs(wn);
na = abs(n);