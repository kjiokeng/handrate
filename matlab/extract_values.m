function [values, period] = extract_values(filename, varargin)
% extract_values - Extract values from a sensor readings file (with contains multiple sensors)
% 
% Syntax:  values = extract_values(filename, opt)
%
% Inputs:
%    filename - The name of the file from which to extract the measures
%    opt - Optional name-value parameters
%		sensor - The type of sensor data to extract. Default ACC
%		start_at - Time at which to start the extraction (in s). Default 1
%       duration - Duration of the extracted sample (in s). Default 3
%    	should_plot - Boolean. If true, the extracted values are plotted. Default: false
%
% Outputs:
%    values - A 4-column 2D matrix containing the timestamps, x, y and z values respectively (columns)
%
% Author: Kevin Jiokeng
% email: kevin.jiokeng@enseeiht.fr
% November 2019; Last revision: 06-November-2019
%------------- BEGIN CODE --------------

% Parse the input parameters
p = inputParser;
addRequired(p, 'filename');
addParameter(p, 'sensor', 'ACC');
addParameter(p, 'start_at', 0);
addParameter(p, 'duration', Inf);
addParameter(p, 'period', 0.005);
addParameter(p, 'should_plot', false);
parse(p, filename, varargin{:});


filename = p.Results.filename;
sensor = p.Results.sensor;
start_at = p.Results.start_at;
duration = p.Results.duration;
period = p.Results.period;
should_plot = p.Results.should_plot;

switch sensor
	case 'ACC'
		sensor = 'android.sensor.linear_acceleration';
	case 'GYR'
		sensor = 'android.sensor.gyroscope';
	case 'GRAVITY'
		sensor = 'android.sensor.gravity';
	case 'RAW_ACC'
		sensor = 'android.sensor.accelerometer';
	case 'MAGNETIC'
		sensor = 'android.sensor.magnetic_field';
	case 'ROT_VEC'
		sensor = 'android.sensor.rotation_vector';
	otherwise
		sensor = 'android.sensor.linear_acceleration';
end


% Import values from the file (ignoring the headers)
%values = csvread(filename, 1, 0);
T = readtable(filename);
T = T(strcmp(T.sensor, sensor),[1 3 4 5]);
values = table2array(T);

% Rebase the timestamps and convert to seconds
values(:,1) = (values(:,1) - values(1,1)) * 1e-9;

% Take only a given portion of the readings
end_at = start_at + duration;
indices = max(ceil(start_at/period),1):min(ceil(end_at)/period,size(values,1));
values = values(indices,:);

% Ploting the values
if should_plot
	figure
	hold on
	plot(values(:,1), values(:,2), 'b', 'LineWidth', 1)
	plot(values(:,1), values(:,3), 'r', 'LineWidth', 1)
	plot(values(:,1), values(:,4), 'g', 'LineWidth', 1)
	title("Heartbeat pattern")
	legend('x', 'y', 'z')
	xlabel("Time (in s)")
	ylabel("Amplitude (m/s^2)")
	xlim([values(1,1) values(end,1)])
end

end