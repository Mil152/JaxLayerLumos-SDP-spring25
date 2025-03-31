clear;
clc;

% Define material properties
material_name = '12_8_EpoxyEAB'; % Change this as needed

% Given values for permittivity (ε(f)) equation 1 using 90˚ incidence angle
% and average between samples A, B , and C
B_ep1 = (1.0702) + (5.32E-002j);
C_ep1 = (0.8609) - (2.38E-002j);
D_ep1 = (-0.5203) + (0.1529j);
G_ep1 = (0.8918) - (9.54E-002j);
H_ep1 = (-2E-003) + (1.7E-003j);
I_ep1 = (1E-004) - (4.8E-003j);
J_ep1 = (-4.85E-005) + (5.91E-008j);

% Define frequency range (GHz)
f_min = 2;  % Hz
f_max = 10;  % Hz
num_points = 100-19; % Number of frequency points, 19 is the number of pts from 0.2-2
frequencies = linspace(f_min, f_max, num_points);

% μ(f) = 1
mu_f = (frequencies * 0) + 1;

% Compute permittivity ε(f)
e_f = B_ep1 + 2 * C_ep1 .* (frequencies .^ D_ep1) + G_ep1 .* (1 - J_ep1 .* (frequencies - H_ep1).^2 - 1i * 2 * I_ep1 .* frequencies) .^ (-1);

% Prepare data for export
data_table = table(frequencies', real(e_f)', imag(e_f)', mu_f', 'VariableNames', {'Frequency_GHz', 'Real_Epsilon', 'Imag_Epsilon', 'Mu'});

% Define file name
file_name = strcat(material_name, '.csv');

% Export to CSV
writetable(data_table, file_name);

% Display message
fprintf('Data exported to %s\n', file_name);
