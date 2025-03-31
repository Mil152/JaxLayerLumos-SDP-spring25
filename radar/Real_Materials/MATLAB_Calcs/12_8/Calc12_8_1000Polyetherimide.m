clear;
clc;

% Define material properties
material_name = '12_8_1000Polyetherimide'; % Change this as needed

% Given values for permittivity (ε(f)) equation 1 using 90˚ incidence angle
B_ep1 = (0.7855) + (9E-004j);
C_ep1 = (0.7872) + (8E-004j);
D_ep1 = (-2.93E-002) - (7E-004j);
G_ep1 = (0.7851) + (1E-003j);
H_ep1 = (8E-004) + (1E-004j);
I_ep1 = (1E-004) - (2.7E-003j);
J_ep1 = (-4.51E-005) - (2.27E-006j);

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
