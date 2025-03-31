clear;
clc;

% Define material properties
material_name = '12_7_CarbonReticulatedFoam'; % Change this as needed

% Given values for permittivity (ε(f)) equation 1 using 90˚ incidence angle
% and average between samples A, B , and C
B_ep1 = (1.47) + (0.41j);
C_ep1 = (-0.46) + (2.01j);
D_ep1 = (-0.96) + (0.25j);
G_ep1 = (0.66) + (0.91j);
H_ep1 = (0.25) + (0.22j);
I_ep1 = (-0.44) + (0.13j);
J_ep1 = (-0.22) + (0.07j);

% Define frequency range (GHz)
f_min = 0.5;  % Hz
f_max = 6;  % Hz
num_points = 100-4-40; % Number of frequency points, 4 is the number of pts from 0.2-0.5, 40 is the number of pts from 6-10
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
