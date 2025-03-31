clear;
clc;

% Define material properties
material_name = '12_7_110Polymethacrylimide-1Foam'; % Change this as needed

% Given values for permittivity (ε(f)) equation 1 using 90˚ incidence angle
B_ep1 = (0.292) + (2E-003j);
C_ep1 = (0.292) + (2E-003j);
D_ep1 = (0);
G_ep1 = (0.292) + (2E-003j);
H_ep1 = (0);
I_ep1 = (0);
J_ep1 = (0);

% Define frequency range (GHz)
f_min = 1;  % Hz
f_max = 10;  % Hz
num_points = 100-9; % Number of frequency points, 9 is the number of pts from 0.2-1
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
