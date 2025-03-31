clear;
clc;

% Define material properties
material_name = '12.1_SRM709LeadOxideGlass'; % Change this to your material name

B_ep1 = (4.0905) - (1.73E-002j);
C_ep1 = (4.0905) - (1.73E-002j);
D_ep1 = (2E-004) - (3.1E-003);
G_ep1 = (4.0905) - (1.75E-002j);
H_ep1 = (1E-004) - (1.4E-003j);
I_ep1 = (7.4E-003) + (4E-003j);
J_ep1 = (5E-004) - (6E-004j);

% Define frequency range
f_min = 0.2; % Hz
f_max = 10; % Hz
num_points = 100-1; % Number of frequency points, -1 for smoother numbers
frequencies = linspace(f_min, f_max, num_points);

% Compute permittivity epsilon(f)
epsilon_f = B_ep1 + (2 * C_ep1 .* (frequencies .^ D_ep1)) + (G_ep1 .* (1 - J_ep1 .* ((frequencies - H_ep1).^2) - 1i * 2 * I_ep1 .* frequencies) .^ (-1));

% μ = 1
mu_f = (frequencies * 0) + 1; % Modify if permeability data is available

% Prepare data for export
data_table = table(frequencies', real(epsilon_f)', imag(epsilon_f)', mu_f', 'VariableNames', {'Frequency_Hz', 'Real_Epsilon', 'Imag_Epsilon', 'Mu'});

% Define file name
file_name = strcat(material_name, '.csv');

% Export to CSV
writetable(data_table, file_name);

% Display message
fprintf('Data exported to %s\n', file_name);