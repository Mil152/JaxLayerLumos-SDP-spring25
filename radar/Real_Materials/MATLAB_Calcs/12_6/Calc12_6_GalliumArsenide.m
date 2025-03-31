clear;
clc;

% Define material properties
material_name = '12_6_GalliumArsenide'; % Change this as needed

% Given values for permittivity (ε(f)) equation 1
B_ep1 = (3.63) - (0.13j);
C_ep1 = (3.64) - (7E-002j);
D_ep1 = (8E-002) + (3E-002j);
E_ep1 = (0.19) - (0.3j);
F_ep1 = (3.73) - (4E-002j);
G_ep1 = (76.9) - (367j);
H_ep1 = (-547) + (598j);

% Define frequency range (GHz)
f_min = 3;  % Hz
f_max = 10;  % Hz
num_points = 100-29; % Number of frequency points, 29 is the number of pts from 0.2-3
frequencies = linspace(f_min, f_max, num_points);

% Compute permeability μ(f) exactly as given in the equation
mu_f = (frequencies * 0) + 1;

% Compute permittivity ε(f) (Real part)
e_f = B_ep1 + real(C_ep1) .* (frequencies .^ D_ep1) + imag(C_ep1) .* (frequencies .^ E_ep1) + F_ep1 .* (1 - (frequencies ./ G_ep1).^2 - 1i * 2 .* frequencies ./ H_ep1).^(-1);

% Prepare data for export
data_table = table(frequencies', real(e_f)', imag(e_f)', mu_f', 'VariableNames', {'Frequency_GHz', 'Real_Epsilon', 'Imag_Epsilon', 'Mu'});

% Define file name
file_name = strcat(material_name, '.csv');

% Export to CSV
writetable(data_table, file_name);

% Display message
fprintf('Data exported to %s\n', file_name);
