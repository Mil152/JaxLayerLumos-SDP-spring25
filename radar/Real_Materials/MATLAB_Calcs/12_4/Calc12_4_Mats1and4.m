clear;
clc;

% Define material properties
material_name = 'VF_12_4_Mats1and4_NiZnCu_FeO4_Composite_About80µm9%VolumeInEpoxy_AND_About80µm29-9%VolumeInEpoxy'; % Change this as needed

% Given values for permeability (χ_m) equation 1
B_x1 = 2.21;
C_x1 = 0.866;
D_x1 = 0.940;

% Given values for permittivity (ε(f)) equation 1
B_ep1 = (2.57) + (0.55j);
C_ep1 = (1.46) + (0.56j);
D_ep1 = (-3E-003) - (4E-003j);
E_ep1 = (-0.44) - (1.1j);
F_ep1 = (20.16) + (7.5j);
G_ep1 = (-11.9) - (73.9j);

% Given values for permeability (χ_m) equation 2
B_x2 = 0.34;
C_x2 = 1.8694;
D_x2 = 0.8412;

% Given values for permittivity (ε(f)) equation 2
B_ep2 = 2.47 + 0.26j;
C_ep2 = 8E-002 - 0.14j;
D_ep2 = -0.61 + 0.68j;
E_ep2 = -0.17 + 0.48j;
F_ep2 = 0.66 - 7E-002j;
G_ep2 = 10.66 - 12.9j;
H_ep2 = -18.6 + 10.47j;

% Volume fraction
f = 0.09;

% Define frequency range (GHz)
f_min = 0.2;  % Hz
f_max = 10;    % Hz
num_points = 99; % Number of frequency points
frequencies = linspace(f_min, f_max, num_points);

% Compute permeability μ(f) exactly as given in the equation
x_1 = B_x1 .* (1 - 1i * frequencies ./ D_x1) ./ (1 - ((frequencies ./ (C_x1)).^2) - (1i * (frequencies ./ (D_x1))));

% Compute permeability μ(f) exactly as given in the equation
x_2 = B_x2 .* (1 - 1i * frequencies ./ D_x2) ./ (1 - ((frequencies ./ (C_x2)).^2) - (1i * (frequencies ./ (D_x2))));

% Compute permittivity ε1(f) (Real part)
e_1 = B_ep1 + C_ep1 .* (frequencies .^ D_ep1) + E_ep1 .* (1 - (frequencies ./ F_ep1).^2 - 2i * (frequencies ./ G_ep1)).^(-1);

% Compute permittivity ε2(f) (Imaginary part)
e_2 = B_ep2 + real(C_ep2) .* (frequencies .^ D_ep2) + imag(C_ep2) .* (frequencies .^ E_ep2) + F_ep2 .* (1 - (frequencies ./ G_ep2).^2 - 2i * (frequencies ./ H_ep2)).^(-1);

% Convert chi to mu
mu_1 = x_1 + 1;
mu_2 = x_2 + 1;

% Use MGT relations to find effective permittivity and permeability
e_e = (e_2)*((1 + ((2 * f)*((e_1 - e_2) / (e_1 + 2*e_2)))) / (1 - f*((e_1 - e_2) / (e_1 + 2*e_2))));
mu_e = (mu_2)*((1 + ((2 * f)*((mu_1 - mu_2) / (mu_1 + 2*mu_2)))) / (1 - f*((mu_1 - mu_2) / (mu_1 + 2*mu_2))));

% Prepare data for export
data_table = table(frequencies', real(e_e)', imag(e_e)', real(mu_e)', imag(mu_e)', 'VariableNames', {'Frequency_GHz', 'Real_Epsilon_MGT', 'Imag_Epsilon_MGT', 'Real_Mu_MGT', 'Imag_Mu_MGT'});

% Define file name
file_name = strcat(material_name, '.csv');

% Export to CSV
writetable(data_table, file_name);

% Display message
fprintf('Data exported to %s\n', file_name);