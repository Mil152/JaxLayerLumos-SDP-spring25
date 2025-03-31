clear;
clc;

% Define material properties
material_name = '12.1_BerylliumOxide'; % Change this to your material name

B_ep1 = (1.6503) + (1E-004j);
C_ep1 = (1.6503) + (4E-004j);
D_ep1 = (-2.3E-003) + (0);
G_ep1 = (1.6502) + (1E-004j);
H_ep1 = (1.042E-005) - (1.02E-006j);
I_ep1 = (2.546E-005) - (3.57E-004j);
J_ep1 = (-1.84E-006) - (1.09E-008j);

% Define frequency range
f_min = 0.2; % Hz
f_max = 10; % Hz
num_points = 100-1; % Number of frequency points, -1 for smoother numbers
frequencies = linspace(f_min, f_max, num_points);

% Compute permittivity epsilon(f)
epsilon_f = B_ep1 + 2 * C_ep1 .* (frequencies .^ D_ep1) + G_ep1 .* (1 - J_ep1 .* (frequencies - H_ep1).^2 - j * 2 * I_ep1 .* frequencies) .^ (-1);
epsilon2_f = imag(epsilon_f)*(2e-1); % Calibrating equation to text data

% Compute permeability (assuming μ = 1 for non-magnetic materials)
mu_f = (frequencies * 0) + 1; % Modify if permeability data is available

% Prepare data for export
data_table = table(frequencies', real(epsilon_f)', imag(epsilon_f)', mu_f', 'VariableNames', {'Frequency_Hz', 'Real_Epsilon', 'Imag_Epsilon', 'Mu'});

% Define file name
file_name = strcat(material_name, '.csv');

% Export to CSV
writetable(data_table, file_name);

% Display message
fprintf('Data exported to %s\n', file_name);

% loglog plot
figure;
loglog(frequencies,(real(epsilon_f)),'b-', 'Linewidth',2); % real epsilon_f in blue
hold on;
loglog(frequencies,(epsilon2_f),'r--', 'Linewidth',2); % imag epsilon_f in red
hold off;

xlim([1e-1, 1e3]);
ylim([1e-4, 10]);

xlabel('Frequency [GHz]', 'FontSize', 12);
ylabel('Epsilon', 'FontSize', 12);
legend({'Re(\epsilon)', 'Im(\epsilon)'}, 'Location', 'Best');
grid on;
title('Real and Imaginary Permittivity vs. Frequency', 'FontSize', 14);