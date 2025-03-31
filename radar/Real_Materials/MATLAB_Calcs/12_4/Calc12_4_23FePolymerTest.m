clear;
clc;

% Define material properties
material_name = 'VF_12_4_23%FePolymerTest'; % Change this as needed

% Given values for permeability (χ_m) equation 1
B_x1 = (8.41);
C_x1 = (0.52);
D_x1 = (0.19);

% Given values for permittivity (ε(f)) equation 1
B_ep1 = (5.41) + (1.1j);
C_ep1 = (4) - (0.79j);
D_ep1 = (-2E-003) - (2.6E-002j);
E_ep1 = (-0.61) + (0.7j);
F_ep1 = (0.18) + (0.57j);
G_ep1 = (0.12) + (7E-002j);

% Given values for permittivity (ε(f)) equation 2
B_ep2 = (4.48) + (0.16j);
C_ep2 = (1.69) - (0.46j);
D_ep2 = (-0.2) - (7E-002j);
E_ep2 = (-0.62) + (0.11j);
F_ep2 = (3.65) + (8E-002j);
G_ep2 = (29.6) - (23.5j);
H_ep2 = (-115) + (53.4j);

% Volume fraction
f = (0.23);

% Define freq range (GHz)
f_min = 0.2;  % Hz
f_max = 10; % Hz
num_points = 99; % Number of frequency points
frequencies = linspace(f_min, f_max, num_points);

% Compute X1(f)
x_1 = (B_x1 .* (1 - 1i * (frequencies ./ D_x1)) ./ (1 - ((frequencies ./ (C_x1)).^2) - (1i * (frequencies ./ (D_x1))))) + 0;

% Compute ε1(f)
e_1 = B_ep1 + (C_ep1 .* (frequencies .^ D_ep1)) + (E_ep1 .* ((1 - ((frequencies ./ F_ep1).^2) - (2i * (frequencies ./ G_ep1))).^(-1)));

% Compute ε2(f)
e_2 = B_ep2 + (real(C_ep2) .* (frequencies .^ D_ep2)) + (imag(C_ep2) .* (frequencies .^ E_ep2)) + (F_ep2 .* (1 - (frequencies ./ G_ep2).^2 - 2i * (frequencies ./ H_ep2)).^(-1));

% Convert chi to mu
mu_1 = x_1 + 1;

% Use MGT relations to find effective permittivity
e_e = (e_2)*((1 + ((2 * f)*((e_1 - e_2) / (e_1 + 2*e_2)))) / (1 - f*((e_1 - e_2) / (e_1 + 2*e_2))));

% Prep data for export
data_table = table(frequencies', real(e_e)', imag(e_e)', real(mu_1)', imag(mu_1)', 'VariableNames', {'Frequency_GHz', 'Real_Epsilon', 'Imag_Epsilon', 'Real_Mu', 'Imag_Mu'});

% Define file name
file_name = strcat(material_name, '.csv');

% Export to CSV
writetable(data_table, file_name);

% Display message
fprintf('Data exported to %s\n', file_name);

% semilog plot
figure;
semilogx(frequencies,(real(e_e)),'b-', 'Linewidth',2); % real e_e in blue
hold on;
semilogx(frequencies,(imag(e_e)),'r--', 'Linewidth',2); % imag e_e in red
hold off;

xlim([1e-2, 1e1]);
ylim([-2, 10]);

xlabel('Frequency [GHz]', 'FontSize', 12);
ylabel('Epsilon', 'FontSize', 12);
legend({'Re(\epsilon)', 'Im(\epsilon)'}, 'Location', 'Best');
grid on;
title('Real and Imaginary Permittivity vs. Frequency', 'FontSize', 14);

figure;
semilogx(frequencies,(real(x_1)),'b-', 'Linewidth',2); % real mu_e in blue
hold on;
semilogx(frequencies,(imag(x_1)),'r--', 'Linewidth',2); % imag mu_e in red
hold off;

xlim([1e-3, 1e1]);
ylim([-2, 10]);

xlabel('Frequency [GHz]', 'FontSize', 12);
ylabel('X', 'FontSize', 12);
legend({'Re(\mu)', 'Im(\mu)'}, 'Location', 'Best');
grid on;
title('Real and Imaginary Permeability vs. Frequency', 'FontSize', 14);