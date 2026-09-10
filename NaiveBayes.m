%% NAIVE BAYES CLASSIFIER
clc
clear all

%% Read apple images

A1=imread('apple_04.jpg');
A2=imread('apple_05.jpg');
A3=imread('apple_06.jpg');
A4=imread('apple_07.jpg');
A5=imread('apple_11.jpg');
A6=imread('apple_12.jpg');
A7=imread('apple_13.jpg');
A8=imread('apple_17.jpg');
A9=imread('apple_19.jpg');

%% Read pear images

P1=imread('pear_01.jpg');
P2=imread('pear_02.jpg');
P3=imread('pear_03.jpg');
P4=imread('pear_09.jpg');

%% Calculate features
% Apples
hsv_value_A1=spalva_color(A1);
metric_A1=apvalumas_roundness(A1);

hsv_value_A2=spalva_color(A2);
metric_A2=apvalumas_roundness(A2);

hsv_value_A3=spalva_color(A3);
metric_A3=apvalumas_roundness(A3);

hsv_value_A4=spalva_color(A4);
metric_A4=apvalumas_roundness(A4);

hsv_value_A5=spalva_color(A5);
metric_A5=apvalumas_roundness(A5);

hsv_value_A6=spalva_color(A6);
metric_A6=apvalumas_roundness(A6);

hsv_value_A7=spalva_color(A7);
metric_A7=apvalumas_roundness(A7);

hsv_value_A8=spalva_color(A8);
metric_A8=apvalumas_roundness(A8);

hsv_value_A9=spalva_color(A9);
metric_A9=apvalumas_roundness(A9);

% Pears
hsv_value_P1=spalva_color(P1);
metric_P1=apvalumas_roundness(P1);

hsv_value_P2=spalva_color(P2);
metric_P2=apvalumas_roundness(P2);

hsv_value_P3=spalva_color(P3);
metric_P3=apvalumas_roundness(P3);

hsv_value_P4=spalva_color(P4);
metric_P4=apvalumas_roundness(P4);


%% TRAINING DATA
% A1, A2, A3, P1, P2

apple_color = [hsv_value_A1 hsv_value_A2 hsv_value_A3];
apple_roundness = [metric_A1 metric_A2 metric_A3];

pear_color = [hsv_value_P1 hsv_value_P2];
pear_roundness = [metric_P1 metric_P2];


%% Calculate mean and standard deviation
% Apples

apple_color_mean = mean(apple_color);
apple_color_std = std(apple_color);

apple_roundness_mean = mean(apple_roundness);
apple_roundness_std = std(apple_roundness);

% Pears

pear_color_mean = mean(pear_color);
pear_color_std = std(pear_color);

pear_roundness_mean = mean(pear_roundness);
pear_roundness_std = std(pear_roundness);


%% Prior probabilities

P_apple = length(apple_color) / ...
    (length(apple_color) + length(pear_color));

P_pear = length(pear_color) / ...
    (length(apple_color) + length(pear_color));


%% TESTING DATA

x1_test = [hsv_value_A4 hsv_value_A5 hsv_value_A6 ...
           hsv_value_A7 hsv_value_A8 hsv_value_A9 ...
           hsv_value_P3 hsv_value_P4];

x2_test = [metric_A4 metric_A5 metric_A6 ...
           metric_A7 metric_A8 metric_A9 ...
           metric_P3 metric_P4];

T_test = [1;1;1;1;1;1;-1;-1];


%% Classify test samples

E_test = zeros(8,1);

for i = 1:8

    new_color = x1_test(i);
    new_roundness = x2_test(i);
    % Probability assuming APPLE
    P_color_apple = gaussian_probability( ...
        new_color, apple_color_mean, apple_color_std);
    P_roundness_apple = gaussian_probability( ...
        new_roundness, apple_roundness_mean, apple_roundness_std);
    probability_apple = ...
        P_color_apple * ...
        P_roundness_apple * ...
        P_apple;

    % Probability assuming PEAR
    P_color_pear = gaussian_probability( ...
        new_color, pear_color_mean, pear_color_std);
    P_roundness_pear = gaussian_probability( ...
        new_roundness, pear_roundness_mean, pear_roundness_std);
    probability_pear = ...
        P_color_pear * ...
        P_roundness_pear * ...
        P_pear;

    % Classification

    if probability_apple > probability_pear
        y = 1;
    else
        y = -1;
    end

    % Error
    E_test(i) = T_test(i) - y;
    fprintf('Sample %d: ', i);
    if y == 1
        fprintf('Apple');
    else
        fprintf('Pear');
    end

    fprintf(' | Error = %d\n', E_test(i));

end

%% Total error

total_error = sum(abs(E_test));

fprintf('\nTotal classification error: %d\n', total_error);