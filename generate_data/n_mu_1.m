clear all
close all
clc

x = linspace(0, 1, 256)';

h = 1 / 200;
t = 0 : h : 1;
t = t(2 : end);

sigma = 1e-4;

%% Build training snapshot matrix
c = linspace(0.775, 1.25, 20);
S = [];

for i = 1 : length(c)
    for j = 1 : length(t)
        u = @(x, t) (1 / sqrt(2 * pi * sigma)) * exp(-(x - c(i) * t).^2/(2 * sigma));
        S = [S u(x, t(j))];
    end
end

%% Build testing snapshot matrix
c_test = 0.775 + 0.0125 : 0.0250 : 1.25 - 0.0125;
S = [];

for i = 1 : length(c_test)
    for j = 1 : length(t)
        u = @(x, t) (1 / sqrt(2 * pi * sigma)) * exp(-(x - c_test(i) * t).^2/(2 * sigma));
        S = [S u(x, t(j))];
    end
end

%% Build training parameter matrix
l = length(c) * length(t);
I = zeros(l, 2);

for i = 1 : length(c)
    for j = 1 : length(t)
        I(length(t) * (i - 1) + j, :) = [c(i), t(j)];
    end
end

%% Build testing parameter matrix
l = length(c_test) * length(t);
I = zeros(l, 2);

for i = 1 : length(c_test)
    for j = 1 : length(t)
        I(length(t) * (i - 1) + j, :) = [c_test(i), t(j)];
    end
end