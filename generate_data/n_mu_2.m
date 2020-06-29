clear all
close all
clc

x = linspace(0, 1, 256)';

h = 1/200;
t = 0 : h : 1;
t = t(2 : end);
t = t(1 : 2 : end);

sigma = 1e-4;

%% Build training snapshot matrix
x0 = linspace(0.025, 0.25, 21);
A = linspace(0.5, 1, 21);
S = [];

for k = 1 : length(A)
    for i = 1 : length(x0)
        for j = 1 : length(t)
            u = @(x, t) 0 * (( x - t ) < x0(i)) + A(k) * (( x - t ) >= x0(i));
            S = [S u(x, t(j))];
        end
    end
end

%% Build testing snapshot matrix
x0_test = linspace(0.025 + 0.01125/2, 0.25 - 0.01125/2, 20);
A_test = linspace(0.5 + 0.025/2, 1 - 0.025/2, 20);
S = [];

for k = 1 : length(A_test)
    for i = 1 : length(x0_test)
        for j = 1 : length(t)
            u = @(x, t) 0 * (( x - t ) < x0_test(i)) + A_test(k) * (( x - t ) >= x0_test(i));
            S = [S u(x, t(j))];
        end
    end
end

%% Build training parameter matrix
l = length(x0) * length(A) * length(t);
I = zeros(l, 3);

for i = 1 : length(x0) * length(A)
    for j = 1 : length(t)
        I(length(t) * (i - 1) + j, 3) = t(j);
    end
end

for k = 1 : length(A)
    for i = 1 : length(x0)
        for j = 1 : length(t)
            I((k - 1) * length(x0) * length(t) + length(t) * (i - 1) + j , 1 : 2) = [A(k), x0(i)];
        end
    end
end

%% Build testing parameter matrix
l = length(x0_test) * length(A_test) * length(t);
I = zeros(l, 3);

for i = 1 : length(x0_test) * length(A_test)
    for j = 1 : length(t)
        I(length(t) * (i - 1) + j, 3) = t(j);
    end
end

for k = 1 : length(A_test)
    for i = 1 : length(x0_test)
        for j = 1 : length(t)
            I((k - 1) * length(x0_test) * length(t) + length(t) * (i - 1) + j , 1 : 2) = [A_test(k), x0_test(i)];
        end
    end
end