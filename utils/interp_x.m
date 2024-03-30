function x_interp = interp_x(x,k)
x_interp = [];

% for i = 1:length(x)-1
%     x_segment = linspace(x(i), x(i+1), k);
%     x_interp = [x_interp, spline([x(i), x(i+1)], [0, 0, 0, 0, 0, 0], x_segment)];
% end
% linear
for i = 1:length(x)-1
    x_interp = [x_interp, linspace(x(i), x(i+1), k)];
end
x_interp = x_interp(1:end-1);  % Remove the last element as it's repeated

end