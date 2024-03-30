function [xx,yy] = interp_y(x,y,k)
xx = [];
yy = [];

for i = 1:k:length(x)-k
    x1 = x(i:i+k);
    y1 = y(i:i+k);
    x2 = interp_x(x1,8)';
    y2 = interp1(x1,y1,x2,'makima');
    xx = [xx;x2];
    yy = [yy;y2];
end

end