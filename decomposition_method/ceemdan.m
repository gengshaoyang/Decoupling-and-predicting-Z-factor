function [modes]=ceemdan(x,Nstd,NR)

% WARNING: for this code works it is necessary to include in the same
%directoy the file emd.m developed by Rilling and Flandrin.
%This file is available at %http://perso.ens-lyon.fr/patrick.flandrin/emd.html
%We use the default stopping criterion.
%We use the last modification: 3.2007
% 
% This version was run on Matlab 7.10.0 (R2010a)
%----------------------------------------------------------------------
%   INPUTs
%   x: signal to decompose
%   Nstd: noise standard deviation
%   NR: number of realizations
%   MaxIter: maximum number of sifting iterations allowed.
%
%  OUTPUTs
%  modes: contain the obtained modes in a matrix with the rows being the modes        
%   its: contain the sifting iterations needed for each mode for each realization (one row for each realization)
% -------------------------------------------------------------------------

x=x(:)';
desvio_x=std(x);
x=x/desvio_x;

modes=zeros(size(x));
temp=zeros(size(x));
aux=zeros(size(x));
acum=zeros(size(x));
iter=zeros(NR,round(log2(length(x))+5));

for i=1:NR
    white_noise{i}=randn(size(x));%creates the noise realizations
end;

for i=1:NR
    modes_white_noise{i}=emd(white_noise{i});%calculates the modes of white gaussian noise
end;

for i=1:NR %calculates the first mode
    temp=x+Nstd*white_noise{i};
    [temp]=emd(temp);
    temp=temp(1,:);
    aux=aux+temp/NR;
%     iter(i,1)=it;
end;

modes=aux; %saves the first mode
k=1;
aux=zeros(size(x));
acum=sum(modes,1);

while  nnz(diff(sign(diff(x-acum))))>2 %calculates the rest of the modes
    for i=1:NR
        tamanio=size(modes_white_noise{i});
        if tamanio(1)>=k+1
            noise=modes_white_noise{i}(k,:);
            noise=noise/std(noise);
            noise=Nstd*noise;
            try
                [temp]=emd(x-acum+std(x-acum)*noise);
                temp=temp(1,:);
            catch
                it=0;
                temp=x-acum;
            end
        else
            [temp]=emd1(x-acum);
            temp=temp(1,:);
        end
        aux=aux+temp/NR;
%     iter(i,k+1)=it;    
    end
    modes=[modes;aux];
    aux=zeros(size(x));
    acum=zeros(size(x));
    acum=sum(modes,1);
    k=k+1;
end
modes=[modes;(x-acum)];
[a b]=size(modes);
% iter=iter(:,1:a);
modes=modes*desvio_x;
% its=iter;


   