function [Amptitude,Xita ] = HybridGradient( img,radius )
    img = im2double(img);
	[Rx,Ry] = gradient(img(:,:,1));
    [Gx,Gy] = gradient(img(:,:,2));
    [Bx,By] = gradient(img(:,:,2));
	gR = sqrt(Rx.^2+Ry.^2);
    gG = sqrt(Gx.^2+Gy.^2);
    gB = sqrt(Bx.^2+By.^2);
    Ig1 = sqrt(gR.^2+gG.^2+gB.^2);
    Ig1 = (Ig1-min(min(Ig1)))/(max(max(Ig1))-min(min(Ig1)));
    
%     L=img;
%     [width,height,~]=size(img);
%     img = padarray(img,[radius,radius],'symmetric');
%     for k=1:3
%         for j=1:height
%             for i=1:width
%                 window=img(i:i+2*radius,j:j+2*radius,k);
%                 if((max(max(window))-min(min(window)))>0.1)
%                     L(i,j,k)=(img(i,j,k)-min(min(window)))/(max(max(window))-min(min(window))+0.01);
%                 end
%             end
%         end
%     end
    [IgR,XitaR] = calcIG(img(:,:,1),radius);
    [IgG,XitaG] = calcIG(img(:,:,2),radius);
    [IgB,XitaB] = calcIG(img(:,:,3),radius);
    Ig2 = sqrt(IgR.^2+IgG.^2+IgB.^2);
    Ig2 = (Ig2-min(min(Ig2)))/(max(max(Ig2))-min(min(Ig2)));
    Xita = (XitaR+XitaG+XitaB)./3;
    Amptitude = Ig1.*Ig2;
    Amptitude =(Amptitude-min(min(Amptitude)))/(max(max(Amptitude))-min(min(Amptitude))).*255;
    Xita = padarray(Xita,[radius,radius],'symmetric');
end

