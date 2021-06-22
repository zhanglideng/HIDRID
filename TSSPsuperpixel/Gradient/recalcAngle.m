function [ig,theta] = recalcAngle( I,radius,weight,alpha )
[rows,cols]=size(I);
I=padarray(I,[radius,radius],'symmetric'); 
alpha=padarray(alpha,[radius,radius]); 
igx=zeros(rows,cols);
igy=zeros(rows,cols);
igx2=zeros(rows,cols);
igy2=zeros(rows,cols);
theta=zeros(rows,cols);
k=2*radius+1;
k1=sum(sum(weight(:,1:radius)));
for y=1:rows
    for x=1:cols
        window=calcBilinear_Interpolation(y+radius,x+radius,alpha(y,x),I,radius);
        igl=weight(:,1:radius).*window(:,1:radius);
        igr=weight(:,k-radius+1:k).*window(:,k-radius+1:k);  
        igx(y,x)=sum(igr(:))/k1-sum(igl(:))/k1; %x??????
        igu=weight(1:radius,:).*window(1:radius,:);
        igd=weight(k-radius+1:k,:).*window(k-radius+1:k,:);
        igy(y,x)=sum(igd(:))/k1-sum(igu(:))/k1; %y??????
        igx2(y,x)=igx(y,x)*cos(alpha(y,x))-igy(y,x)*sin(alpha(y,x));%根据角度还原后得到igx2;igy2;这样的igx2，igy2值相似,用于求角度
        igy2(y,x)=igx(y,x)*sin(alpha(y,x))+igy(y,x)*cos(alpha(y,x));
        if igx(y,x)==0
        	theta(y,x)=pi/2;
        else
            theta(y,x)=atan(igy2(y,x)/igx2(y,x));  
        end
    end
end
ig=sqrt(igx.^2+igy.^2);
end


