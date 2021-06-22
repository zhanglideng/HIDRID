function alpha = calcAngle( I,radius,weight )
[rows,cols]=size(I);
I=padarray(I,[radius,radius],'symmetric'); 
igx=zeros(rows,cols);
igy=zeros(rows,cols);
alpha=zeros(rows,cols);
k=2*radius+1;
k1=sum(sum(weight(:,1:radius)));
parfor y=1:rows
        for x=1:cols
            ymin=y;
            ymax=y+2*radius;
            xmin=x;
            xmax=x+2*radius;
            window=I(ymin:ymax,xmin:xmax); 
            igl=weight(:,1:radius).*window(:,1:radius);
            igr=weight(:,k-radius+1:k).*window(:,k-radius+1:k);  
            igx(y,x)=sum(igr(:))/k1-sum(igl(:))/k1; %x??????
            igu=weight(1:radius,:).*window(1:radius,:);
            igd=weight(k-radius+1:k,:).*window(k-radius+1:k,:);
            igy(y,x)=sum(igd(:))/k1-sum(igu(:))/k1; %y??????
            if igx(y,x)==0
                alpha(y,x)=pi/2;
            else
                alpha(y,x)=-atan(igy(y,x)/igx(y,x));  
            end
        end
end
end








