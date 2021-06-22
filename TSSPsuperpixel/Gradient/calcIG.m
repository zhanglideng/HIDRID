function  [IG,alpha]  = calcIG( I,radius )
k=2*radius+1;
sigma=radius/2.0;
weight=zeros(k,k);
i=radius+1;
for y=1:k
    for x=1:k
        if (y-i)^2+(x-i)^2<=radius^2
            weight(y,x)=exp(-((y-i)^2+(x-i)^2)/(2*sigma^2));
        end
     end
end
alpha=calcAngle(I,radius,weight);
[IG,betta]=recalcAngle(I,radius,weight,alpha);
end



