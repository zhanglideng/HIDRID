function window = calcBilinear_Interpolation(j,i,alpha,I,radius)
k=2*radius+1;
window=zeros(k,k);
for n=j-radius:j+radius
    for m=i-radius:i+radius
        if (i-m)^2+(j-n)^2<=radius^2 
            y=(n-j)*cos(alpha)-(m-i)*sin(alpha)+j;
            x=(n-j)*sin(alpha)+(m-i)*cos(alpha)+i;
            p1=I(floor(y),floor(x)); %up_left
            p2=I(floor(y),ceil(x));  %up_right
            p3=I(ceil(y),floor(x));   %down_left
            p4=I(ceil(y),ceil(x));  %down_right
            s=x-floor(x);
            t=y-floor(y);
            b=n-j+radius+1;
            a=m-i+radius+1;
            window(b,a)=(1-t)*(1-s)*p1+(1-t)*s*p2+(1-s)*t*p3+p4*s*t;
        end    
    end
end

end

