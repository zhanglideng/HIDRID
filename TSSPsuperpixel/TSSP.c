#include<mex.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
void rgbtolab(int* rin, int* gin, int* bin, int sz, double* lvec, double* avec, double* bvec)
{
    int i; int sR, sG, sB;
    double R,G,B;
    double X,Y,Z;
    double r, g, b;
    double xr, yr, zr;
    const double epsilon = 0.008856;
    const double kappa   = 903.3;		
    
    const double Xr = 0.950456;
    const double Yr = 1.0;		
    const double Zr = 1.088754;
    double fx, fy, fz;
    double lval,aval,bval;
    
    for(i = 0; i < sz; i++)
    {
        sR = rin[i]; sG = gin[i]; sB = bin[i];
        R = sR/255.0;
        G = sG/255.0;
        B = sB/255.0;
        
        if(R <= 0.04045)	r = R/12.92;
        else				r = pow((R+0.055)/1.055,2.4);
        if(G <= 0.04045)	g = G/12.92;
        else				g = pow((G+0.055)/1.055,2.4);
        if(B <= 0.04045)	b = B/12.92;
        else				b = pow((B+0.055)/1.055,2.4);
        
        X = r*0.4124564 + g*0.3575761 + b*0.1804375;
        Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
        Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
        

        xr = X/Xr;
        yr = Y/Yr;
        zr = Z/Zr;
        
        if(xr > epsilon)	fx = pow(xr, 1.0/3.0);
        else				fx = (kappa*xr + 16.0)/116.0;
        if(yr > epsilon)	fy = pow(yr, 1.0/3.0);
        else				fy = (kappa*yr + 16.0)/116.0;
        if(zr > epsilon)	fz = pow(zr, 1.0/3.0);
        else				fz = (kappa*zr + 16.0)/116.0;
        
        lval = 116.0*fy-16.0;
        aval = 500.0*(fx-fy);
        bval = 200.0*(fy-fz);
        
        lvec[i] = lval; avec[i] = aval; bvec[i] = bval;
    }
}
void getLABXYSeeds(int STEP, int width, int height, int* seedIndices, int* numseeds)
{
    const bool hexgrid = false;
	int n;
    int xstrips, ystrips;
    int xerr, yerr;
    double xerrperstrip,yerrperstrip;
    int xoff,yoff;
    int x,y;
    int xe,ye;
    int seedx,seedy;
    int i;

	xstrips = (0.5+(double)(width)/(double)(STEP));
	ystrips = (0.5+(double)(height)/(double)(STEP));
    
    xerr = width  - STEP*xstrips;if(xerr < 0){xstrips--;xerr = width - STEP*xstrips;}
    yerr = height - STEP*ystrips;if(yerr < 0){ystrips--;yerr = height- STEP*ystrips;}
    
	xerrperstrip = (double)(xerr)/(double)(xstrips);
	yerrperstrip = (double)(yerr)/(double)(ystrips);
    
	xoff = STEP/2;
	yoff = STEP/2;
    
    n = 0;
	for( y = 0; y < ystrips; y++ )
	{
		ye = y*yerrperstrip;
		for( x = 0; x < xstrips; x++ )
		{
			xe = x*xerrperstrip;
            seedx = (x*STEP+xoff+xe);
            if(hexgrid){ seedx = x*STEP+(xoff<<(y&0x1))+xe; if(seedx >= width)seedx = width-1; }
            seedy = (y*STEP+yoff+ye);
            i = seedy*width + seedx;
			seedIndices[n] = i;
			n++;
		}
	}
    *numseeds = n;
}
void EnforceSuperpixelConnectivity(int* labels, int width, int height, int numSuperpixels,int* nlabels, int* finalNumberOfLabels)
{
	
	
    int i,j,k;
    int n,c,count;
    int x,y;
    int ind;
    int oindex, adjlabel;
    int label;
    const int dx4[4] = {-1,  0,  1,  0};
	const int dy4[4] = { 0, -1,  0,  1};
    const int sz = width*height;
    const int SUPSZ = sz/numSuperpixels;
    int* xvec = mxMalloc(sizeof(int)*SUPSZ*10);
	int* yvec = mxMalloc(sizeof(int)*SUPSZ*10);

	for( i = 0; i < sz; i++ ) nlabels[i] = -1;
    oindex = 0;
    adjlabel = 0;
    label = 0;
	for( j = 0; j < height; j++ )
	{
		for( k = 0; k < width; k++ )
		{
			if( 0 > nlabels[oindex] )
			{
				nlabels[oindex] = label;
				
				xvec[0] = k;
				yvec[0] = j;
				
				{for( n = 0; n < 4; n++ )
				{
					int x = xvec[0] + dx4[n];
					int y = yvec[0] + dy4[n];
					if( (x >= 0 && x < width) && (y >= 0 && y < height) )
					{
						int nindex = y*width + x;
						if(nlabels[nindex] >= 0) adjlabel = nlabels[nindex];
					}
				}}
                
				count = 1;
				for( c = 0; c < count; c++ )
				{
					for( n = 0; n < 4; n++ )
					{
						x = xvec[c] + dx4[n];
						y = yvec[c] + dy4[n];
                        
						if( (x >= 0 && x < width) && (y >= 0 && y < height) )
						{
							int nindex = y*width + x;
                            
							if( 0 > nlabels[nindex] && labels[oindex] == labels[nindex] )
							{
								xvec[count] = x;
								yvec[count] = y;
								nlabels[nindex] = label;
								count++;
							}
						}
                        
					}
				}
				
				if(count <= SUPSZ >> 2)
				{
					for( c = 0; c < count; c++ )
					{
                        ind = yvec[c]*width+xvec[c];
						nlabels[ind] = adjlabel;
					}
					label--;
				}
				label++;
			}
			oindex++;
		}
	}
	*finalNumberOfLabels = label;
    
	mxFree(xvec);
	mxFree(yvec);
}
void PerformSuperpixelSLIC(double* lvec, double* avec, double* bvec, double* kseedsl, double* kseedsa, double* kseedsb, double* kseedsx, double* kseedsy, int width, int height, int numseeds, int* klabels, int STEP, double compactness,double ** MapL,double ** MapA,double ** MapB,double ** Amptitude,double ** Xita,double ** Gray,int radius,int gamma)
{
	
	int cx,cy;
	int idx,idy,idz;
	int windows=2*radius+1;
	int flag;
	double alpha;
	double WQ,WQSum;
	double WPColor,WPXita,WPSum;
    double pQL,pQA,pQB;
	double pL,pA,pB;
	double qL,qA,qB;
	double pXita,qXita;
	double pGray,qGray;
    double distP,distQ,distC;
	double px,py,ps,pt,p1,p2,p3,p4;
	
	
	int dm,mm;
	double dx,dy;
	double xx,yy;
	double igmax;
	
	
	double ** circle = (double **)mxMalloc(sizeof(double *)*windows);
	for(idx=0;idx<windows;idx++)
    {
		circle[idx] = (double *)mxMalloc(sizeof(double)*windows);
        for(idy=0;idy<windows;idy++)
        {
			if(((idx-radius)*(idx-radius)+(idy-radius)*(idy-radius)>radius*radius)||(idx>idy)||(idx+idy>windows)) 
				circle[idx][idy]=0;
			else
				circle[idx][idy]=1;
        }
    }
	
	
    int x1, y1, x2, y2;
	double sL, sA, sB;
	double dist;
	double distxy;
    int itr;
    int n;
    int x,y;
    int ind;
    int r,c;
    int i,k;
    int sz = width*height;
	const int numk = numseeds;
	int offset = STEP;
    double* clustersize = mxMalloc(sizeof(double)*numk);
    double* inv         = mxMalloc(sizeof(double)*numk);
    double* sigmal      = mxMalloc(sizeof(double)*numk);
    double* sigmaa      = mxMalloc(sizeof(double)*numk);
    double* sigmab      = mxMalloc(sizeof(double)*numk);
    double* sigmax      = mxMalloc(sizeof(double)*numk);
    double* sigmay      = mxMalloc(sizeof(double)*numk);
	double* sigmaIG      = mxMalloc(sizeof(double)*numk);
    double* distvec     = mxMalloc(sizeof(double)*sz);
	double invwt = 1.0/((STEP/compactness)*(STEP/compactness));

	for( itr = 0; itr < 10; itr++ )
	{
		for(i = 0; i < sz; i++){distvec[i] = DBL_MAX;}
		for( n = 0; n < numk; n++ )
		{
            x1 = kseedsx[n]-offset; if(x1 < 0) x1 = 0;
            y1 = kseedsy[n]-offset; if(y1 < 0) y1 = 0;
            x2 = kseedsx[n]+offset; if(x2 > width)  x2 = width;
            y2 = kseedsy[n]+offset; if(y2 > height) y2 = height;
			for( y = y1; y < y2; y++ )
			{
				for( x = x1; x < x2; x++ )
				{                 
					pL=MapL[x+radius][y+radius];
					pA=MapA[x+radius][y+radius];
					pB=MapB[x+radius][y+radius];
					
					sL = kseedsl[n];
					sA = kseedsa[n];
					sB = kseedsb[n];
					
					distP=distQ=0;
					WPSum=WQSum=0;
					pQL=pQA=pQB=0;
					cx=round(kseedsx[n]);
                    cy=round(kseedsy[n]);
					pGray=Gray[x+radius][y+radius];		
					pXita=Xita[x+radius][y+radius];
					alpha=-atan2(y-kseedsy[n],x-kseedsx[n]);
					for( idy = 0; idy < windows; idy++ )
                    {
                        for( idx = 0; idx < windows; idx++ )
                        {
							qL=MapL[x+idx][y+idy];
							qA=MapA[x+idx][y+idy];
							qB=MapB[x+idx][y+idy];
							
							WQ=0;
							if((idx-radius)*(idx-radius)+(idy-radius)*(idy-radius)<=radius*radius)
                            {
								py=(idx-radius)*cos(alpha)-(idy-radius)*sin(alpha)+radius;
								px=(idx-radius)*sin(alpha)+(idy-radius)*cos(alpha)+radius;
								p1=circle[(int)floor(py)][(int)floor(px)]; 
								p2=circle[(int)floor(py)][(int)ceil(px)];  
								p3=circle[(int)ceil(py)][(int)floor(px)];   
								p4=circle[(int)ceil(py)][(int)ceil(px)];
								ps=px-floor(px);
								pt=py-floor(py);
								
								WQ=(1-pt)*(1-ps)*p1+(1-pt)*ps*p2+(1-ps)*pt*p3+p4*ps*pt;
								WQSum+=WQ;
							}
							qGray=Gray[x+idx][y+idy];
							qXita=Xita[x+idx][y+idy];
							WPColor=exp(-(pGray-qGray)*(pGray-qGray)/(2*40*40));
							WPXita=exp(-(pXita-qXita)*(pXita-qXita)/(2*0.5*0.5));
							WPSum+=WPColor*WPXita*WQ;
							distP+=WPColor*WPXita*WQ*((qL-sL)*(qL-sL)+(qA-sA)*(qA-sA)+(qB-sB)*(qB-sB));
							
							pQL+=qL*WQ;
							pQA+=qA*WQ;
							pQB+=qB*WQ;
						}	
					}
					distP=distP/WPSum;
					distQ=(pQL/WQSum-sL)*(pQL/WQSum-sL)+
						  (pQA/WQSum-sA)*(pQA/WQSum-sA)+
						  (pQB/WQSum-sB)*(pQB/WQSum-sB);
					dist=distP<distQ?distP:distQ;
					distxy =(x-kseedsx[n])*(x-kseedsx[n])+(y-kseedsy[n])*(y-kseedsy[n]);
					dist += distxy*invwt;

					
					if(abs(x-(int)kseedsx[n])>abs(y-(int)kseedsy[n]))
						dm=abs(x-(int)(kseedsx[n]));
					else
						dm=abs(y-(int)(kseedsy[n]));
					dx=(kseedsx[n]-x)*1.0/dm;
					dy=(kseedsy[n]-y)*1.0/dm;	
					for(igmax=0,xx=x,yy=y,mm=0;mm<dm;mm++)
					{
						if(xx>width-1||xx<0||yy<0||yy>height-1)
							break;
						if(igmax<Amptitude[(int)xx][(int)yy])
							igmax=Amptitude[(int)xx][(int)yy];
						xx+=dx;
						yy+=dy;
					}
					dist*=1+gamma*igmax;
					
					
					
					i = y*width + x;						
					if(dist < distvec[i])
					{
						distvec[i] = dist;
						klabels[i]  = n;
					}
				}
			}
		}
		
        for(k = 0; k < numk; k++)
        {
            sigmal[k] = 0;
            sigmaa[k] = 0;
            sigmab[k] = 0;
            sigmax[k] = 0;
            sigmay[k] = 0;
            clustersize[k] = 0;
        }
		ind = 0;
        for( r = 0; r < height; r++ )
        {
            for( c = 0; c < width; c++ )
            {
                if(klabels[ind] >= 0)
                {
                    sigmal[klabels[ind]] += lvec[ind];
                    sigmaa[klabels[ind]] += avec[ind];
                    sigmab[klabels[ind]] += bvec[ind];
                    sigmax[klabels[ind]] += c;
                    sigmay[klabels[ind]] += r;
                    clustersize[klabels[ind]] += 1.0;
                }
                ind++;
            }
        }
        
		{for( k = 0; k < numk; k++ )
		{
			if( clustersize[k] <= 0 ) clustersize[k] = 1;
			inv[k] = 1.0/clustersize[k];
		}}
		
		{for( k = 0; k < numk; k++ )
		{
			kseedsl[k] = sigmal[k]*inv[k];
			kseedsa[k] = sigmaa[k]*inv[k];
			kseedsb[k] = sigmab[k]*inv[k];
			kseedsx[k] = sigmax[k]*inv[k];
			kseedsy[k] = sigmay[k]*inv[k];
		}}
	}
    mxFree(sigmal);
    mxFree(sigmaa);
    mxFree(sigmab);
    mxFree(sigmax);
    mxFree(sigmay);
    mxFree(clustersize);
    mxFree(inv);
    mxFree(distvec);
    for (i=0; i<windows; i++)
		mxFree(circle[i]);

}
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
	
	int i,j,k;
	int x,y;
    int ii;
	
	
	int numSuperpixels = mxGetScalar(prhs[0]);
	double compactness = 20;
	
	
	const mwSize* dims=mxGetDimensions(prhs[1]);
	int width = dims[1];
	int height = dims[0];
	int sz=width*height;
    unsigned char* imgbytes = (double * )mxGetData(prhs[1]);
    int* rin    = mxMalloc( sizeof(int)      * sz ) ;
    int* gin    = mxMalloc( sizeof(int)      * sz ) ;
    int* bin    = mxMalloc( sizeof(int)      * sz ) ;
	int* klabels = mxMalloc( sizeof(int)         * sz );
    int* clabels = mxMalloc( sizeof(int)         * sz ); 
    int* seedIndices = mxMalloc( sizeof(int)     * sz );
    double* lvec    = mxMalloc( sizeof(double)      * sz ) ;
    double* avec    = mxMalloc( sizeof(double)      * sz ) ;
    double* bvec    = mxMalloc( sizeof(double)      * sz ) ;
	for(x = 0, ii = 0; x < width; x++)
	{
		for(y = 0; y < height; y++)
		{
			i = y*width+x;
			rin[i] = imgbytes[ii];
			gin[i] = imgbytes[ii+sz];
			bin[i] = imgbytes[ii+sz+sz];
			ii++;
		}
	}
	rgbtolab(rin,gin,bin,sz,lvec,avec,bvec);
	
	
    const mwSize* dims2=mxGetDimensions(prhs[2]);
	int width2 = dims2[1];
	int height2 = dims2[0];
	int sz2=width2*height2;
    double* imgbytes2 = (double * )mxGetData(prhs[2]);
    double** MapL,** MapA,** MapB;
    MapL = (double **)mxMalloc(sizeof(double *)*width2);
    MapA = (double **)mxMalloc(sizeof(double *)*width2);
    MapB = (double **)mxMalloc(sizeof(double *)*width2);
    for (i=0,ii=0; i<width2; i++)
    {
		MapL[i] = (double *)mxMalloc(sizeof(double *)*height2);
		MapA[i] = (double *)mxMalloc(sizeof(double *)*height2);
        MapB[i] = (double *)mxMalloc(sizeof(double *)*height2);
        for (j=0; j<height2; j++)
        {
            MapL[i][j] = imgbytes2[ii];
            MapA[i][j] = imgbytes2[ii+sz2];
            MapB[i][j] = imgbytes2[ii+sz2+sz2];       
            ++ii;        
        }
    }
	
    const mwSize* dims3=mxGetDimensions(prhs[3]);
	int width3 = dims3[1];
	int height3 = dims3[0];
	double* imgbytes3 = (double * )mxGetData(prhs[3]);
    double** Gray= (double **)mxMalloc(sizeof(double *)*width3);
    for (i=0,ii=0; i<width3; i++)
    {
		Gray[i] = (double *)mxMalloc(sizeof(double *)*height3);
        for (j=0; j<height3; j++)
        {
            Gray[i][j] = imgbytes3[ii];   
            ++ii;        
        }
    }
	
	
	const mwSize* dims4=mxGetDimensions(prhs[4]);
	int width4 = dims4[1];
	int height4 = dims4[0];
	double* imgbytes4 = (double * )mxGetData(prhs[4]);
	double** Amptitude = (double **)mxMalloc(sizeof(double *)*width4);
	for (i=0,ii=0; i<width4; i++)
    {
		Amptitude[i] = (double *)mxMalloc(sizeof(double *)*height4);
        for (j=0; j<height4; j++)
        {
            Amptitude[i][j] = imgbytes4[ii];   
            ++ii;        
        }
    }
	
	
    const mwSize* dims5=mxGetDimensions(prhs[5]);
	int width5 = dims5[1];
	int height5 = dims5[0];
    double* imgbytes5 = (double * )mxGetData(prhs[5]);
    double** Xita= (double **)mxMalloc(sizeof(double *)*width5);
    for (i=0,ii=0; i<width5; i++)
    {
		Xita[i] = (double *)mxMalloc(sizeof(double *)*height5);
        for (j=0; j<height5; j++)
        {
            Xita[i][j] = imgbytes5[ii];   
            ++ii;        
        }
    }

	
	int radius  = mxGetScalar(prhs[6]);
	int gamma = mxGetScalar(prhs[7]);
	
	
    int numseeds;
    int finalNumberOfLabels;
    int step = sqrt((double)(sz)/(double)(numSuperpixels))+0.5;
    getLABXYSeeds(step,width,height,seedIndices,&numseeds);
    double* kseedsx    = mxMalloc( sizeof(double)      * numseeds ) ;
    double* kseedsy    = mxMalloc( sizeof(double)      * numseeds ) ;
    double* kseedsl    = mxMalloc( sizeof(double)      * numseeds ) ;
    double* kseedsa    = mxMalloc( sizeof(double)      * numseeds ) ;
    double* kseedsb    = mxMalloc( sizeof(double)      * numseeds ) ;
    for(k = 0; k < numseeds; k++)
    {
        kseedsx[k] = seedIndices[k]%width;
        kseedsy[k] = seedIndices[k]/width;
        kseedsl[k] = lvec[seedIndices[k]];
        kseedsa[k] = avec[seedIndices[k]];
        kseedsb[k] = bvec[seedIndices[k]];
    }
    PerformSuperpixelSLIC(lvec, avec, bvec, kseedsl,kseedsa,kseedsb,kseedsx,kseedsy,width,height,numseeds,klabels,step,compactness,MapL,MapA,MapB,Amptitude,Xita,Gray,radius,gamma);
    EnforceSuperpixelConnectivity(klabels,width,height,numSuperpixels,clabels,&finalNumberOfLabels);
    
	
    plhs[0] = mxCreateNumericMatrix(height,width,mxINT32_CLASS,mxREAL);
    int* outlabels = mxGetData(plhs[0]);
    for(x = 0, ii = 0; x < width; x++)
    {
        for(y = 0; y < height; y++)
        {
            i = y*width+x;
            outlabels[ii] = clabels[i];
            ii++;
        }
    }
    plhs[1] = mxCreateNumericMatrix(1,1,mxINT32_CLASS,mxREAL);
    int* outputNumSuperpixels = (int*)mxGetData(plhs[1]);
    *outputNumSuperpixels = finalNumberOfLabels;
	
    
    mxFree(rin);
    mxFree(gin);
    mxFree(bin);
    mxFree(lvec);
    mxFree(avec);
    mxFree(bvec);
    mxFree(klabels);
    mxFree(clabels);
    mxFree(seedIndices);
    mxFree(kseedsx);
    mxFree(kseedsy);
    mxFree(kseedsl);
    mxFree(kseedsa);
    mxFree(kseedsb); 
    for (i=0; i<width2; i++)
    {
       	mxFree(MapL[i]);
        mxFree(MapA[i]);
        mxFree(MapB[i]);
    }
	for (i=0; i<width3; i++)
		mxFree(Gray[i]);
	for (i=0; i<width4; i++)
		mxFree(Amptitude[i]);   	
	for (i=0; i<width5; i++)
		mxFree(Xita[i]);

}
