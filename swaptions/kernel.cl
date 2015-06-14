// HJM_type.h
#define FTYPE double

typedef struct
{
    int Id;
    FTYPE dSimSwaptionMeanPrice;
    FTYPE dSimSwaptionStdError;
    FTYPE dStrike;
    FTYPE dCompounding;
    FTYPE dMaturity;
    FTYPE dTenor;
    FTYPE dPaymentInterval;
    int iN;
    FTYPE dYears;
    int iFactors;
    FTYPE pdYield[SIZE];
    FTYPE ppdFactors[][SIZE];
} parm;

// HJM.h
FTYPE RanUnif( long *s );
FTYPE CumNormalInv( FTYPE u );
void icdf_SSE(const int N, FTYPE *in, FTYPE *out);
void icdf_baseline(const int N, FTYPE *in, FTYPE *out);
int HJM_SimPath_Forward_SSE(FTYPE **ppdHJMPath, int iN, int iFactors, FTYPE dYears, FTYPE *pdForward, FTYPE *pdTotalDrift,
        FTYPE **ppdFactors, long *lRndSeed);
int Discount_Factors_SSE(FTYPE *pdDiscountFactors, int iN, FTYPE dYears, FTYPE *pdRatePath);
int Discount_Factors_opt(FTYPE *pdDiscountFactors, int iN, FTYPE dYears, FTYPE *pdRatePath);
int HJM_SimPath_Forward_Blocking_SSE(FTYPE **ppdHJMPath, int iN, int iFactors, FTYPE dYears, FTYPE *pdForward, FTYPE *pdTotalDrift,
        FTYPE **ppdFactors, long *lRndSeed, int BLOCKSIZE);
int HJM_SimPath_Forward_Blocking(FTYPE **ppdHJMPath, int iN, int iFactors, FTYPE dYears, FTYPE *pdForward, FTYPE *pdTotalDrift,
        FTYPE **ppdFactors, long *lRndSeed, int BLOCKSIZE);

int Discount_Factors_Blocking(FTYPE *pdDiscountFactors, int iN, FTYPE dYears, FTYPE *pdRatePath, int BLOCKSIZE);
int Discount_Factors_Blocking_SSE(FTYPE *pdDiscountFactors, int iN, FTYPE dYears, FTYPE *pdRatePath, int BLOCKSIZE);

int HJM_Swaption_Blocking_SSE(FTYPE *pdSwaptionPrice, //Output vector that will store simulation results in the form:
        //Swaption Price
        //Swaption Standard Error
        //Swaption Parameters
        FTYPE dStrike,
        FTYPE dCompounding,     //Compounding convention used for quoting the strike (0 => continuous,
        //0.5 => semi-annual, 1 => annual).
        FTYPE dMaturity,      //Maturity of the swaption (time to expiration)
        FTYPE dTenor,      //Tenor of the swap
        FTYPE dPaymentInterval, //frequency of swap payments e.g. dPaymentInterval = 0.5 implies a swap payment every half
        //year
        //HJM Framework Parameters (please refer HJM.cpp for explanation of variables and functions)
        int iN,
        int iFactors,
        FTYPE dYears,
        FTYPE *pdYield,
        FTYPE **ppdFactors,
        //Simulation Parameters
        long iRndSeed,
        long lTrials, int blocksize, int tid);

int HJM_Swaption_Blocking(FTYPE *pdSwaptionPrice, //Output vector that will store simulation results in the form:
        //Swaption Price
        //Swaption Standard Error
        //Swaption Parameters
        FTYPE dStrike,
        FTYPE dCompounding,     //Compounding convention used for quoting the strike (0 => continuous,
        //0.5 => semi-annual, 1 => annual).
        FTYPE dMaturity,      //Maturity of the swaption (time to expiration)
        FTYPE dTenor,      //Tenor of the swap
        FTYPE dPaymentInterval, //frequency of swap payments e.g. dPaymentInterval = 0.5 implies a swap payment every half
        //year
        //HJM Framework Parameters (please refer HJM.cpp for explanation of variables and functions)
        int iN,
        int iFactors,
        FTYPE dYears,
        FTYPE *pdYield,
        FTYPE **ppdFactors,
        //Simulation Parameters
        long iRndSeed,
        long lTrials, int blocksize, int tid);

// nr_routines.h
int      choldc(FTYPE **a, int n);
void     gaussj(FTYPE **a, int n, FTYPE **b, int m);
void     nrerror( char error_text[] );
int      *ivector(long nl, long nh);
void     free_ivector(int *v, long nl, long nh);
FTYPE   *dvector( long nl, long nh );
void     free_dvector( FTYPE *v, long nl, long nh );
FTYPE   **dmatrix( long nrl, long nrh, long ncl, long nch );
void     free_dmatrix( FTYPE **m, long nrl, long nrh, long ncl, long nch );

// HJM_Securities.h
int HJM_SimPath_Yield(FTYPE **ppdHJMPath, int iN, int iFactors, FTYPE dYears, FTYPE *pdYield, FTYPE **ppdFactors,
        long *lRndSeed);
int HJM_SimPath_Forward(FTYPE **ppdHJMPath, int iN, int iFactors, FTYPE dYears, FTYPE *pdForward, FTYPE *pdTotalDrift,
        FTYPE **ppdFactors, long *iSeed);
int HJM_Yield_to_Forward(FTYPE *pdForward, int iN, FTYPE *pdYield);
int HJM_Factors(FTYPE **ppdFactors,int iN, int iFactors, FTYPE *pdVol, FTYPE **ppdFacBreak);
int HJM_Drifts(FTYPE *pdTotalDrift, FTYPE **ppdDrifts, int iN, int iFactors, FTYPE dYears, FTYPE **ppdFactors);
int HJM_Correlations(FTYPE **ppdHJMCorr, int iN, int iFactors, FTYPE **ppdFactors);
int HJM_Forward_to_Yield(FTYPE *pdYield, int iN, FTYPE *pdForward);
int Discount_Factors(FTYPE *pdDiscountFactors, int iN, FTYPE dYears, FTYPE *pdRatePath);
FTYPE dMax( FTYPE dA, FTYPE dB );

__kernel void kernel_func(__global parm* swaptions, int nSwaptions) {
    int tid = get_global_id(0);
    int nWorkItems = get_global_size(0);
    FTYPE pdSwaptionPrice[2];
    int chunksize = nSwaptions/nWorkItems;
    int beg = tid*chunksize;
    int end = (tid+1)*chunksize;
    if(tid == nWorkItems -1)
        end = nSwaptions;

    for(int i=beg; i < end; i++) {
        /*
           int iSuccess = HJM_Swaption_Blocking(pdSwaptionPrice,  swaptions[i].dStrike,
           swaptions[i].dCompounding, swaptions[i].dMaturity,
           swaptions[i].dTenor, swaptions[i].dPaymentInterval,
           swaptions[i].iN, swaptions[i].iFactors, swaptions[i].dYears,
           swaptions[i].pdYield, swaptions[i].ppdFactors,
           100, NUM_TRIALS, BLOCK_SIZE, 0);
           assert(iSuccess == 1);
         */
        swaptions[i].dSimSwaptionMeanPrice = pdSwaptionPrice[0];
        swaptions[i].dSimSwaptionStdError = pdSwaptionPrice[1];
    }
};

// CumNormalInv.cpp
static FTYPE a[4] = {
    2.50662823884,
    -18.61500062529,
    41.39119773534,
    -25.44106049637
};

static FTYPE b[4] = {
    -8.47351093090,
    23.08336743743,
    -21.06224101826,
    3.13082909833
};

static FTYPE c[9] = {
    0.3374754822726147,
    0.9761690190917186,
    0.1607979714918209,
    0.0276438810333863,
    0.0038405729373609,
    0.0003951896511919,
    0.0000321767881768,
    0.0000002888167364,
    0.0000003960315187
};

FTYPE CumNormalInv( FTYPE u )
{
    // Returns the inverse of cumulative normal distribution function.
    // Reference: Moro, B., 1995, "The Full Monte," RISK (February), 57-58.

    FTYPE x, r;

    x = u - 0.5;
    if( fabs (x) < 0.42 )
    {
        r = x * x;
        r = x * fma(fma(fma(a[3], r, a[2]), r, a[1]), r, a[0]) / fma(fma(fma(fma(b[3], r, b[2]), r, b[1]), r, b[0]), r, 1.0);
        return (r);
    }

    r = u;
    if( x > 0.0 ) r = 1.0 - u;
    r = log(-log(r));
    r = fma(fma(fma(fma(fma(fma(fma(fma(c[8], r, c[7]), r, c[6]), r, c[5]), r, c[4]), r, c[3]), r, c[2]), r, c[1]), r, c[0]);
    if( x < 0.0 ) r = -r;

    return (r);

}

// icdf.cpp
void icdf_baseline(const int N, FTYPE *in, FTYPE *out){

    FTYPE z, r;

    const FTYPE
        a1 = -3.969683028665376e+01,
           a2 =  2.209460984245205e+02,
           a3 = -2.759285104469687e+02,
           a4 =  1.383577518672690e+02,
           a5 = -3.066479806614716e+01,
           a6 =  2.506628277459239e+00;

    const FTYPE
        b1 = -5.447609879822406e+01,
           b2 =  1.615858368580409e+02,
           b3 = -1.556989798598866e+02,
           b4 =  6.680131188771972e+01,
           b5 = -1.328068155288572e+01;

    const FTYPE
        c1 = -7.784894002430293e-03,
           c2 = -3.223964580411365e-01,
           c3 = -2.400758277161838e+00,
           c4 = -2.549732539343734e+00,
           c5 =  4.374664141464968e+00,
           c6 =  2.938163982698783e+00;

    const FTYPE
        //d0 =  0.0,
        d1 =  7.784695709041462e-03,
           d2 =  3.224671290700398e-01,
           d3 =  2.445134137142996e+00,
           d4 =  3.754408661907416e+00;

    // Limits of the approximation region.
#define U_LOW 0.02425

    const FTYPE u_low   = U_LOW, u_high  = 1.0 - U_LOW;

    for(int i=0; i<N; i++){
        FTYPE u = in[i];
        // Rational approximation for the lower region. ( 0 < u < u_low )
        if( u < u_low ){
            z = sqrt(-2.0*log(u));
            z = (((((c1*z+c2)*z+c3)*z+c4)*z+c5)*z+c6) / ((((d1*z+d2)*z+d3)*z+d4)*z+1.0);
        }
        // Rational approximation for the central region. ( u_low <= u <= u_high )
        else if( u <= u_high ){
            z = u - 0.5;
            r = z*z;
            z = (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*z / (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0);
        }
        // Rational approximation for the upper region. ( u_high < u < 1 )
        else {
            z = sqrt(-2.0*log(1.0-u));
            z = -(((((c1*z+c2)*z+c3)*z+c4)*z+c5)*z+c6) /  ((((d1*z+d2)*z+d3)*z+d4)*z+1.0);
        }
        out[i] = z;
    }
    return;
}

// MaxFunction.cpp
FTYPE dMax( FTYPE dA, FTYPE dB )
{
    return (dA>dB ? dA:dB);
}

// RanUnif.cpp
FTYPE RanUnif( long *s )
{
    // uniform random number generator
    long   ix, k1;
    FTYPE dRes;

    ix = *s;
    *s = ix+1;
    ix *= 1513517L;
    ix %= 2147483647L;
    k1 = ix/127773L;
    ix = 16807L*( ix - k1*127773L ) - k1 * 2836L;
    if (ix < 0) ix = ix + 2147483647L;
    dRes = (ix * 4.656612875e-10);
    return (dRes);
}

// HJM.cpp
int HJM_SimPath_Yield(FTYPE **ppdHJMPath,  //Matrix that stores generated HJM path (Output)
        int iN,				//Number of time-steps
        int iFactors,			//Number of factors in the HJM framework
        FTYPE dYears,		//Number of years
        FTYPE *pdYield,		//Input yield curve (at t=0) for dYears (iN time steps)
        FTYPE **ppdFactors,	//Matrix of Factor Volatilies
        long *lRndSeed)
{
    //This function returns a single generated HJM Path for the given inputs

    int iSuccess = 0;						//return variable

    FTYPE *pdForward;						//Vector that will store forward curve computed from given yield curve
    FTYPE **ppdDrifts;						//Matrix that will store drifts for different maturities for each factor
    FTYPE *pdTotalDrift;					//Vector that stores total drift for each maturity

    pdForward = dvector(0, iN-1);
    ppdDrifts = dmatrix(0, iFactors-1, 0, iN-2);
    pdTotalDrift = dvector(0, iN-2);

    //generating forward curve at t=0 from supplied yield curve
    iSuccess = HJM_Yield_to_Forward(pdForward, iN, pdYield);
    if (iSuccess!=1)
    {
        free_dvector(pdForward, 0, iN-1);
        free_dmatrix(ppdDrifts, 0, iFactors-1, 0, iN-2);
        free_dvector(pdTotalDrift, 0, iN-1);
        return iSuccess;
    }

    //computation of drifts from factor volatilities
    iSuccess = HJM_Drifts(pdTotalDrift, ppdDrifts, iN, iFactors, dYears, ppdFactors);
    if (iSuccess!=1)
    {
        free_dvector(pdForward, 0, iN-1);
        free_dmatrix(ppdDrifts, 0, iFactors-1, 0, iN-2);
        free_dvector(pdTotalDrift, 0, iN-1);
        return iSuccess;
    }

    //generating HJM Path
    iSuccess = HJM_SimPath_Forward(ppdHJMPath, iN, iFactors, dYears, pdForward, pdTotalDrift,ppdFactors, lRndSeed);
    if (iSuccess!=1)
    {
        free_dvector(pdForward, 0, iN-1);
        free_dmatrix(ppdDrifts, 0, iFactors-1, 0, iN-2);
        free_dvector(pdTotalDrift, 0, iN-1);
        return iSuccess;
    }

    free_dvector(pdForward, 0, iN-1);
    free_dmatrix(ppdDrifts, 0, iFactors-1, 0, iN-2);
    free_dvector(pdTotalDrift, 0, iN-1);
    iSuccess = 1;
    return iSuccess;
}

int HJM_Yield_to_Forward(FTYPE *pdForward,	//Forward curve to be outputted
        int iN,				//Number of time-steps
        FTYPE *pdYield)		//Input yield curve
{
    //This function computes forward rates from supplied yield rates.

    int iSuccess=0;
    int i;

    //forward curve computation
    pdForward[0] = pdYield[0];
    for(i=1;i<=iN-1; ++i){
        pdForward[i] = (i+1)*pdYield[i] - i*pdYield[i-1];	//as per formula
        //printf("pdForward: %f = (%d+1)*%f - %d*%f \n", pdForward[i], i, pdYield[i], i, pdYield[i-1]);
    }
    iSuccess=1;
    return iSuccess;
}

int HJM_Factors(FTYPE **ppdFactors,	//Output matrix that stores factor volatilities for different maturities
        int iN,
        int iFactors,
        FTYPE *pdVol,			//Input vector of total volatilities for different maturities
        FTYPE **ppdFacBreak)	//Input matrix of factor weights for each maturity
{
    //This function computes individual volatilities  for each factor for different maturities.
    //The function is called when the user inputs total volatility data and the weight distribution
    //according to which the total variance has to be split accross various factors.

    //For instance, the user may supply				   Maturity:	1	   2	  3      4
    //total vol (pdVol) as								  Sigma:  1.35%, 1.30%, 1.25%, 1.20%,....
    //and the weight breakdown (ppdFacBreak) as        Factor 1:   0.55,  0.60,  0.65,  0.69,....
    //												   Factor 2:   0.44,  0.39,  0.34,  0.30,....
    //												   Factor 3:   0.01,  0.01,  0.01,  0.01,....
    //Note that the weights add up to 1 in each case. Also, the weights are based on variance not volatility.

    //Based on these inputs, the function will calculate individual volatilties for each factor for each maturity.
    //The output (ppdFactors) may look something like: Maturity:	1	   2	  3      4
    //												   Factor 1:  1.00%  1.00%  1.00%  1.00%
    //												   Factor 2:  0.90%  0.82%  0.74%  0.67%
    //												   Factor 3:  0.10%  0.08%  0.05%  0.03%
    // (Please note that in this example the figures have been rounded and therefore may not be exact.)

    int i,j; //looping variables
    int iSuccess = 0;

    //Computation of factor volatilities
    for(i = 0; i<=iFactors-1; ++i)
        for(j=0; j<=iN-2;++j)
            ppdFactors[i][j] = sqrt((ppdFacBreak[i][j])*(pdVol[j])*(pdVol[j]));

    iSuccess =1;
    return iSuccess;
}

int HJM_Drifts(FTYPE *pdTotalDrift,	//Output vector that stores the total drift correction for each maturity
        FTYPE **ppdDrifts,		//Output matrix that stores drift correction for each factor for each maturity
        int iN,
        int iFactors,
        FTYPE dYears,
        FTYPE **ppdFactors)		//Input factor volatilities
{
    //This function computes drift corrections required for each factor for each maturity based on given factor volatilities

    int iSuccess =0;
    int i, j, l; //looping variables
    FTYPE ddelt = (FTYPE) (dYears/iN);
    FTYPE dSumVol;

    //computation of factor drifts for shortest maturity
    for (i=0; i<=iFactors-1; ++i)
        ppdDrifts[i][0] = 0.5*ddelt*(ppdFactors[i][0])*(ppdFactors[i][0]);

    //computation of factor drifts for other maturities
    for (i=0; i<=iFactors-1;++i)
        for (j=1; j<=iN-2; ++j)
        {
            ppdDrifts[i][j] = 0;
            for(l=0;l<=j-1;++l)
                ppdDrifts[i][j] -= ppdDrifts[i][l];
            dSumVol=0;
            for(l=0;l<=j;++l)
                dSumVol += ppdFactors[i][l];
            ppdDrifts[i][j] += 0.5*ddelt*(dSumVol)*(dSumVol);
        }

    //computation of total drifts for all maturities
    for(i=0;i<=iN-2;++i)
    {
        pdTotalDrift[i]=0;
        for(j=0;j<=iFactors-1;++j)
            pdTotalDrift[i]+= ppdDrifts[j][i];
    }

    iSuccess=1;
    return iSuccess;
}

int HJM_SimPath_Forward(FTYPE **ppdHJMPath,	//Matrix that stores generated HJM path (Output)
        int iN,					//Number of time-steps
        int iFactors,			//Number of factors in the HJM framework
        FTYPE dYears,			//Number of years
        FTYPE *pdForward,		//t=0 Forward curve
        FTYPE *pdTotalDrift,	//Vector containing total drift corrections for different maturities
        FTYPE **ppdFactors,	//Factor volatilities
        long *lRndSeed)			//Random number seed
{
    //This function computes and stores an HJM Path for given inputs

    int iSuccess = 0;
    int i,j,l; //looping variables

    FTYPE ddelt; //length of time steps
    FTYPE dTotalShock; //total shock by which the forward curve is hit at (t, T-t)
    FTYPE *pdZ; //vector to store random normals

    ddelt = (FTYPE)(dYears/iN);

    pdZ = dvector(0, iFactors -1); //assigning memory

    for(i=0;i<=iN-1;++i)
        for(j=0;j<=iN-1;++j)
            ppdHJMPath[i][j]=0; //initializing HJMPath to zero

    //t=0 forward curve stored iN first row of ppdHJMPath
    for(i=0;i<=iN-1; ++i)
        ppdHJMPath[0][i] = pdForward[i];

    //Generation of HJM Path
    for (j=1;j<=iN-1;++j)
    {

        for (l=0;l<=iFactors-1;++l)
            pdZ[l]= CumNormalInv(RanUnif(lRndSeed)); //shocks to hit various factors for forward curve at t

        for (l=0;l<=iN-(j+1);++l)
        {
            dTotalShock = 0;
            for (i=0;i<=iFactors-1;++i)
                dTotalShock += ppdFactors[i][l]* pdZ[i];
            ppdHJMPath[j][l] = ppdHJMPath[j-1][l+1]+ pdTotalDrift[l]*ddelt + sqrt(ddelt)*dTotalShock;
            //as per formula
        }
    }

    free_dvector(pdZ, 0, iFactors -1);
    iSuccess = 1;
    return iSuccess;
}

int HJM_Correlations(FTYPE **ppdHJMCorr,//Matrix that stores correlations among factor volatilities for different maturities
        int iN,
        int iFactors,
        FTYPE **ppdFactors)
{
    //This function is based on factor.xls created by Mark Broadie
    //The function computes correlations between factor volatilities for different maturities

    int iSuccess = 0;
    int i, j, l; //looping variables
    FTYPE *pdTotalVol; //vector that stores total volatility data for different maturities
    FTYPE **ppdWeights; //matrix that stores ratio of each factor to total volatility for different maturities

    pdTotalVol = dvector(0,iN-2);
    ppdWeights = dmatrix(0, iFactors-1,0, iN-2);

    //Total Volatility computed from given factor volatilities
    for(i=0;i<=iN-2;++i)
    {
        pdTotalVol[i]=0;
        for(j=0;j<=iFactors-1;++j)
            pdTotalVol[i] += ppdFactors[j][i]*ppdFactors[j][i];
        pdTotalVol[i] = sqrt(pdTotalVol[i]);
    }

    //Weights computed
    for(i=0;i<=iN-2;++i)
        for(j=0;j<=iFactors-1;++j)
            ppdWeights[j][i] = ppdFactors[j][i]/pdTotalVol[i];

    //Output matrix initialized to zero
    for(i=0;i<=iN-2;++i)
        for(j=0;j<=iN-2;++j)
            ppdHJMCorr[i][j]=0;

    //Correlations computed
    for(i=0;i<=iN-2;++i)
        for(j=i;j<=iN-2;++j)
            for(l=0;l<=iFactors-1;++l)
                ppdHJMCorr[i][j] += ppdWeights[l][i]*ppdWeights[l][j];

    free_dvector(pdTotalVol, 0,iN-2);
    free_dmatrix(ppdWeights, 0, iFactors-1,0, iN-2);
    iSuccess = 1;
    return iSuccess;
}

int HJM_Forward_to_Yield(FTYPE *pdYield,	//Output yield curve
        int iN,
        FTYPE *pdForward)	//Input forward curve
{
    //This function computes yield rates from supplied forward rates.

    int iSuccess=0;
    int i;

    //t=0 yield curve
    pdYield[0] = pdForward[0];
    for(i=1;i<=iN-1; ++i)
        pdYield[i] = (i*pdYield[i-1] + pdForward[i])/(i+1);

    iSuccess=1;
    return iSuccess;
}

int Discount_Factors(FTYPE *pdDiscountFactors,
        int iN,
        FTYPE dYears,
        FTYPE *pdRatePath)
{
    int i,j;                                //looping variables
    int iSuccess;                   //return variable

    FTYPE ddelt;                   //HJM time-step length
    ddelt = (FTYPE) (dYears/iN);

    //initializing the discount factor vector
    for (i=0; i<=iN-1; ++i)
        pdDiscountFactors[i] = 1.0;

    for (i=1; i<=iN-1; ++i)
        for (j=0; j<=i-1; ++j)
            pdDiscountFactors[i] *= exp(-pdRatePath[j]*ddelt);

    iSuccess = 1;
    return iSuccess;
}

int Discount_Factors_opt(FTYPE *pdDiscountFactors,
        int iN,
        FTYPE dYears,
        FTYPE *pdRatePath)
{
    int i,j;				//looping variables
    int iSuccess;			//return variable

    FTYPE ddelt;			//HJM time-step length
    ddelt = (FTYPE) (dYears/iN);

    FTYPE *pdexpRes;
    pdexpRes = dvector(0,iN-2);

    //initializing the discount factor vector
    for (i=0; i<=iN-1; ++i)
        pdDiscountFactors[i] = 1.0;

    //precompute the exponientials
    for (j=0; j<=(i-2); ++j){ pdexpRes[j] = -pdRatePath[j]*ddelt; }
    for (j=0; j<=(i-2); ++j){ pdexpRes[j] = exp(pdexpRes[j]);  }

    for (i=1; i<=iN-1; ++i)
        for (j=0; j<=i-1; ++j)
            pdDiscountFactors[i] *= pdexpRes[j];

    free_dvector(pdexpRes, 0, iN-2);
    iSuccess = 1;
    return iSuccess;
}

int Discount_Factors_Blocking(FTYPE *pdDiscountFactors,
        int iN,
        FTYPE dYears,
        FTYPE *pdRatePath,
        int BLOCKSIZE)
{
    int i,j,b;				//looping variables
    int iSuccess;			//return variable

    FTYPE ddelt;			//HJM time-step length
    ddelt = (FTYPE) (dYears/iN);

    FTYPE *pdexpRes;
    pdexpRes = dvector(0,(iN-1)*BLOCKSIZE-1);
    //precompute the exponientials
    for (j=0; j<=(iN-1)*BLOCKSIZE-1; ++j){ pdexpRes[j] = -pdRatePath[j]*ddelt; }
    for (j=0; j<=(iN-1)*BLOCKSIZE-1; ++j){ pdexpRes[j] = exp(pdexpRes[j]);  }


    //initializing the discount factor vector
    for (i=0; i<(iN)*BLOCKSIZE; ++i)
        pdDiscountFactors[i] = 1.0;

    for (i=1; i<=iN-1; ++i){
        //printf("\nVisiting timestep %d : ",i);
        for (b=0; b<BLOCKSIZE; b++){
            //printf("\n");
            for (j=0; j<=i-1; ++j){
                pdDiscountFactors[i*BLOCKSIZE + b] *= pdexpRes[j*BLOCKSIZE + b];
                //printf("(%f) ",pdexpRes[j*BLOCKSIZE + b]);
            }
        } // end Block loop
    }

    free_dvector(pdexpRes, 0,(iN-1)*BLOCKSIZE-1);
    iSuccess = 1;
    return iSuccess;
}

// HJM_SimPath_Forward_Blocking.cpp
void serialB(FTYPE **pdZ, FTYPE **randZ, int BLOCKSIZE, int iN, int iFactors)
{
    for(int l=0;l<=iFactors-1;++l){
        for(int b=0; b<BLOCKSIZE; b++){
            for (int j=1;j<=iN-1;++j){
                pdZ[l][BLOCKSIZE*j + b]= CumNormalInv(randZ[l][BLOCKSIZE*j + b]);  /* 18% of the total executition time */
            }
        }
    }
}

int HJM_SimPath_Forward_Blocking(FTYPE **ppdHJMPath,	//Matrix that stores generated HJM path (Output)
        int iN,					//Number of time-steps
        int iFactors,			//Number of factors in the HJM framework
        FTYPE dYears,			//Number of years
        FTYPE *pdForward,		//t=0 Forward curve
        FTYPE *pdTotalDrift,	//Vector containing total drift corrections for different maturities
        FTYPE **ppdFactors,	//Factor volatilities
        long *lRndSeed,			//Random number seed
        int BLOCKSIZE)
{
    //This function computes and stores an HJM Path for given inputs

    int iSuccess = 0;
    int i,j,l; //looping variables
    FTYPE **pdZ; //vector to store random normals
    FTYPE **randZ; //vector to store random normals
    FTYPE dTotalShock; //total shock by which the forward curve is hit at (t, T-t)
    FTYPE ddelt, sqrt_ddelt; //length of time steps

    ddelt = (FTYPE)(dYears/iN);
    sqrt_ddelt = sqrt(ddelt);

    pdZ   = dmatrix(0, iFactors-1, 0, iN*BLOCKSIZE -1); //assigning memory
    randZ = dmatrix(0, iFactors-1, 0, iN*BLOCKSIZE -1); //assigning memory

    // =====================================================
    // t=0 forward curve stored iN first row of ppdHJMPath
    // At time step 0: insert expected drift
    // rest reset to 0
    for(int b=0; b<BLOCKSIZE; b++){
        for(j=0;j<=iN-1;j++){
            ppdHJMPath[0][BLOCKSIZE*j + b] = pdForward[j];

            for(i=1;i<=iN-1;++i)
            { ppdHJMPath[i][BLOCKSIZE*j + b]=0; } //initializing HJMPath to zero
        }
    }
    // -----------------------------------------------------

    // =====================================================
    // sequentially generating random numbers


    for(int b=0; b<BLOCKSIZE; b++){
        for(int s=0; s<1; s++){
            for (j=1;j<=iN-1;++j){
                for (l=0;l<=iFactors-1;++l){
                    //compute random number in exact same sequence
                    randZ[l][BLOCKSIZE*j + b + s] = RanUnif(lRndSeed);  /* 10% of the total executition time */
                }
            }
        }
    }

    // =====================================================
    // shocks to hit various factors for forward curve at t

    /* 18% of the total executition time */
    serialB(pdZ, randZ, BLOCKSIZE, iN, iFactors);

    // =====================================================
    // Generation of HJM Path1
    for(int b=0; b<BLOCKSIZE; b++){ // b is the blocks
        for (j=1;j<=iN-1;++j) {// j is the timestep

            for (l=0;l<=iN-(j+1);++l){ // l is the future steps
                dTotalShock = 0;

                for (i=0;i<=iFactors-1;++i){// i steps through the stochastic factors
                    dTotalShock += ppdFactors[i][l]* pdZ[i][BLOCKSIZE*j + b];
                }

                ppdHJMPath[j][BLOCKSIZE*l+b] = ppdHJMPath[j-1][BLOCKSIZE*(l+1)+b]+ pdTotalDrift[l]*ddelt + sqrt_ddelt*dTotalShock;
                //as per formula
            }
        }
    } // end Blocks
    // -----------------------------------------------------

    free_dmatrix(pdZ, 0, iFactors -1, 0, iN*BLOCKSIZE -1);
    free_dmatrix(randZ, 0, iFactors -1, 0, iN*BLOCKSIZE -1);
    iSuccess = 1;
    return iSuccess;
}

// HJM_Swaption_Blocking.cpp
int HJM_Swaption_Blocking(FTYPE *pdSwaptionPrice, //Output vector that will store simulation results in the form:
        //Swaption Price
        //Swaption Standard Error
        //Swaption Parameters
        FTYPE dStrike,
        FTYPE dCompounding,     //Compounding convention used for quoting the strike (0 => continuous,
        //0.5 => semi-annual, 1 => annual).
        FTYPE dMaturity,	      //Maturity of the swaption (time to expiration)
        FTYPE dTenor,	      //Tenor of the swap
        FTYPE dPaymentInterval, //frequency of swap payments e.g. dPaymentInterval = 0.5 implies a swap payment every half
        //year
        //HJM Framework Parameters (please refer HJM.cpp for explanation of variables and functions)
        int iN,
        int iFactors,
        FTYPE dYears,
        FTYPE *pdYield,
        FTYPE **ppdFactors,
        //Simulation Parameters
        long iRndSeed,
        long lTrials,
        int BLOCKSIZE, int tid)

{
    int iSuccess = 0;
    int i;
    int b; //block looping variable
    long l; //looping variables

    FTYPE ddelt = (FTYPE)(dYears/iN);				//ddelt = HJM matrix time-step width. e.g. if dYears = 5yrs and
    //iN = no. of time points = 10, then ddelt = step length = 0.5yrs
    int iFreqRatio = (int)(dPaymentInterval/ddelt + 0.5);		// = ratio of time gap between swap payments and HJM step-width.
    //e.g. dPaymentInterval = 1 year. ddelt = 0.5year. This implies that a swap
    //payment will be made after every 2 HJM time steps.

    FTYPE dStrikeCont;				//Strike quoted in continuous compounding convention.
    //As HJM rates are continuous, the K in max(R-K,0) will be dStrikeCont and not dStrike.
    if(dCompounding==0) {
        dStrikeCont = dStrike;		//by convention, dCompounding = 0 means that the strike entered by user has been quoted
        //using continuous compounding convention
    } else {
        //converting quoted strike to continuously compounded strike
        dStrikeCont = (1/dCompounding)*log(1+dStrike*dCompounding);
    }
    //e.g., let k be strike quoted in semi-annual convention. Therefore, 1$ at the end of
    //half a year would earn = (1+k/2). For converting to continuous compounding,
    //(1+0.5*k) = exp(K*0.5)
    // => K = (1/0.5)*ln(1+0.5*k)

    //HJM Framework vectors and matrices
    int iSwapVectorLength;  // Length of the HJM rate path at the time index corresponding to swaption maturity.

    FTYPE **ppdHJMPath;    // **** per Trial data **** //

    FTYPE *pdForward;
    FTYPE **ppdDrifts;
    FTYPE *pdTotalDrift;

    // *******************************
    // ppdHJMPath = dmatrix(0,iN-1,0,iN-1);
    ppdHJMPath = dmatrix(0,iN-1,0,iN*BLOCKSIZE-1);    // **** per Trial data **** //
    pdForward = dvector(0, iN-1);
    ppdDrifts = dmatrix(0, iFactors-1, 0, iN-2);
    pdTotalDrift = dvector(0, iN-2);

    //==================================
    // **** per Trial data **** //
    FTYPE *pdDiscountingRatePath;	  //vector to store rate path along which the swaption payoff will be discounted
    FTYPE *pdPayoffDiscountFactors;  //vector to store discount factors for the rate path along which the swaption 
    //payoff will be discounted
    FTYPE *pdSwapRatePath;			  //vector to store the rate path along which the swap payments made will be discounted	
    FTYPE *pdSwapDiscountFactors;	  //vector to store discount factors for the rate path along which the swap
    //payments made will be discounted
    FTYPE *pdSwapPayoffs;			  //vector to store swap payoffs

    int iSwapStartTimeIndex;
    int iSwapTimePoints;
    FTYPE dSwapVectorYears;

    FTYPE dSwaptionPayoff;
    FTYPE dDiscSwaptionPayoff;
    FTYPE dFixedLegValue;

    // Accumulators
    FTYPE dSumSimSwaptionPrice;
    FTYPE dSumSquareSimSwaptionPrice;

    // Final returned results
    FTYPE dSimSwaptionMeanPrice;
    FTYPE dSimSwaptionStdError;

    // *******************************
    pdPayoffDiscountFactors = dvector(0, iN*BLOCKSIZE-1);
    pdDiscountingRatePath = dvector(0, iN*BLOCKSIZE-1);
    // *******************************

    iSwapVectorLength = (int) (iN - dMaturity/ddelt + 0.5);	//This is the length of the HJM rate path at the time index
    //corresponding to swaption maturity.
    // *******************************
    pdSwapRatePath = dvector(0, iSwapVectorLength*BLOCKSIZE - 1);
    pdSwapDiscountFactors  = dvector(0, iSwapVectorLength*BLOCKSIZE - 1);
    // *******************************
    pdSwapPayoffs = dvector(0, iSwapVectorLength - 1);

    iSwapStartTimeIndex = (int) (dMaturity/ddelt + 0.5);	//Swap starts at swaption maturity
    iSwapTimePoints = (int) (dTenor/ddelt + 0.5);			//Total HJM time points corresponding to the swap's tenor
    dSwapVectorYears = (FTYPE) (iSwapVectorLength*ddelt);

    //now we store the swap payoffs in the swap payoff vector
    for (i=0;i<=iSwapVectorLength-1;++i)
        pdSwapPayoffs[i] = 0.0; //initializing to zero
    for (i=iFreqRatio;i<=iSwapTimePoints;i+=iFreqRatio)
    {
        if(i != iSwapTimePoints)
            pdSwapPayoffs[i] = exp(dStrikeCont*dPaymentInterval) - 1; //the bond pays coupon equal to this amount
        if(i == iSwapTimePoints)
            pdSwapPayoffs[i] = exp(dStrikeCont*dPaymentInterval); //at terminal time point, bond pays coupon plus par amount
    }

    //generating forward curve at t=0 from supplied yield curve
    iSuccess = HJM_Yield_to_Forward(pdForward, iN, pdYield);
    if (iSuccess!=1)
        return iSuccess;

    //computation of drifts from factor volatilities
    iSuccess = HJM_Drifts(pdTotalDrift, ppdDrifts, iN, iFactors, dYears, ppdFactors);
    if (iSuccess!=1)
        return iSuccess;

    dSumSimSwaptionPrice = 0.0;
    dSumSquareSimSwaptionPrice = 0.0;

    //Simulations begin:
    for (l=0;l<=lTrials-1;l+=BLOCKSIZE) {
        //For each trial a new HJM Path is generated
        iSuccess = HJM_SimPath_Forward_Blocking(ppdHJMPath, iN, iFactors, dYears, pdForward, pdTotalDrift,ppdFactors, &iRndSeed, BLOCKSIZE); /* GC: 51% of the time goes here */
        if (iSuccess!=1)
            return iSuccess;

        //now we compute the discount factor vector

        for(i=0;i<=iN-1;++i){
            for(b=0;b<=BLOCKSIZE-1;b++){
                pdDiscountingRatePath[BLOCKSIZE*i + b] = ppdHJMPath[i][0 + b];
            }
        }
        iSuccess = Discount_Factors_Blocking(pdPayoffDiscountFactors, iN, dYears, pdDiscountingRatePath, BLOCKSIZE); /* 15% of the time goes here */

        if (iSuccess!=1)
            return iSuccess;

        //now we compute discount factors along the swap path
        for (i=0;i<=iSwapVectorLength-1;++i){
            for(b=0;b<BLOCKSIZE;b++){
                pdSwapRatePath[i*BLOCKSIZE + b] =
                    ppdHJMPath[iSwapStartTimeIndex][i*BLOCKSIZE + b];
            }
        }
        iSuccess = Discount_Factors_Blocking(pdSwapDiscountFactors, iSwapVectorLength, dSwapVectorYears, pdSwapRatePath, BLOCKSIZE);
        if (iSuccess!=1)
            return iSuccess;

        // ========================
        // Simulation
        for (b=0;b<BLOCKSIZE;b++){
            dFixedLegValue = 0.0;
            for (i=0;i<=iSwapVectorLength-1;++i){
                dFixedLegValue += pdSwapPayoffs[i]*pdSwapDiscountFactors[i*BLOCKSIZE + b];
            }
            dSwaptionPayoff = dMax(dFixedLegValue - 1.0, 0);

            dDiscSwaptionPayoff = dSwaptionPayoff*pdPayoffDiscountFactors[iSwapStartTimeIndex*BLOCKSIZE + b];

            // ========= end simulation ======================================

            // accumulate into the aggregating variables =====================
            dSumSimSwaptionPrice += dDiscSwaptionPayoff;
            dSumSquareSimSwaptionPrice += dDiscSwaptionPayoff*dDiscSwaptionPayoff;
        } // END BLOCK simulation
    }

    // Simulation Results Stored
    dSimSwaptionMeanPrice = dSumSimSwaptionPrice/lTrials;
    dSimSwaptionStdError = sqrt((dSumSquareSimSwaptionPrice-dSumSimSwaptionPrice*dSumSimSwaptionPrice/lTrials)/
            (lTrials-1.0))/sqrt((FTYPE)lTrials);

    //results returned
    pdSwaptionPrice[0] = dSimSwaptionMeanPrice;
    pdSwaptionPrice[1] = dSimSwaptionStdError;

    iSuccess = 1;
    return iSuccess;
}

// nr_routines.c
#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}

int choldc(FTYPE **a, int n)
{
    // modifications:  float -> FTYPE
    // nrerror removed
    // routine returns int instead of void, where
    //    1 means success, and 0 failure
    // p vector removed
    // upper triangular part of A is zeroed out

    int i,j,k;
    FTYPE sum;

    for (i=1;i<=n;i++) {
        for (j=i;j<=n;j++) {
            for (sum=a[i][j],k=i-1;k>=1;k--) sum -= a[i][k]*a[j][k];
            if (i == j) {
                if (sum <= 0.0)
                    // matrix is not positive definite
                    return(0);
                a[i][i]=sqrt(sum);
            } else a[j][i]=sum/a[i][i];
        }
    }

    for (i=1;i<=n-1;i++)
        for (j=i+1;j<=n;j++)
            a[i][j] = 0.0;

    return(1);
}

/*
void gaussj(FTYPE **a, int n, FTYPE **b, int m)
{
    int *indxc,*indxr,*ipiv;
    int i,icol,irow,j,k,l,ll;
    FTYPE big,dum,pivinv,temp;

    indxc=ivector(1,n);
    indxr=ivector(1,n);
    ipiv=ivector(1,n);
    icol=-1; irow=-1;
    for (j=1;j<=n;j++) ipiv[j]=0;
    for (i=1;i<=n;i++) {
        big=0.0;
        for (j=1;j<=n;j++)
            if (ipiv[j] != 1)
                for (k=1;k<=n;k++) {
                    if (ipiv[k] == 0) {
                        if (fabs(a[j][k]) >= big) {
                            big=fabs(a[j][k]);
                            irow=j;
                            icol=k;
                        }
                    } else if (ipiv[k] > 1) {
                        std::string str = "gaussj: Singular Matrix-1";
                        nrerror(const_cast< char* >(str.c_str()));
                    }
                }
        ++(ipiv[icol]);
        if (irow != icol) {
            for (l=1;l<=n;l++) SWAP(a[irow][l],a[icol][l])
                for (l=1;l<=m;l++) SWAP(b[irow][l],b[icol][l])
        }
        indxr[i]=irow;
        indxc[i]=icol;
        if (a[icol][icol] == 0.0) {
            std::string str = "gaussj: Singular Matrix-2";
            nrerror(const_cast< char* >(str.c_str()));
        }
        pivinv=1.0/a[icol][icol];
        a[icol][icol]=1.0;
        for (l=1;l<=n;l++) a[icol][l] *= pivinv;
        for (l=1;l<=m;l++) b[icol][l] *= pivinv;
        for (ll=1;ll<=n;ll++)
            if (ll != icol) {
                dum=a[ll][icol];
                a[ll][icol]=0.0;
                for (l=1;l<=n;l++) a[ll][l] -= a[icol][l]*dum;
                for (l=1;l<=m;l++) b[ll][l] -= b[icol][l]*dum;
            }
    }
    for (l=n;l>=1;l--) {
        if (indxr[l] != indxc[l])
            for (k=1;k<=n;k++)
                SWAP(a[k][indxr[l]],a[k][indxc[l]]);
    }
    free_ivector(ipiv,1,n);
    free_ivector(indxr,1,n);
    free_ivector(indxc,1,n);
}
*/

#undef SWAP
#undef NRANSI

void nrerror( char error_text[] )
{
    /*
    // Numerical Recipes standard error handler
    fprintf( stderr,"Numerical Recipes run-time error...\n" );
    fprintf( stderr,"%s\n",error_text );
    fprintf( stderr,"...now exiting to system...\n" );
    exit(1);
    */
}

/*
int *ivector(long nl, long nh)
{
    // allocate an int vector with subscript range v[nl..nh]
    int *v;

    v=(int *)malloc((size_t) ((nh-nl+2)*sizeof(int)));
    if (!v) {
        std::string str = "allocation failure in ivector()";
        nrerror(const_cast< char* >(str.c_str()));
    }
    return v-nl+1;
}

void free_ivector(int *v, long nl, long nh)
{
    // free an int vector allocated with ivector()
    free((char *) (v+nl-1));
}
*/

FTYPE *dvector( long nl, long nh )
{
    // allocate a FTYPE vector with subscript range v[nl..nh]

    FTYPE *v;

    v=(FTYPE *)malloc((size_t) ((nh-nl+2)*sizeof(FTYPE)));
    if (!v) {
        std::string str = "allocation failure in dvector()";
        nrerror(const_cast< char* >(str.c_str()));
    }
    return v-nl+1;

}

void free_dvector( FTYPE *v, long nl, long nh )
{
    // free a FTYPE vector allocated with dvector()

    free((char*) (v+nl-1));
}

FTYPE **dmatrix( long nrl, long nrh, long ncl, long nch )
{
    // allocate a FTYPE matrix with subscript range m[nrl..nrh][ncl..nch]

    long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
    FTYPE **m;

    // allocate pointers to rows
    m=(FTYPE **) malloc((size_t)((nrow+1)*sizeof(FTYPE*)));
    if (!m) {
        std::string str = "allocation failure 1 in dmatrix()";
        nrerror(const_cast< char* >(str.c_str()));
    }
    m += 1;
    m -= nrl;

    // allocate rows and set pointers to them
    m[nrl]=(FTYPE *) malloc((size_t)((nrow*ncol+1)*sizeof(FTYPE)));
    if (!m[nrl]) {
        std::string str = "allocation failure 2 in dmatrix()";
        nrerror(const_cast< char* >(str.c_str()));
    }
    m[nrl] += 1;
    m[nrl] -= ncl;

    for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

    // return pointer to array of pointers to rows
    return m;

}

void free_dmatrix( FTYPE **m, long nrl, long nrh, long ncl, long nch )
{
    // free a FTYPE matrix allocated by dmatrix()

    free((char*) (m[nrl]+ncl-1));
    free((char*) (m+nrl-1));

}
*/
