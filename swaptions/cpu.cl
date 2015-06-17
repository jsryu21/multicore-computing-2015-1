// HJM_type.h
#define FTYPE double

typedef struct
{
    FTYPE dSimSwaptionMeanPrice;
    FTYPE dSimSwaptionStdError;
    FTYPE dStrike;
    FTYPE dCompounding;
    FTYPE dMaturity;
    FTYPE dTenor;
    FTYPE dPaymentInterval;
    FTYPE dYears;
    int Id;
    int iN;
    int iFactors;
} __attribute__ ((aligned (8))) parm;

// HJM.h
FTYPE RanUnif( long* s );
FTYPE CumNormalInv( FTYPE u );
int HJM_SimPath_Forward_Blocking(
        __global FTYPE* pdHJMPath,
        int iN,
        int iFactors,
        FTYPE dYears,
        __global FTYPE* pdForward,
        __global FTYPE* pdTotalDrift,
        __global FTYPE* pdFactors,
        __global FTYPE* pdZ,
        __global FTYPE* pdRandZ,
        long* lRndSeed,
        int BLOCK_SIZE,
        FTYPE ddelt);

int Discount_Factors_Blocking(__global FTYPE* pdDiscountFactors
        , int iN
        , FTYPE dYears
        , __global FTYPE* pdRatePath
        , __global FTYPE* pdExpRes
        , int BLOCK_SIZE);

int HJM_Swaption_Blocking(
        FTYPE* pdSwaptionPrice, //Output vector that will store simulation results in the form:
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
        __global FTYPE* pdYield,
        __global FTYPE* pdForward,
        __global FTYPE* pdTotalDrift,
        __global FTYPE* pdPayoffDiscountFactors,
        __global FTYPE* pdDiscountingRatePath,
        __global FTYPE* pdSwapRatePath,
        __global FTYPE* pdSwapDiscountFactors,
        __global FTYPE* pdSwapPayoffs,
        __global FTYPE* pdExpRes,
        __global FTYPE* pdFactors,
        __global FTYPE* pdHJMPath,
        __global FTYPE* pdDrifts,
        __global FTYPE* pdZ,
        __global FTYPE* pdRandZ,
        //Simulation Parameters
        long iRndSeed,
        long lTrials,
        int BLOCK_SIZE,
        int tid,
        FTYPE ddelt,
        int iSwapVectorLength);

// HJM_Securities.h
int HJM_Yield_to_Forward(
        __global FTYPE* pdForward,
        int iN,
        __global FTYPE* pdYield);

int HJM_Drifts(
        __global FTYPE* pdTotalDrift,
        __global FTYPE* pdDrifts,
        int iN,
        int iFactors,
        FTYPE dYears,
        __global FTYPE* pdFactors,
        FTYPE ddelt);

FTYPE dMax( FTYPE dA, FTYPE dB );

__kernel void kernel_func(
        __global parm* swaptions,
        int nSwaptions,
        int NUM_TRIALS,
        int BLOCK_SIZE,
        __global FTYPE* pdYield,
        __global FTYPE* pdForward,
        __global FTYPE* pdTotalDrift,
        __global FTYPE* pdPayoffDiscountFactors,
        __global FTYPE* pdDiscountingRatePath,
        __global FTYPE* pdSwapRatePath,
        __global FTYPE* pdSwapDiscountFactors,
        __global FTYPE* pdSwapPayoffs,
        __global FTYPE* pdExpRes,
        __global FTYPE* pdFactors,
        __global FTYPE* pdHJMPath,
        __global FTYPE* pdDrifts,
        __global FTYPE* pdZ,
        __global FTYPE* pdRandZ) {
    int tid = get_global_id(0);
    int nWorkItems = get_global_size(0);
    FTYPE pdSwaptionPrice[2];
    int chunksize = nSwaptions/nWorkItems;
    int beg = tid*chunksize;
    int end = (tid+1)*chunksize;

    if(tid == nWorkItems - 1) {
        end = nSwaptions;
    }

    for(int i=beg; i < end; i++) {
        int iN = swaptions[i].iN;
        FTYPE dMaturity = swaptions[i].dMaturity;
        FTYPE dYears = swaptions[i].dYears;
        FTYPE ddelt = (FTYPE)(dYears/iN);
        int iSwapVectorLength = (int)(iN - dMaturity / ddelt + 0.5);
        int iFactors = swaptions[i].iFactors;

        int iSuccess = HJM_Swaption_Blocking(
                pdSwaptionPrice,
                swaptions[i].dStrike,
                swaptions[i].dCompounding,
                dMaturity,
                swaptions[i].dTenor,
                swaptions[i].dPaymentInterval,
                iN,
                iFactors,
                dYears,
                &pdYield[i * iN],
                &pdForward[i * iN],
                &pdTotalDrift[i * (iN - 1)],
                &pdPayoffDiscountFactors[i * (iN * BLOCK_SIZE)],
                &pdDiscountingRatePath[i * (iN * BLOCK_SIZE)],
                &pdSwapRatePath[i * (iSwapVectorLength * BLOCK_SIZE)],
                &pdSwapDiscountFactors[i * (iSwapVectorLength * BLOCK_SIZE)],
                &pdSwapPayoffs[i * iSwapVectorLength],
                &pdExpRes[i * ((iN - 1) * BLOCK_SIZE)],
                &pdFactors[i * iFactors * (iN - 1)],
                &pdHJMPath[i * iN * (iN * BLOCK_SIZE)],
                &pdDrifts[i * iFactors * (iN - 1)],
                &pdZ[i * iFactors * (iN * BLOCK_SIZE)],
                &pdRandZ[i * iFactors * (iN * BLOCK_SIZE)],
                100,
                NUM_TRIALS,
                BLOCK_SIZE,
                tid,
                ddelt,
                iSwapVectorLength);
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
        r = x * ((( a[3]*r + a[2]) * r + a[1]) * r + a[0])/
            ((((b[3] * r+ b[2]) * r + b[1]) * r + b[0]) * r + 1.0);
        return (r);
    }

    r = u;
    if( x > 0.0 ) r = 1.0 - u;
    r = log(-log(r));
    r = c[0] + r * (c[1] + r *
            (c[2] + r * (c[3] + r *
                         (c[4] + r * (c[5] + r * (c[6] + r * (c[7] + r*c[8])))))));
    if( x < 0.0 ) r = -r;

    return (r);
}

// MaxFunction.cpp
FTYPE dMax( FTYPE dA, FTYPE dB )
{
    return (dA>dB ? dA:dB);
}

// RanUnif.cpp
FTYPE RanUnif( long* s )
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
int HJM_Yield_to_Forward(
        __global FTYPE* pdForward,	//Forward curve to be outputted
        int iN,				//Number of time-steps
        __global FTYPE* pdYield)		//Input yield curve
{
    //This function computes forward rates from supplied yield rates.

    int iSuccess=0;
    int i;

    //forward curve computation
    pdForward[0] = pdYield[0];
    for(i=1;i<=iN-1; ++i){
        pdForward[i] = (i+1)*pdYield[i] - i*pdYield[i-1];	//as per formula
    }
    iSuccess=1;
    return iSuccess;
}

int HJM_Drifts(
        __global FTYPE* pdTotalDrift,	//Output vector that stores the total drift correction for each maturity
        __global FTYPE* pdDrifts,		//Output matrix that stores drift correction for each factor for each maturity
        int iN,
        int iFactors,
        FTYPE dYears,
        __global FTYPE* pdFactors,
        FTYPE ddelt)		//Input factor volatilities
{
    //This function computes drift corrections required for each factor for each maturity based on given factor volatilities

    int iSuccess =0;
    int i, j, l; //looping variables
    FTYPE dSumVol;

    //computation of factor drifts for shortest maturity
    for (i=0; i<=iFactors-1; ++i)
        pdDrifts[i * (iN - 1)] = 0.5*ddelt*(pdFactors[i * (iN - 1)])*(pdFactors[i * (iN - 1)]);

    //computation of factor drifts for other maturities
    for (i=0; i<=iFactors-1;++i)
        for (j=1; j<=iN-2; ++j)
        {
            pdDrifts[i * (iN - 1) + j] = 0;
            for(l=0;l<=j-1;++l)
                pdDrifts[i * (iN - 1) + j] -= pdDrifts[i * (iN - 1) + l];
            dSumVol=0;
            for(l=0;l<=j;++l)
                dSumVol += pdFactors[i * (iN - 1) + l];
            pdDrifts[i * (iN - 1) + j] += 0.5*ddelt*(dSumVol)*(dSumVol);
        }

    //computation of total drifts for all maturities
    for(i=0;i<=iN-2;++i)
    {
        pdTotalDrift[i]=0;
        for(j=0;j<=iFactors-1;++j)
            pdTotalDrift[i]+= pdDrifts[j * (iN - 1) + i];
    }

    iSuccess=1;
    return iSuccess;
}

int Discount_Factors_Blocking(
        __global FTYPE* pdDiscountFactors,
        int iN,
        FTYPE dYears,
        __global FTYPE* pdRatePath,
        __global FTYPE* pdExpRes,
        int BLOCK_SIZE)
{
    int i,j,b;				//looping variables
    int iSuccess;			//return variable

    FTYPE ddelt;			//HJM time-step length
    ddelt = (FTYPE) (dYears/iN);

    //precompute the exponientials
    for (j=0; j<=(iN-1)*BLOCK_SIZE-1; ++j){ pdExpRes[j] = -pdRatePath[j]*ddelt; }
    for (j=0; j<=(iN-1)*BLOCK_SIZE-1; ++j){ pdExpRes[j] = exp(pdExpRes[j]);  }

    //initializing the discount factor vector
    for (i=0; i<(iN)*BLOCK_SIZE; ++i)
        pdDiscountFactors[i] = 1.0;

    for (i=1; i<=iN-1; ++i){
        for (b=0; b<BLOCK_SIZE; b++){
            for (j=0; j<=i-1; ++j){
                pdDiscountFactors[i*BLOCK_SIZE + b] *= pdExpRes[j*BLOCK_SIZE + b];
            }
        }
    }

    iSuccess = 1;
    return iSuccess;
}

// HJM_SimPath_Forward_Blocking.cpp
int HJM_SimPath_Forward_Blocking(
        __global FTYPE* pdHJMPath,	//Matrix that stores generated HJM path (Output)
        int iN,					//Number of time-steps
        int iFactors,			//Number of factors in the HJM framework
        FTYPE dYears,			//Number of years
        __global FTYPE* pdForward,		//t=0 Forward curve
        __global FTYPE* pdTotalDrift,	//Vector containing total drift corrections for different maturities
        __global FTYPE* pdFactors,	//Factor volatilities
        __global FTYPE* pdZ, //vector to store random normals
        __global FTYPE* pdRandZ, //vector to store random normals
        long* lRndSeed,			//Random number seed
        int BLOCK_SIZE,
        FTYPE ddelt)
{
    //This function computes and stores an HJM Path for given inputs

    int iSuccess = 0;
    int i,j,l; //looping variables
    FTYPE dTotalShock; //total shock by which the forward curve is hit at (t, T-t)
    FTYPE sqrt_ddelt; //length of time steps

    sqrt_ddelt = sqrt(ddelt);

    // t=0 forward curve stored iN first row of ppdHJMPath
    // At time step 0: insert expected drift
    // rest reset to 0
    for(int b=0; b<BLOCK_SIZE; b++){
        for(j=0;j<=iN-1;j++){
            pdHJMPath[BLOCK_SIZE*j + b] = pdForward[j];

            for(i=1;i<=iN-1;++i)
            { pdHJMPath[i * (iN * BLOCK_SIZE) + BLOCK_SIZE*j + b]=0; } //initializing HJMPath to zero
        }
    }

    // sequentially generating random numbers
    for(int b=0; b<BLOCK_SIZE; b++){
        for(int s=0; s<1; s++){
            for (j=1;j<=iN-1;++j){
                for (l=0;l<=iFactors-1;++l){
                    //compute random number in exact same sequence
                    // 10% of the total executition time
                    pdRandZ[l * (iN * BLOCK_SIZE) + BLOCK_SIZE*j + b + s] = RanUnif(lRndSeed);
                }
            }
        }
    }

    // shocks to hit various factors for forward curve at t
    // 18% of the total executition time
    for(int l=0;l<=iFactors-1;++l){
        for(int b=0; b<BLOCK_SIZE; b++){
            for (int j=1;j<=iN-1;++j){
                // 18% of the total executition time
                pdZ[l * (iN * BLOCK_SIZE) + BLOCK_SIZE*j + b]= CumNormalInv(pdRandZ[l * (iN * BLOCK_SIZE) + BLOCK_SIZE*j + b]);
            }
        }
    }

    // Generation of HJM Path1
    for(int b=0; b<BLOCK_SIZE; b++){ // b is the blocks
        for (j=1;j<=iN-1;++j) {// j is the timestep

            for (l=0;l<=iN-(j+1);++l){ // l is the future steps
                dTotalShock = 0;

                for (i=0;i<=iFactors-1;++i){// i steps through the stochastic factors
                    dTotalShock += pdFactors[i * (iN - 1) + l]* pdZ[i * (iN * BLOCK_SIZE) + BLOCK_SIZE*j + b];
                }

                pdHJMPath[j * (iN * BLOCK_SIZE) + BLOCK_SIZE*l+b] = pdHJMPath[(j-1) * (iN * BLOCK_SIZE) + BLOCK_SIZE*(l+1)+b]+ pdTotalDrift[l]*ddelt + sqrt_ddelt*dTotalShock;
                //as per formula
            }
        }
    }

    iSuccess = 1;
    return iSuccess;
}

// HJM_Swaption_Blocking.cpp
int HJM_Swaption_Blocking(
        FTYPE* pdSwaptionPrice, //Output vector that will store simulation results in the form:
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
        __global FTYPE* pdYield,
        __global FTYPE* pdForward,
        __global FTYPE* pdTotalDrift,
        __global FTYPE* pdPayoffDiscountFactors,
        __global FTYPE* pdDiscountingRatePath,
        __global FTYPE* pdSwapRatePath,
        __global FTYPE* pdSwapDiscountFactors,
        __global FTYPE* pdSwapPayoffs,
        __global FTYPE* pdExpRes,
        __global FTYPE* pdFactors,
        __global FTYPE* pdHJMPath,
        __global FTYPE* pdDrifts,
        __global FTYPE* pdZ,
        __global FTYPE* pdRandZ,
        //Simulation Parameters
        long iRndSeed,
        long lTrials,
        int BLOCK_SIZE,
        int tid,
        FTYPE ddelt,
        int iSwapVectorLength)
{
    int iSuccess = 0;
    int i;
    int b; //block looping variable
    long l; //looping variables

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
    iSuccess = HJM_Drifts(pdTotalDrift, pdDrifts, iN, iFactors, dYears, pdFactors, ddelt);
    if (iSuccess!=1)
        return iSuccess;

    dSumSimSwaptionPrice = 0.0;
    dSumSquareSimSwaptionPrice = 0.0;

    //Simulations begin:
    for (l=0;l<=lTrials-1;l+=BLOCK_SIZE) {
        //For each trial a new HJM Path is generated
        // GC: 51% of the time goes here
        iSuccess = HJM_SimPath_Forward_Blocking(pdHJMPath, iN, iFactors, dYears, pdForward, pdTotalDrift,pdFactors, pdZ, pdRandZ, &iRndSeed, BLOCK_SIZE, ddelt);
        if (iSuccess!=1)
            return iSuccess;

        //now we compute the discount factor vector

        for(i=0;i<=iN-1;++i){
            for(b=0;b<=BLOCK_SIZE-1;b++){
                pdDiscountingRatePath[BLOCK_SIZE*i + b] = pdHJMPath[i * (iN * BLOCK_SIZE) + b];
            }
        }
        // 15% of the time goes here
        iSuccess = Discount_Factors_Blocking(pdPayoffDiscountFactors, iN, dYears, pdDiscountingRatePath, pdExpRes, BLOCK_SIZE);

        if (iSuccess!=1)
            return iSuccess;

        //now we compute discount factors along the swap path
        for (i=0;i<=iSwapVectorLength-1;++i){
            for(b=0;b<BLOCK_SIZE;b++){
                pdSwapRatePath[i*BLOCK_SIZE + b] = pdHJMPath[iSwapStartTimeIndex * (iN * BLOCK_SIZE) + i*BLOCK_SIZE + b];
            }
        }
        iSuccess = Discount_Factors_Blocking(pdSwapDiscountFactors, iSwapVectorLength, dSwapVectorYears, pdSwapRatePath, pdExpRes, BLOCK_SIZE);
        if (iSuccess!=1)
            return iSuccess;

        // Simulation
        for (b=0;b<BLOCK_SIZE;b++){
            dFixedLegValue = 0.0;
            for (i=0;i<=iSwapVectorLength-1;++i){
                dFixedLegValue += pdSwapPayoffs[i]*pdSwapDiscountFactors[i*BLOCK_SIZE + b];
            }
            dSwaptionPayoff = dMax(dFixedLegValue - 1.0, 0);

            dDiscSwaptionPayoff = dSwaptionPayoff*pdPayoffDiscountFactors[iSwapStartTimeIndex*BLOCK_SIZE + b];

            // accumulate into the aggregating variables =====================
            dSumSimSwaptionPrice += dDiscSwaptionPayoff;
            dSumSquareSimSwaptionPrice += dDiscSwaptionPayoff*dDiscSwaptionPayoff;
        }
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
