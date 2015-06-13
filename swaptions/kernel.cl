
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
    FTYPE pddFactors[][SIZE];
} parm;

__kernel void kernel_func(__global parm* swaptions, int nSwaptions) {
    /*
    int tid = get_global_id(0);
    int nWorkItems = get_global_size(0);
    FTYPE pdSwaptionPrice[2];
    int chunksize = nSwaptions/nWorkItems;
    int beg = tid*chunksize;
    int end = (tid+1)*chunksize;
    if(tid == nWorkItems -1)
        end = nSwaptions;
    for(int i=beg; i < end; i++) {
        int iSuccess = HJM_Swaption_Blocking(pdSwaptionPrice,  swaptions[i].dStrike,
                swaptions[i].dCompounding, swaptions[i].dMaturity,
                swaptions[i].dTenor, swaptions[i].dPaymentInterval,
                swaptions[i].iN, swaptions[i].iFactors, swaptions[i].dYears,
                swaptions[i].pdYield, swaptions[i].ppdFactors,
                100, NUM_TRIALS, BLOCK_SIZE, 0);
        assert(iSuccess == 1);
        swaptions[i].dSimSwaptionMeanPrice = pdSwaptionPrice[0];
        swaptions[i].dSimSwaptionStdError = pdSwaptionPrice[1];
    }
    */
};
