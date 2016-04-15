#ifndef __PRINT_RESULTS_H__
#define __PRINT_RESULTS_H__

#ifndef __cplusplus
void print_results(char *name, char class, int n1, int n2, int n3, int niter,
    double t, double mops, char *optype, logical verified, char *npbversion,
    char *compiletime, char *cs1, char *cs2, char *cs3, char *cs4, char *cs5,
    char *cs6, char *cs7);
#else
void print_results(char *name, char _class, int n1, int n2, int n3, int niter,
    double t, double mops, char *optype, bool verified, char *npbversion,
    char *compiletime, char *cs1, char *cs2, char *cs3, char *cs4, char *cs5,
    char *cs6, char *cs7);

#endif


#endif //__PRINT_RESULTS_H__
