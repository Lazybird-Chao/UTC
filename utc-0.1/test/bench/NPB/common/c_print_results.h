#ifndef __C_PRINT_RESULTS_H__
#define __C_PRINT_RESULTS_H__

#ifndef __cplusplus
void c_print_results( const char   *name,
        char   class,
        int    n1,
        int    n2,
        int    n3,
        int    niter,
        double t,
        double mops,
		const char   *optype,
        int    passed_verification,
		const char   *npbversion,
		const char   *compiletime,
		const char   *cc,
		const char   *clink,
		const char   *c_lib,
		const char   *c_inc,
		const char   *cflags,
		const char   *clinkflags );
#else
void c_print_results( const char   *name,
                      char   _class,
                      int    n1,
                      int    n2,
                      int    n3,
                      int    niter,
                      double t,
                      double mops,
					  const char   *optype,
                      int    passed_verification,
                      const char   *npbversion,
					  const char   *compiletime,
					  const char   *cc,
					  const char   *clink,
					  const char   *c_lib,
					  const char   *c_inc,
					  const char   *cflags,
					  const char   *clinkflags );

#endif


#endif //__PRINT_RESULTS_H__
