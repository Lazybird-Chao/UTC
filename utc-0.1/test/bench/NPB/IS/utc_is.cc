#include "../../helper_getopt.h"
#include "npbparams.h"
#include "c_print_results.h"

#include "Utc.h"

#include <iostream>
#include <string>

using namespace iUtc;


/******************/
/* default values */
/******************/
#ifndef CLASS
#define CLASS 'S'
#endif
#ifndef NUM_PROCS
#define NUM_PROCS            1
#endif
#define MIN_PROCS            1


/*************/
/*  CLASS S  */
/*************/
#if CLASS == 'S'
#define  TOTAL_KEYS_LOG_2    16
#define  MAX_KEY_LOG_2       11
#define  NUM_BUCKETS_LOG_2   9
#endif


/*************/
/*  CLASS W  */
/*************/
#if CLASS == 'W'
#define  TOTAL_KEYS_LOG_2    20
#define  MAX_KEY_LOG_2       16
#define  NUM_BUCKETS_LOG_2   10
#endif

/*************/
/*  CLASS A  */
/*************/
#if CLASS == 'A'
#define  TOTAL_KEYS_LOG_2    23
#define  MAX_KEY_LOG_2       19
#define  NUM_BUCKETS_LOG_2   10
#endif


/*************/
/*  CLASS B  */
/*************/
#if CLASS == 'B'
#define  TOTAL_KEYS_LOG_2    25
#define  MAX_KEY_LOG_2       21
#define  NUM_BUCKETS_LOG_2   10
#endif


/*************/
/*  CLASS C  */
/*************/
#if CLASS == 'C'
#define  TOTAL_KEYS_LOG_2    27
#define  MAX_KEY_LOG_2       23
#define  NUM_BUCKETS_LOG_2   10
#endif


/*************/
/*  CLASS D  */
/*************/
#if CLASS == 'D'
#define  TOTAL_KEYS_LOG_2    29
#define  MAX_KEY_LOG_2       27
#define  NUM_BUCKETS_LOG_2   10
#undef   MIN_PROCS
#define  MIN_PROCS           4
#endif


#define  TOTAL_KEYS          (1 << TOTAL_KEYS_LOG_2)
#define  MAX_KEY             (1 << MAX_KEY_LOG_2)
#define  NUM_BUCKETS         (1 << NUM_BUCKETS_LOG_2)
#define  NUM_KEYS            (TOTAL_KEYS/NUM_PROCS*MIN_PROCS)

/*****************************************************************/
/* On larger number of processors, since the keys are (roughly)  */
/* gaussian distributed, the first and last processor sort keys  */
/* in a large interval, requiring array sizes to be larger. Note */
/* that for large NUM_PROCS, NUM_KEYS is, however, a small number*/
/* The required array size also depends on the bucket size used. */
/* The following values are validated for the 1024-bucket setup. */
/*****************************************************************/
#if   NUM_PROCS < 256
#define  SIZE_OF_BUFFERS     3*NUM_KEYS/2
#elif NUM_PROCS < 512
#define  SIZE_OF_BUFFERS     5*NUM_KEYS/2
#elif NUM_PROCS < 1024
#define  SIZE_OF_BUFFERS     4*NUM_KEYS
#else
#define  SIZE_OF_BUFFERS     13*NUM_KEYS/2
#endif

/*****************************************************************/
/* NOTE: THIS CODE CANNOT BE RUN ON ARBITRARILY LARGE NUMBERS OF */
/* PROCESSORS. THE LARGEST VERIFIED NUMBER IS 1024. INCREASE     */
/* MAX_PROCS AT YOUR PERIL                                       */
/*****************************************************************/
#if CLASS == 'S'
#define  MAX_PROCS           128
#else
#define  MAX_PROCS           1024
#endif

#define  MAX_ITERATIONS      10
#define  TEST_ARRAY_SIZE     5

/*************************************/
/* Typedef: if necessary, change the */
/* size of int here by changing the  */
/* int type to, say, long            */
/*************************************/
typedef  int  INT_TYPE;
typedef  long INT_TYPE2;

/********************/
/* Some global info */
/********************/
INT_TYPE *key_buff_ptr_global,         /* used by full_verify to get */
         total_local_keys,             /* copies of rank info        */
         total_lesser_keys;

int      passed_verification;

INT_TYPE *key_array;
INT_TYPE *key_buff2;


/**********************/
/* Partial verif info */
/**********************/
INT_TYPE2 test_index_array[TEST_ARRAY_SIZE],
         test_rank_array[TEST_ARRAY_SIZE],

         S_test_index_array[TEST_ARRAY_SIZE] =
                             {48427,17148,23627,62548,4431},
         S_test_rank_array[TEST_ARRAY_SIZE] =
                             {0,18,346,64917,65463},

         W_test_index_array[TEST_ARRAY_SIZE] =
                             {357773,934767,875723,898999,404505},
         W_test_rank_array[TEST_ARRAY_SIZE] =
                             {1249,11698,1039987,1043896,1048018},

         A_test_index_array[TEST_ARRAY_SIZE] =
                             {2112377,662041,5336171,3642833,4250760},
         A_test_rank_array[TEST_ARRAY_SIZE] =
                             {104,17523,123928,8288932,8388264},

         B_test_index_array[TEST_ARRAY_SIZE] =
                             {41869,812306,5102857,18232239,26860214},
         B_test_rank_array[TEST_ARRAY_SIZE] =
                             {33422937,10244,59149,33135281,99},

         C_test_index_array[TEST_ARRAY_SIZE] =
                             {44172927,72999161,74326391,129606274,21736814},
         C_test_rank_array[TEST_ARRAY_SIZE] =
                             {61147,882988,266290,133997595,133525895},

         D_test_index_array[TEST_ARRAY_SIZE] =
                             {1317351170,995930646,1157283250,1503301535,1453734525},
         D_test_rank_array[TEST_ARRAY_SIZE] =
                             {1,36538729,1978098519,2145192618,2147425337};

/***********************/
/* function prototypes */
/***********************/
double	randlc( double *X, double *A );

/*****************************************************************/
/*************           R  A  N  D  L  C             ************/
/*************                                        ************/
/*************    portable random number generator    ************/
/*****************************************************************/

double	randlc( double *X, double *A )
{
      static int        KS=0;
      static double	R23, R46, T23, T46;
      double		T1, T2, T3, T4;
      double		A1;
      double		A2;
      double		X1;
      double		X2;
      double		Z;
      int     		i, j;

      if (KS == 0)
      {
        R23 = 1.0;
        R46 = 1.0;
        T23 = 1.0;
        T46 = 1.0;

        for (i=1; i<=23; i++)
        {
          R23 = 0.50 * R23;
          T23 = 2.0 * T23;
        }
        for (i=1; i<=46; i++)
        {
          R46 = 0.50 * R46;
          T46 = 2.0 * T46;
        }
        KS = 1;
      }

/*  Break A into two parts such that A = 2^23 * A1 + A2 and set X = N.  */

      T1 = R23 * *A;
      j  = T1;
      A1 = j;
      A2 = *A - T23 * A1;

/*  Break X into two parts such that X = 2^23 * X1 + X2, compute
    Z = A1 * X2 + A2 * X1  (mod 2^23), and then
    X = 2^23 * Z + A2 * X2  (mod 2^46).                            */

      T1 = R23 * *X;
      j  = T1;
      X1 = j;
      X2 = *X - T23 * X1;
      T1 = A1 * X2 + A2 * X1;

      j  = R23 * T1;
      T2 = j;
      Z = T1 - T23 * T2;
      T3 = T23 * Z + A2 * X2;
      j  = R46 * T3;
      T4 = j;
      *X = T3 - T46 * T4;
      return(R46 * *X);
}

/*****************************************************************/
/************   F  I  N  D  _  M  Y  _  S  E  E  D    ************/
/************                                         ************/
/************ returns parallel random number seq seed ************/
/*****************************************************************/

/*
 * Create a random number sequence of total length nn residing
 * on np number of processors.  Each processor will therefore have a
 * subsequence of length nn/np.  This routine returns that random
 * number which is the first random number for the subsequence belonging
 * to processor rank kn, and which is used as seed for proc kn ran # gen.
 */

double   find_my_seed( int  kn,       /* my processor rank, 0<=kn<=num procs */
                       int  np,       /* np = num procs                      */
                       long nn,       /* total num of ran numbers, all procs */
                       double s,      /* Ran num seed, for ex.: 314159265.00 */
                       double a )     /* Ran num gen mult, try 1220703125.00 */
{

  long   i;

  double t1,t2,t3,an;
  long   mq,nq,kk,ik;



      nq = nn / np;

      for( mq=0; nq>1; mq++,nq/=2 )
          ;

      t1 = a;

      for( i=1; i<=mq; i++ )
        t2 = randlc( &t1, &t1 );

      an = t1;

      kk = kn;
      t1 = s;
      t2 = an;

      for( i=1; i<=100; i++ )
      {
        ik = kk / 2;
        if( 2 * ik !=  kk )
            t3 = randlc( &t1, &t2 );
        if( ik == 0 )
            break;
        t3 = randlc( &t2, &t2 );
        kk = ik;
      }

      return( t1 );

}

/*****************************************************************/
/*************      C  R  E  A  T  E  _  S  E  Q      ************/
/*****************************************************************/

void create_seq( double seed, double a , INT_TYPE* key_array)
{
	double x;
	int    i, k;

    k = MAX_KEY/4;

	for (i=0; i<NUM_KEYS; i++)
	{
	    x = randlc(&seed, &a);
	    x += randlc(&seed, &a);
    	x += randlc(&seed, &a);
	    x += randlc(&seed, &a);

        key_array[i] = k*x;
	}
}

/*****************************************************************/
/*************    F  U  L  L  _  V  E  R  I  F  Y     ************/
/*****************************************************************/


void full_verify( void )
{
    MPI_Status  status;
    MPI_Request request;

    INT_TYPE    i, j;
    INT_TYPE    k, last_local_key;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    Timer timer;
    timer.start();
/*  Now, finally, sort the keys:  */
    for( i=0; i<total_local_keys; i++ )
        key_array[--key_buff_ptr_global[key_buff2[i]]-
                                 total_lesser_keys] = key_buff2[i];
    last_local_key = (total_local_keys<1)? 0 : (total_local_keys-1);

/*  Send largest key value to next processor  */
    if( my_rank > 0 )
        MPI_Irecv( &k,
                   1,
                   MPI_INT,
                   my_rank-1,
                   1000,
                   MPI_COMM_WORLD,
                   &request );
    if( my_rank < comm_size-1 )
        MPI_Send( &key_array[last_local_key],
                  1,
                  MPI_INT,
                  my_rank+1,
                  1000,
                  MPI_COMM_WORLD );
    if( my_rank > 0 )
        MPI_Wait( &request, &status );

/*  Confirm that neighbor's greatest key value
    is not greater than my least key value       */
    j = 0;
    if( my_rank > 0 && total_local_keys > 0 )
        if( k > key_array[0] ){
            j++;
            std::cout<<"rank"<<my_rank<<" "<<"at 0 out of order"<<std::endl;
        }

/*  Confirm keys correctly sorted: count incorrectly sorted keys, if any */
    for( i=1; i<total_local_keys; i++ )
        if( key_array[i-1] > key_array[i] ){
            j++;
            std::cout<<"rank"<<my_rank<<" "<<"at "<<i<<" out of order.."
            		<< key_array[i-1]<<":"<<key_array[i]<<std::endl;
        }


    if( j != 0 )
    {
        printf( "Processor %d:  Full_verify: number of keys out of sort: %d\n",
                my_rank, j );
    }
    else
        passed_verification++;

    double t_verify = timer.stop();

}

/*****************************************************************/
/*************             R  A  N  K             ****************/
/*****************************************************************/

class rank_kernel{
public:
	void init(INT_TYPE* key_array, INT_TYPE* key_buff2,  double* timeVal){
		myTrank = getTrank();
		myLrank = getLrank();
		if(getUniqueExecution()){
			this->key_array = key_array;
			this->key_buff2 = key_buff2;
			this->timeVal = timeVal;

			this->key_buff1 = (INT_TYPE*)malloc(SIZE_OF_BUFFERS*sizeof(INT_TYPE));
			this->bucket_size = (INT_TYPE*)malloc((NUM_BUCKETS+TEST_ARRAY_SIZE)*sizeof(INT_TYPE));
			this->bucket_size_totals = (INT_TYPE*)malloc((NUM_BUCKETS+TEST_ARRAY_SIZE)*sizeof(INT_TYPE));
			this->bucket_ptrs = (INT_TYPE*)malloc(NUM_BUCKETS*sizeof(INT_TYPE));
			this->process_bucket_distrib_ptr1 =
					(INT_TYPE*)malloc((NUM_BUCKETS+TEST_ARRAY_SIZE)*sizeof(INT_TYPE));
			this->process_bucket_distrib_ptr2 =
					(INT_TYPE*)malloc((NUM_BUCKETS+TEST_ARRAY_SIZE)*sizeof(INT_TYPE));
			this->bucket_size_array = (INT_TYPE*)malloc(NUM_BUCKETS*sizeof(INT_TYPE)*getLsize());
			sbarrier.set(getLsize());
			tmp_key_size_array = (INT_TYPE**)malloc(sizeof(INT_TYPE*)*getLsize());
		}
	}

	void run(int iteration){
		INT_TYPE    i, k;

		INT_TYPE    shift = MAX_KEY_LOG_2 - NUM_BUCKETS_LOG_2;
		INT_TYPE    key;
		INT_TYPE2   bucket_sum_accumulator, j, m;
		INT_TYPE    local_bucket_sum_accumulator;
		INT_TYPE    min_key_val, max_key_val;
		INT_TYPE    *key_buff_ptr;

		Timer timer1, timer2;
		double local_timeVal[4];
		INT_TYPE local_start;
		INT_TYPE local_end;
		int key_perthread = NUM_KEYS / getLsize();
		int key_residue = NUM_KEYS % getLsize();
		if(myLrank < key_residue){
			local_start = (key_perthread+1)*myLrank;
			local_end = local_start + key_perthread;
			key_perthread++;
		}
		else{
			local_start = (key_perthread+1)*key_residue + key_perthread*(myLrank-key_residue);
			local_end = local_start + key_perthread -1;
		}

		timer1.start();
		timer2.start();
		/*  Iteration alteration of keys */
		if(myTrank ==0){
			key_array[iteration] = iteration;
			key_array[iteration+MAX_ITERATIONS] = MAX_KEY - iteration;
		}
		if(myLrank ==0){
			/*  Initialize */
			for( i=0; i<NUM_BUCKETS+TEST_ARRAY_SIZE; i++ )
				bucket_size[i] = 0;
			/*  Determine where the partial verify test keys are, load into  */
			/*  top of array bucket_size                                     */
			for( i=0; i<TEST_ARRAY_SIZE; i++ )
				if( (test_index_array[i]/NUM_KEYS) == getPrank() )
					bucket_size[NUM_BUCKETS+i] =
								  key_array[test_index_array[i] % NUM_KEYS];
		}
		if(getUniqueExecution()){
			for( i=0; i<NUM_BUCKETS+TEST_ARRAY_SIZE; i++ )
				bucket_size_totals[i] = 0;
		}
		if(getUniqueExecution()){
			for( i=0; i<NUM_BUCKETS+TEST_ARRAY_SIZE; i++ )
				process_bucket_distrib_ptr1[i] = 0;
		}
		if(getUniqueExecution()){
			for( i=0; i<NUM_BUCKETS+TEST_ARRAY_SIZE; i++ )
				process_bucket_distrib_ptr2[i] = 0;
		}
		for(i=0; i<NUM_BUCKETS; i++)
			bucket_size_array[myLrank*NUM_BUCKETS + i]=0;
		//sbarrier.wait();
		/*  Determine the number of keys in each bucket */
		// compute by each thread for its local keys
		for(i= local_start; i<= local_end; i++){
			bucket_size_array[myLrank*NUM_BUCKETS + (key_array[i]>>shift)]++;
		}
		sbarrier.wait();
		//std::cout<<"here1"<<std::endl;
		/*  Accumulative bucket sizes are the bucket pointers.
		    These are global sizes accumulated upon to each bucket */
		INT_TYPE *bucket_ptrs_local = (INT_TYPE*)malloc(NUM_BUCKETS*sizeof(INT_TYPE));
		//std::cout<<"here2"<<std::endl;
		bucket_ptrs_local[0] = 0;
		for( k=0; k< myLrank; k++ )
			bucket_ptrs_local[0] += bucket_size_array[k*NUM_BUCKETS + 0];

		for( i=1; i< NUM_BUCKETS; i++ ) {
			bucket_ptrs_local[i] = bucket_ptrs_local[i-1];
			for( k=0; k< myLrank; k++ )
				bucket_ptrs_local[i] += bucket_size_array[k*NUM_BUCKETS+i];
			for( k=myLrank; k< getLsize(); k++ )
				bucket_ptrs_local[i] += bucket_size_array[k*NUM_BUCKETS+i-1];
		}
		/*  Sort into appropriate bucket */
		for(i = local_start; i<=local_end; i++){
			k = key_array[i];
			key_buff1[bucket_ptrs_local[k>>shift]++] = k;
		}
		// get bucket_size for current proc
		if(getUniqueExecution()){
			for(k =0; k<getLsize(); k++){
				for(i=0; i<NUM_BUCKETS; i++){
					bucket_size[i] += bucket_size_array[k*NUM_BUCKETS +i];
				}
			}
		}
		sbarrier.wait();
		local_timeVal[1]=timer2.stop();  // compute time
		//std::cout<<"here3"<<std::endl;
		if(myLrank==0){
			timer2.start();
			MPI_Allreduce( bucket_size,
			                   bucket_size_totals,
			                   NUM_BUCKETS+TEST_ARRAY_SIZE,
			                   MPI_INT,
			                   MPI_SUM,
			                   MPI_COMM_WORLD );

			local_timeVal[2] = timer2.stop();   // communicate time

		/*  Determine Redistibution of keys: accumulate the bucket size totals
		    till this number surpasses NUM_KEYS (which the average number of keys
		    per processor).  Then all keys in these buckets go to processor 0.
		    Continue accumulating again until supassing 2*NUM_KEYS. All keys
		    in these buckets go to processor 1, etc.  This algorithm guarantees
		    that all processors have work ranking; no processors are left idle.
		    The optimum number of buckets, however, does not result in as high
		    a degree of load balancing (as even a distribution of keys as is
		    possible) as is obtained from increasing the number of buckets, but
		    more buckets results in more computation per processor so that the
		    optimum number of buckets turns out to be 1024 for machines tested.
		    Note that process_bucket_distrib_ptr1 and ..._ptr2 hold the bucket
		    number of first and last bucket which each processor will have after
		    the redistribution is done.                                          */
			timer2.start();
			bucket_sum_accumulator = 0;
			local_bucket_sum_accumulator = 0;
			send_displ[0] = 0;
			process_bucket_distrib_ptr1[0] = 0;
			for( i=0, j=0; i<NUM_BUCKETS; i++ )
			{
				bucket_sum_accumulator       += bucket_size_totals[i];
				local_bucket_sum_accumulator += bucket_size[i];
				if( bucket_sum_accumulator >= (j+1)*NUM_KEYS )
				{
					send_count[j] = local_bucket_sum_accumulator;
					if( j != 0 )
					{
						send_displ[j] = send_displ[j-1] + send_count[j-1];
						process_bucket_distrib_ptr1[j] =
												process_bucket_distrib_ptr2[j-1]+1;
					}
					process_bucket_distrib_ptr2[j++] = i;
					local_bucket_sum_accumulator = 0;
				}
			}

		/*  When NUM_PROCS approaching NUM_BUCKETS, it is highly possible
			that the last few processors don't get any buckets.  So, we
			need to set counts properly in this case to avoid any fallouts.    */
			while( j < getPsize() )
			{
				send_count[j] = 0;
				process_bucket_distrib_ptr1[j] = 1;
				j++;
			}
			local_timeVal[1]+=timer2.stop(); // compute time

			timer2.start();
		/*  This is the redistribution section:  first find out how many keys
			each processor will send to every other processor:                 */
			MPI_Alltoall( send_count,
						  1,
						  MPI_INT,
						  recv_count,
						  1,
						  MPI_INT,
						  MPI_COMM_WORLD );

		/*  Determine the receive array displacements for the buckets */
			recv_displ[0] = 0;
			for( i=1; i<getPsize(); i++ )
				recv_displ[i] = recv_displ[i-1] + recv_count[i-1];


		/*  Now send the keys to respective processors  */
			MPI_Alltoallv( key_buff1,
						   send_count,
						   send_displ,
						   MPI_INT,
						   key_buff2,
						   recv_count,
						   recv_displ,
						   MPI_INT,
						   MPI_COMM_WORLD );
			local_timeVal[2]+= timer2.stop(); // communicate time
		}
		intra_Barrier();

		timer2.start();
	/*  The starting and ending bucket numbers on each processor are
		multiplied by the interval size of the buckets to obtain the
		smallest possible min and greatest possible max value of any
		key on each processor                                          */
		min_key_val = process_bucket_distrib_ptr1[getPrank()] << shift;
		max_key_val = ((process_bucket_distrib_ptr2[getPrank()] + 1) << shift)-1;
		//std::cout<<"min"<<min_key_val<<" max"<<max_key_val<<std::endl;
		if(getUniqueExecution()){
		/*  Clear the work array */
			for( i=0; i<max_key_val-min_key_val+1; i++ )
				key_buff1[i] = 0;
		}
		if(getUniqueExecution()){
		/*  Determine the total number of keys on all other
			processors holding keys of lesser value         */
			m = 0;
			for( k=0; k<getPrank(); k++ )
				for( i= process_bucket_distrib_ptr1[k];
					 i<=process_bucket_distrib_ptr2[k];
					 i++ )
					m += bucket_size_totals[i]; /*  m has total # of lesser keys */
			lesskeys=m;
		}
		if(getUniqueExecution()){
		/*  Determine total number of keys on this processor */
			j = 0;
			for( i= process_bucket_distrib_ptr1[getPrank()];
				 i<=process_bucket_distrib_ptr2[getPrank()];
				 i++ )
				j += bucket_size_totals[i];     /* j has total # of local keys   */
			totalkeys = j;
		}
		//std::cout<<"my buckets:"<<buckets_forcompute<<std::endl;
		sbarrier.wait();
		//key_buff_ptr = key_buff1 - min_key_val;

		key_perthread = totalkeys / getLsize();
		key_residue = totalkeys % getLsize();
		if(myLrank < key_residue){
			local_start = (key_perthread+1)*myLrank;
			local_end = local_start + key_perthread;
			key_perthread++;
		}
		else{
			local_start = (key_perthread+1)*key_residue + key_perthread*(myLrank-key_residue);
			local_end = local_start + key_perthread -1;
		}

		tmp_key_size_array[myLrank] = (INT_TYPE*)malloc(sizeof(INT_TYPE)*(max_key_val-min_key_val+1));
		for(i=0; i<max_key_val-min_key_val+1; i++)
			tmp_key_size_array[myLrank][i]=0;
		key_buff_ptr = tmp_key_size_array[myLrank]-min_key_val;
		for(i= local_start; i<=local_end; i++)
			key_buff_ptr[key_buff2[i]]++;
		key_perthread = (max_key_val-min_key_val+1) / getLsize();
		key_residue = (max_key_val-min_key_val+1) % getLsize();
		if(myLrank < key_residue){
			local_start = (key_perthread+1)*myLrank;
			local_end = local_start + key_perthread;
			key_perthread++;
		}
		else{
			local_start = (key_perthread+1)*key_residue + key_perthread*(myLrank-key_residue);
			local_end = local_start + key_perthread -1;
		}
		sbarrier.wait();
		for(i= local_start; i<=local_end; i++){
			for(k=0; k<getLsize(); k++)
				key_buff1[i]+=tmp_key_size_array[k][i];

		}
		key_buff_ptr = key_buff1 - min_key_val;

		intra_Barrier();
		if(myLrank==0){
			for( i=min_key_val; i<max_key_val; i++ )
				key_buff_ptr[i+1] += key_buff_ptr[i];

			/*key_buff1[0]=0;
			for(k=0; k< getLsize();k++)
				key_buff1[0]+=tmp_key_size_array[k][0];
			for(i=1; i<max_key_val-min_key_val+1;i++){
				key_buff1[i]=key_buff1[i-1];
				for(k=0; k< getLsize();k++)
					key_buff1[i]+=tmp_key_size_array[k][i];
			}*/

			/* This is the partial verify test section */
			/* Observe that test_rank_array vals are   */
			/* shifted differently for different cases */
			for( i=0; i<TEST_ARRAY_SIZE; i++ )
			{
				k = bucket_size_totals[i+NUM_BUCKETS];    /* Keys were hidden here */
				if( min_key_val <= k  &&  k <= max_key_val )
				{
					/* Add the total of lesser keys, m, here */
					INT_TYPE2 key_rank = key_buff_ptr[k-1] + lesskeys;
					int failed = 0;

					switch( CLASS )
					{
						case 'S':
							if( i <= 2 )
							{
								if( key_rank != test_rank_array[i]+iteration )
									failed = 1;
								else
									passed_verification++;
							}
							else
							{
								if( key_rank != test_rank_array[i]-iteration )
									failed = 1;
								else
									passed_verification++;
							}
							break;
						case 'W':
							if( i < 2 )
							{
								if( key_rank != test_rank_array[i]+(iteration-2) )
									failed = 1;
								else
									passed_verification++;
							}
							else
							{
								if( key_rank != test_rank_array[i]-iteration )
									failed = 1;
								else
									passed_verification++;
							}
							break;
						case 'A':
							if( i <= 2 )
						{
								if( key_rank != test_rank_array[i]+(iteration-1) )
									failed = 1;
								else
									passed_verification++;
						}
							else
							{
								if( key_rank != test_rank_array[i]-(iteration-1) )
									failed = 1;
								else
									passed_verification++;
							}
							break;
						case 'B':
							if( i == 1 || i == 2 || i == 4 )
						{
								if( key_rank != test_rank_array[i]+iteration )
									failed = 1;
								else
									passed_verification++;
						}
							else
							{
								if( key_rank != test_rank_array[i]-iteration )
									failed = 1;
								else
									passed_verification++;
							}
							break;
						case 'C':
							if( i <= 2 )
						{
								if( key_rank != test_rank_array[i]+iteration )
									failed = 1;
								else
									passed_verification++;
						}
							else
							{
								if( key_rank != test_rank_array[i]-iteration )
									failed = 1;
								else
									passed_verification++;
							}
							break;
						case 'D':
							if( i < 2 )
						{
								if( key_rank != test_rank_array[i]+iteration )
									failed = 1;
								else
									passed_verification++;
						}
							else
							{
								if( key_rank != test_rank_array[i]-iteration )
									failed = 1;
								else
									passed_verification++;
							}
							break;
					}
					if( failed == 1 )
						printf( "Failed partial verification: "
								"iteration %d, processor %d, test key %d\n",
								 iteration, getPrank(), (int)i );
				}
			}
		}
		sbarrier.wait();
		local_timeVal[1]+=timer2.stop();
		local_timeVal[0]+=timer1.stop();
		if(myLrank ==0){
			timeVal[0]= local_timeVal[0];
			timeVal[1]= local_timeVal[1];
			timeVal[2]= local_timeVal[2];
		}
		if( myLrank==0 && iteration == MAX_ITERATIONS )
		{
			key_buff_ptr_global = key_buff_ptr;
			total_local_keys    = totalkeys;
			total_lesser_keys   = 0;  /* no longer set to 'm', see note above */
		}

		free(bucket_ptrs_local);
		free(tmp_key_size_array[myLrank]);
	}

	~rank_kernel(){
		free(key_buff1);
		free(bucket_size);
		free(bucket_size_totals);
		free(bucket_ptrs);
		free(process_bucket_distrib_ptr1);
		free(process_bucket_distrib_ptr2);
		free(tmp_key_size_array);
	}

private:
	/* local */
	static thread_local int myTrank;
	static thread_local int myLrank;

	/* local shared */
	INT_TYPE  	*key_array,
				*key_buff1,
				*key_buff2,
				*bucket_size,
				*bucket_size_totals,
				*bucket_ptrs,
				*process_bucket_distrib_ptr1,
				*process_bucket_distrib_ptr2;
	int      send_count[MAX_PROCS], recv_count[MAX_PROCS],
	         send_displ[MAX_PROCS], recv_displ[MAX_PROCS];


	INT_TYPE *bucket_size_array;
	INT_TYPE lesskeys, totalkeys;
	INT_TYPE** tmp_key_size_array;

	double *timeVal;

	SpinBarrier sbarrier;
	SpinLock rwlock;

};
thread_local int rank_kernel::myTrank;
thread_local int rank_kernel::myLrank;



/*****************************************************************/
/*************             M  A  I  N             ****************/
/*****************************************************************/
void usage() {
    std::string help =
        "use option: -t nthreads   :threads per node of Task\n"
		"            -p nprocs       :number of nodes running on \n";
    std::cerr<<help.c_str();
    exit(-1);
}


int main(int argc, char* argv[]){
	key_array = (INT_TYPE*)malloc(SIZE_OF_BUFFERS*sizeof(INT_TYPE));
	key_buff2 = (INT_TYPE*)malloc(SIZE_OF_BUFFERS*sizeof(INT_TYPE));
	int iteration, itemp;

	UtcContext ctx(argc, argv);
	int nthreads=0;
	int nprocs=0;
	int opt;
	extern char *optarg;
	extern int optind;
	opt=getopt(argc, argv, "t:p:");
	//std::cout<<"opt="<<(char)opt<<std::endl;
	if(opt ==EOF)
		usage();
	while( opt!=EOF ){
		//std::cout<<"opt="<<(char)opt<<std::endl;
		switch (opt){
			case 't':
				//std::cout<<"optarg="<<optarg<<std::endl;
				nthreads = atoi(optarg);
				break;
			case 'p':
				nprocs = atoi(optarg);
				break;
			case '?':
				usage();
				break;
			default:
				usage();
				break;
		}
		opt=getopt(argc, argv, "t:p:");
	}
	if(nthreads<1 || nprocs<1){
		usage();
	}
	else{
		if(ctx.getProcRank()==0)
			std::cout<<"Threads: "<<nthreads<<"    Procs: "<<nprocs<<std::endl;
	}
	/*  Initialize the verification arrays if a valid class */
	for( int i=0; i<TEST_ARRAY_SIZE; i++ ){
		switch( CLASS )
		{
			case 'S':
				test_index_array[i] = S_test_index_array[i];
				test_rank_array[i]  = S_test_rank_array[i];
				break;
			case 'A':
				test_index_array[i] = A_test_index_array[i];
				test_rank_array[i]  = A_test_rank_array[i];
				break;
			case 'W':
				test_index_array[i] = W_test_index_array[i];
				test_rank_array[i]  = W_test_rank_array[i];
				break;
			case 'B':
				test_index_array[i] = B_test_index_array[i];
				test_rank_array[i]  = B_test_rank_array[i];
				break;
			case 'C':
				test_index_array[i] = C_test_index_array[i];
				test_rank_array[i]  = C_test_rank_array[i];
				break;
			case 'D':
				test_index_array[i] = D_test_index_array[i];
				test_rank_array[i]  = D_test_rank_array[i];
				break;
		}
	}


	if(ctx.getProcRank()==0){
		printf( "\n\n NAS Parallel Benchmarks 3.3 -- IS Benchmark\n\n" );
		printf( " Size:  %ld  (class %c)\n", (long)TOTAL_KEYS*MIN_PROCS, CLASS );
		printf( " Iterations:   %d\n", MAX_ITERATIONS );
	}

	if(ctx.numProcs()!= NUM_PROCS){
		if( ctx.getProcRank() == 0 )
		printf( "\n ERROR: compiled for %d processes\n"
				" Number of active processes: %d\n"
				" Exiting program!\n\n", NUM_PROCS, ctx.numProcs() );
		exit(1);
	}

	if(NUM_PROCS< MIN_PROCS || NUM_PROCS >MAX_PROCS){
		if( ctx.getProcRank() == 0 )
		   printf( "\n ERROR: number of processes %d not within range %d-%d"
				   "\n Exiting program!\n\n", NUM_PROCS, MIN_PROCS, MAX_PROCS);
	   exit( 1 );
	}

	/*  Generate random number sequence and subsequent keys on all procs */
	create_seq( find_my_seed( ctx.getProcRank(),
							  ctx.numProcs(),
							  4*(long)TOTAL_KEYS*MIN_PROCS,
							  314159265.00,      /* Random number gen seed */
							  1220703125.00 ),   /* Random number gen mult */
				1220703125.00, key_array);                 /* Random number gen mult */

	std::vector<int> rvec;
	for(int i=0; i< nprocs; i++)
			for(int j=0; j<nthreads;j++)
				rvec.push_back(i);
	ProcList rlist(rvec);
	Task<rank_kernel> is_rank_instance("is-rank-compute", rlist);

	double timeVal[3]={0, 0, 0};
	double timeVal2[3]={0,0,0};
	is_rank_instance.init(key_array, key_buff2, timeVal);
	//is_rank_instance.run(1);
	//is_rank_instance.wait();
	ctx.Barrier();

	passed_verification =0;
	if( ctx.getProcRank() == 0 && CLASS != 'S' )
		printf( "\n   iteration\n" );

	Timer timer;
	double totaltime=0;
	timer.start();
	for(int iteration=1; iteration<= MAX_ITERATIONS; iteration++){
		if( ctx.getProcRank() == 0 && CLASS != 'S' )
			printf( "        %d\n", iteration );
		is_rank_instance.run(iteration);
		is_rank_instance.wait();
		timeVal2[0]+=timeVal[0];
		timeVal2[1]+=timeVal[1];
		timeVal2[2]+=timeVal[2];
	}
	totaltime = timer.stop();
	double max_totaltime=0;
	ctx.Barrier();
	MPI_Reduce( &totaltime,&max_totaltime,1,MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
	double timeValgather[3];
	MPI_Reduce(timeVal2, timeValgather, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	is_rank_instance.finish();

	full_verify();
	/*  Obtain verification counter sum */
	itemp = passed_verification;
	MPI_Reduce( &itemp,
				&passed_verification,
				1,
				MPI_INT,
				MPI_SUM,
				0,
				MPI_COMM_WORLD );
	/*  The final printout  */
	if(ctx.getProcRank()==0){
		if( passed_verification != 5*MAX_ITERATIONS + ctx.numProcs() )
			passed_verification = 0;
		double Mops = ((double) (MAX_ITERATIONS)*TOTAL_KEYS*MIN_PROCS)
                                                              /max_totaltime/1000000.0;
		c_print_results("IS", CLASS, (int)TOTAL_KEYS, MIN_PROCS, 0,
				MAX_ITERATIONS, max_totaltime, Mops,
				"keys ranked",
				passed_verification,
				NPBVERSION, COMPILETIME, CC,
			    CLINK, C_LIB, C_INC, CFLAGS, CLINKFLAGS);

		/* print timing information */
		for(int i=0; i<3; i++)
			timeVal[i] = timeValgather[i]/ctx.numProcs();
		printf("\nTotal time:         %10.4lf \n", max_totaltime);
		printf("\nTotal run time:     %10.4lf \n", timeVal[0]);
		printf("\ncompute time:       %10.4lf \n", timeVal[1]);
		printf("\ncomm time:          %10.4lf \n", timeVal[2]);
		printf( "\n" );
	}


	return 0;
}





