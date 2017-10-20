/*
 * mc_task.cu
 *
 *  Created on: Oct 5, 2017
 *      Author: chaoliu
 */

#include "mc_task.h"
#include "mc_kernel.h"
#include "../../../common/helper_err.h"

//#include "curand.h"

void IntegralCaculator::initImpl(long loopN, unsigned int seed, double range_lower, double range_upper)
{
	if(__localThreadId==0)
	{
		m_seed = seed;
		m_range_lower = range_lower;
		m_range_upper = range_upper;
		m_loopN = loopN/__numGlobalThreads;
		m_res = 0;
	}
	inter_Barrier();
	if(__globalThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}

}

void IntegralCaculator::runImpl(double runtime[][5], int bsize, int gsize){
	Timer timer, timer0;

	int blocksize = bsize;
	int gridsize = gsize;
	long loopPerThread = (m_loopN+blocksize*gridsize-1)/(blocksize*gridsize);
	dim3 block(blocksize, 1, 1);
	dim3 grid(gridsize, 1, 1);
	GpuData<double> res_d(gridsize);

	inter_Barrier();
	timer0.start();
	timer.start();
	double copytime = 0;

	/*
	 * call kernel
	 */
	timer.start();
	mc_kernel<<<grid, block, 0, __streamId>>>(
			res_d.getD(true),
			m_range_upper,
			m_range_lower,
			m_seed,
			loopPerThread);
	cudaStreamSynchronize(__streamId);
	checkCudaErr(cudaGetLastError());
	double kernelTime = timer.stop();

	/*
	 * copy data out
	 */
	timer.start();
	res_d.sync();
	copytime = timer.stop();

	/*
	 * reduce local results and to compute final
	 */
	timer.start();
	double local_res = 0;
	for(int i =0; i<res_d.getSize(); i++){
		local_res += res_d.at(i);
	}
	//__fastSpinLock.lock();
	m_res = local_res*(m_range_upper - m_range_lower)/m_loopN;
	//__fastSpinLock.unlock();
	double hostCompTime = timer.stop();

	/*__fastIntraSync.wait();
	double *res_gather;
	if(__localThreadId==0){
		res_gather = (double*)malloc(__numGroupProcesses*sizeof(double));
	}
	*/
	double res_reduce;
	timer.start();
	//TaskGatherBy<double>(this, &m_res, 1, res_gather, 1, 0);
	TaskReduceSumBy<double>(this, &m_res,&res_reduce, 1, 0);
	__fastIntraSync.wait();
	double commtime = timer.stop();
	double result = 0.0;
	result = res_reduce / __numGroupProcesses;
	/*
	if(__globalThreadId==0){
		for(int i=0; i<__numGroupProcesses; i++){
			result += res_gather[i];
		}
		result /= __numGlobalThreads;
		std::cout<<result<<std::endl;
	}
	*/
	inter_Barrier();
	double totalTime = timer0.stop();
	if(__globalThreadId == 0)
		std::cout<<result<<std::endl;

	runtime[__localThreadId][0] = totalTime;
	runtime[__localThreadId][1] = kernelTime + hostCompTime;
	runtime[__localThreadId][2] = kernelTime;
	runtime[__localThreadId][3] = copytime;
	runtime[__localThreadId][4] = commtime;

	if(__globalThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}

}




