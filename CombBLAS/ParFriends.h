#ifndef _PAR_FRIENDS_H_
#define _PAR_FRIENDS_H_

#include "mpi.h"
#include "sys/time.h"
#include <iostream>
#include "SpParMat.h"	
#include "SpParHelper.h"
#include "MPIType.h"
#include "Friends.h"

using namespace std;

template <class IT, class NT, class DER>
class SpParMat;

/*************************************************************************************************/
/**************************** FRIEND FUNCTIONS FOR PARALLEL CLASSES ******************************/
/*************************************************************************************************/


/**
 * Parallel A = B*C routine that uses only MPI-1 features
 * Relies on simple blocking broadcast
 **/  
template <typename SR, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB> 
SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDERA,UDERB>::T_promote> Mult_AnXBn_Synch 
		(const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B )

{
	typedef typename promote_trait<NU1,NU2>::T_promote N_promote;
	typedef typename promote_trait<UDERA,UDERB>::T_promote DER_promote;
	
	if(A.getncol() != B.getnrow())
	{
		cout<<"Can not multiply, dimensions does not match"<<endl;
		MPI::COMM_WORLD.Abort(DIMMISMATCH);
		return SpParMat< IU,N_promote,DER_promote >();
	}

	int stages, dummy; 	// last two parameters of ProductGrid are ignored for Synch multiplication
	shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, dummy, dummy);		
	IU C_m = A.spSeq->getnrow();
	IU C_n = B.spSeq->getncol();
	
	const_cast< UDERB* >(B.spSeq)->Transpose();	
	GridC->GetWorld().Barrier();

	IU ** ARecvSizes = SpHelper::allocate2D<IU>(UDERA::esscount, stages);
	IU ** BRecvSizes = SpHelper::allocate2D<IU>(UDERB::esscount, stages);
	
	SpParHelper::GetSetSizes( *(A.spSeq), ARecvSizes, (A.commGrid)->GetRowWorld());
	SpParHelper::GetSetSizes( *(B.spSeq), BRecvSizes, (B.commGrid)->GetColWorld());

	// Remotely fetched matrices are stored as pointers
	UDERA * ARecv; 
	UDERB * BRecv;
	vector< SpTuples<IU,N_promote>  *> tomerge;

	int Aself = (A.commGrid)->GetRankInProcRow();
	int Bself = (B.commGrid)->GetRankInProcCol();	

	for(int i = 0; i < stages; ++i) 
	{
		vector<IU> ess;	
		if(i == Aself)
		{	
			ARecv = A.spSeq;	// shallow-copy 
		}
		else
		{
			ess.resize(UDERA::esscount);
			for(int j=0; j< UDERA::esscount; ++j)	
			{
				ess[j] = ARecvSizes[j][i];		// essentials of the ith matrix in this row	
			}
			ARecv = new UDERA();				// first, create the object
		}

		SpParHelper::BCastMatrix(GridC->GetRowWorld(), *ARecv, ess, i);	// then, receive its elements	
		ess.clear();	
		
		if(i == Bself)
		{
			BRecv = B.spSeq;	// shallow-copy
		}
		else
		{
			ess.resize(UDERB::esscount);		
			for(int j=0; j< UDERB::esscount; ++j)	
			{
				ess[j] = BRecvSizes[j][i];	
			}	
			BRecv = new UDERB();
		}
		SpParHelper::BCastMatrix(GridC->GetColWorld(), *BRecv, ess, i);	// then, receive its elements

		SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>(*ARecv, *BRecv, false, true);
		if(!C_cont->isZero()) 
			tomerge.push_back(C_cont);

		if(i != Aself)	
		{
			delete ARecv;		
		}
		if(i != Bself)	
		{
			delete BRecv;
		}
	}

	SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
	SpHelper::deallocate2D(BRecvSizes, UDERB::esscount);
			
	DER_promote * C = new DER_promote(MergeAll<SR>(tomerge, C_m, C_n), false, NULL);	// First get the result in SpTuples, then convert to UDER
	for(int i=0; i<tomerge.size(); ++i)
	{
		delete tomerge[i];
	}
	const_cast< UDERB* >(B.spSeq)->Transpose();	// transpose back to original
	
	return SpParMat<IU,N_promote,DER_promote> (C, GridC);		// return the result object
}

/**
 * Parallel A = B*C routine that uses one-sided MPI-2 features
 * General active target syncronization via MPI_Win_Post, MPI_Win_Start, MPI_Win_Complete, MPI_Win_Wait
 * Tested on my dual core Macbook with 1,4,9,16,25 MPI processes
 * No memory hog: splits the matrix into two along the column, prefetches the next half matrix while computing on the current one 
 **/  
template <typename SR, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB> 
SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDERA,UDERB>::T_promote> Mult_AnXBn_ActiveTarget 
		(const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B )

{
	typedef typename promote_trait<NU1,NU2>::T_promote N_promote;
	typedef typename promote_trait<UDERA,UDERB>::T_promote DER_promote;

	if(A.getncol() != B.getnrow())
	{
		cout<<"Can not multiply, dimensions does not match"<<endl;
		MPI::COMM_WORLD.Abort(DIMMISMATCH);
		return SpParMat< IU,N_promote,DER_promote >();
	}
	int stages, Aoffset, Boffset; 	// stages = inner dimension of matrix blocks
	shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, Aoffset, Boffset);		

	IU C_m = A.spSeq->getnrow();
	IU C_n = B.spSeq->getncol();
		
	UDERA A1seq, A2seq;
	(A.spSeq)->Split( A1seq, A2seq); 
	
	// ABAB: It should be able to perform split/merge with the transpose option [single function call]
	const_cast< UDERB* >(B.spSeq)->Transpose();
	
	UDERB B1seq, B2seq;
	(B.spSeq)->Split( B1seq, B2seq);
	
	// Create row and column windows (collective operation, i.e. everybody exposes its window to others)
	vector<MPI::Win> rowwins1, rowwins2, colwins1, colwins2;
	SpParHelper::SetWindows((A.commGrid)->GetRowWorld(), A1seq, rowwins1);
	SpParHelper::SetWindows((A.commGrid)->GetRowWorld(), A2seq, rowwins2);
	SpParHelper::SetWindows((B.commGrid)->GetColWorld(), B1seq, colwins1);
	SpParHelper::SetWindows((B.commGrid)->GetColWorld(), B2seq, colwins2);

	SpParHelper::SetWinErrHandler(rowwins1);	// set the error handler to THROW_EXCEPTIONS
	SpParHelper::SetWinErrHandler(rowwins2);	
	SpParHelper::SetWinErrHandler(colwins1);	
	SpParHelper::SetWinErrHandler(colwins2);	

	// ABAB: We can optimize the call to create windows in the absence of passive synchronization
	// 	MPI_Info info; 
	// 	MPI_Info_create ( &info ); 
	// 	MPI_Info_set( info, "no_locks", "true" ); 
	// 	MPI_Win_create( . . ., info, . . . ); 
	// 	MPI_Info_free( &info ); 
	
	IU ** ARecvSizes1 = SpHelper::allocate2D<IU>(UDERA::esscount, stages);
	IU ** ARecvSizes2 = SpHelper::allocate2D<IU>(UDERA::esscount, stages);
	IU ** BRecvSizes1 = SpHelper::allocate2D<IU>(UDERB::esscount, stages);
	IU ** BRecvSizes2 = SpHelper::allocate2D<IU>(UDERB::esscount, stages);
		
	SpParHelper::GetSetSizes( A1seq, ARecvSizes1, (A.commGrid)->GetRowWorld());
	SpParHelper::GetSetSizes( A2seq, ARecvSizes2, (A.commGrid)->GetRowWorld());
	SpParHelper::GetSetSizes( B1seq, BRecvSizes1, (B.commGrid)->GetColWorld());
	SpParHelper::GetSetSizes( B2seq, BRecvSizes2, (B.commGrid)->GetColWorld());
	
	// Remotely fetched matrices are stored as pointers
	UDERA * ARecv1, * ARecv2; 
	UDERB * BRecv1, * BRecv2;
	vector< SpTuples<IU,N_promote>  *> tomerge;

	MPI::Group row_group = (A.commGrid)->GetRowWorld().Get_group();
	MPI::Group col_group = (B.commGrid)->GetColWorld().Get_group();

	int Aself = (A.commGrid)->GetRankInProcRow();
	int Bself = (B.commGrid)->GetRankInProcCol();	
	
	GridC->GetWorld().Barrier();	

	SpParHelper::Print("Writing to file\n");
	ofstream oput;
	GridC->OpenDebugFile("deb", oput);

	oput << "A1seq: " << A1seq.getnrow() << " " << A1seq.getncol() << " " << A1seq.getnnz() << endl;
	oput << "A2seq: " << A2seq.getnrow() << " " << A2seq.getncol() << " " << A2seq.getnnz() << endl;
	oput << "B1seq: " << B1seq.getnrow() << " " << B1seq.getncol() << " " << B1seq.getnnz() << endl;
	oput << "B2seq: " << B2seq.getnrow() << " " << B2seq.getncol() << " " << B2seq.getnnz() << endl;

	SpParHelper::Print("Wrote to file\n");
	GridC->GetWorld().Barrier();
	
	// Start exposure epochs to all windows
	try
	{
		SpParHelper::PostExposureEpoch(Aself, rowwins1, row_group);
		SpParHelper::PostExposureEpoch(Aself, rowwins2, row_group);
		SpParHelper::PostExposureEpoch(Bself, colwins1, col_group);
		SpParHelper::PostExposureEpoch(Bself, colwins2, col_group);
	}
    	catch(MPI::Exception e)
	{
		oput << "Exception while posting exposure epoch" << endl;
       		oput << e.Get_error_string() << endl;
     	}

	GridC->GetWorld().Barrier();
	SpParHelper::Print("Exposure epochs posted\n");	
	GridC->GetWorld().Barrier();

	
	int Aowner = (0+Aoffset) % stages;		
	int Bowner = (0+Boffset) % stages;
	try
	{	
		SpParHelper::AccessNFetch(ARecv1, Aowner, rowwins1, row_group, ARecvSizes1);
		SpParHelper::AccessNFetch(ARecv2, Aowner, rowwins2, row_group, ARecvSizes2);	// Start prefetching next half 

		for(int j=0; j< rowwins1.size(); ++j)	// wait for the first half to complete
			rowwins1[j].Complete();
		
		SpParHelper::AccessNFetch(BRecv1, Bowner, colwins1, col_group, BRecvSizes1);
		SpParHelper::AccessNFetch(BRecv2, Bowner, colwins2, col_group, BRecvSizes2);	// Start prefetching next half 
			
		for(int j=0; j< colwins1.size(); ++j)
			colwins1[j].Complete();
	}

    	catch(MPI::Exception e)
	{
		oput << "Exception while starting access epoch or the first fetch" << endl;
       		oput << e.Get_error_string() << endl;
     	}
	
	for(int i = 1; i < stages; ++i) 
	{
		SpParHelper::Print("Stage starting\n");
		SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>(*ARecv1, *BRecv1, false, true);

		SpParHelper::Print("Multiplied\n");

		if(!C_cont->isZero()) 
			tomerge.push_back(C_cont);

		SpParHelper::Print("Pushed back\n");

		
		GridC->GetWorld().Barrier();
		bool remoteA = false;
		bool remoteB = false;

		delete ARecv1;		// free the memory of the previous first half
		for(int j=0; j< rowwins2.size(); ++j)	// wait for the previous second half to complete
			rowwins2[j].Complete();
		SpParHelper::Print("Completed A\n");

		delete BRecv1;
		for(int j=0; j< colwins2.size(); ++j)	// wait for the previous second half to complete
			colwins2[j].Complete();
		
		SpParHelper::Print("Completed B\n");

		
		GridC->GetWorld().Barrier();
		Aowner = (i+Aoffset) % stages;		
		Bowner = (i+Boffset) % stages;
	

		// start fetching the current first half 
		SpParHelper::AccessNFetch(ARecv1, Aowner, rowwins1, row_group, ARecvSizes1);
		SpParHelper::AccessNFetch(BRecv1, Bowner, colwins1, col_group, BRecvSizes1);
	
		SpParHelper::Print("Fetched next\n");
		
		GridC->GetWorld().Barrier();
		// while multiplying the already completed previous second halfs
		C_cont = MultiplyReturnTuples<SR>(*ARecv2, *BRecv2, false, true);	
		if(!C_cont->isZero()) 
			tomerge.push_back(C_cont);

		SpParHelper::Print("Multiplied and pushed\n");
		GridC->GetWorld().Barrier();

		delete ARecv2;		// free memory of the previous second half
		delete BRecv2;

		for(int j=0; j< rowwins1.size(); ++j)	// wait for the current first half to complte
			rowwins1[j].Complete();
		for(int j=0; j< colwins1.size(); ++j)
			colwins1[j].Complete();
		
		SpParHelper::Print("Completed next\n");	
		GridC->GetWorld().Barrier();

		// start prefetching the current second half 
		SpParHelper::AccessNFetch(ARecv2, Aowner, rowwins2, row_group, ARecvSizes2);
		SpParHelper::AccessNFetch(BRecv2, Bowner, colwins2, col_group, BRecvSizes2);
	}
	//SpParHelper::Print("Stages finished\n");
	SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>(*ARecv1, *BRecv1, false, true);
	if(!C_cont->isZero()) 
		tomerge.push_back(C_cont);

	delete ARecv1;		// free the memory of the previous first half
	for(int j=0; j< rowwins2.size(); ++j)	// wait for the previous second half to complete
		rowwins2[j].Complete();
	delete BRecv1;
	for(int j=0; j< colwins2.size(); ++j)	// wait for the previous second half to complete
		colwins2[j].Complete();	

	C_cont = MultiplyReturnTuples<SR>(*ARecv2, *BRecv2, false, true);	
	if(!C_cont->isZero()) 
		tomerge.push_back(C_cont);
		
	delete ARecv2;
	delete BRecv2;

	SpHelper::deallocate2D(ARecvSizes1, UDERA::esscount);
	SpHelper::deallocate2D(ARecvSizes2, UDERA::esscount);
	SpHelper::deallocate2D(BRecvSizes1, UDERB::esscount);
	SpHelper::deallocate2D(BRecvSizes2, UDERB::esscount);
			
	DER_promote * C = new DER_promote(MergeAll<SR>(tomerge, C_m, C_n), false, NULL);	// First get the result in SpTuples, then convert to UDER
	for(int i=0; i<tomerge.size(); ++i)
	{
		delete tomerge[i];
	}

	// MPI_Win_Wait() works like a barrier as it waits for all origins to finish their remote memory operation on "this" window
	SpParHelper::WaitNFree(rowwins1);
	SpParHelper::WaitNFree(rowwins2);
	SpParHelper::WaitNFree(colwins1);
	SpParHelper::WaitNFree(colwins2);	
	
	(A.spSeq)->Merge(A1seq, A2seq);
	(B.spSeq)->Merge(B1seq, B2seq);	
	
	const_cast< UDERB* >(B.spSeq)->Transpose();	// transpose back to original
	return SpParMat<IU,N_promote,DER_promote> (C, GridC);		// return the result object
}


/**
 * Parallel A = B*C routine that uses one-sided MPI-2 features
 * Passive target syncronization via MPI_Win_Lock, MPI_Win_Unlock
 * No memory hog: splits the matrix into two along the column, prefetches the next half matrix while computing on the current one 
 **/  
template <typename SR, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB> 
SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDERA,UDERB>::T_promote> Mult_AnXBn_PassiveTarget 
		(const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B )

{
	typedef typename promote_trait<NU1,NU2>::T_promote N_promote;
	typedef typename promote_trait<UDERA,UDERB>::T_promote DER_promote;

	if(A.getncol() != B.getnrow())
	{
		cout<<"Can not multiply, dimensions does not match"<<endl;
		MPI::COMM_WORLD.Abort(DIMMISMATCH);
		return SpParMat< IU,N_promote,DER_promote >();
	}
	int stages, Aoffset, Boffset; 	// stages = inner dimension of matrix blocks
	shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, Aoffset, Boffset);		

	IU C_m = A.spSeq->getnrow();
	IU C_n = B.spSeq->getncol();

	UDERA A1seq, A2seq;
	(A.spSeq)->Split( A1seq, A2seq); 
	
	// ABAB: It should be able to perform split/merge with the transpose option [single function call]
	const_cast< UDERB* >(B.spSeq)->Transpose();
	
	UDERB B1seq, B2seq;
	(B.spSeq)->Split( B1seq, B2seq);
	
	// Create row and column windows (collective operation, i.e. everybody exposes its window to others)
	vector<MPI::Win> rowwins1, rowwins2, colwins1, colwins2;
	SpParHelper::SetWindows((A.commGrid)->GetRowWorld(), A1seq, rowwins1);
	SpParHelper::SetWindows((A.commGrid)->GetRowWorld(), A2seq, rowwins2);
	SpParHelper::SetWindows((B.commGrid)->GetColWorld(), B1seq, colwins1);
	SpParHelper::SetWindows((B.commGrid)->GetColWorld(), B2seq, colwins2);

	IU ** ARecvSizes1 = SpHelper::allocate2D<IU>(UDERA::esscount, stages);
	IU ** ARecvSizes2 = SpHelper::allocate2D<IU>(UDERA::esscount, stages);
	IU ** BRecvSizes1 = SpHelper::allocate2D<IU>(UDERB::esscount, stages);
	IU ** BRecvSizes2 = SpHelper::allocate2D<IU>(UDERB::esscount, stages);
		
	SpParHelper::GetSetSizes( A1seq, ARecvSizes1, (A.commGrid)->GetRowWorld());
	SpParHelper::GetSetSizes( A2seq, ARecvSizes2, (A.commGrid)->GetRowWorld());
	SpParHelper::GetSetSizes( B1seq, BRecvSizes1, (B.commGrid)->GetColWorld());
	SpParHelper::GetSetSizes( B2seq, BRecvSizes2, (B.commGrid)->GetColWorld());

	// Remotely fetched matrices are stored as pointers
	UDERA * ARecv1, * ARecv2; 
	UDERB * BRecv1, * BRecv2;
	vector< SpTuples<IU,N_promote> *> tomerge;	// sorted triples to be merged

	MPI::Group row_group = (A.commGrid)->GetRowWorld().Get_group();
	MPI::Group col_group = (B.commGrid)->GetColWorld().Get_group();

	int Aself = (A.commGrid)->GetRankInProcRow();
	int Bself = (B.commGrid)->GetRankInProcCol();	
	
	int Aowner = (0+Aoffset) % stages;		
	int Bowner = (0+Boffset) % stages;
	
	SpParHelper::LockNFetch(ARecv1, Aowner, rowwins1, row_group, ARecvSizes1);
	SpParHelper::LockNFetch(ARecv2, Aowner, rowwins2, row_group, ARecvSizes2);	// Start prefetching next half 
	SpParHelper::LockNFetch(BRecv1, Bowner, colwins1, col_group, BRecvSizes1);
	SpParHelper::LockNFetch(BRecv2, Bowner, colwins2, col_group, BRecvSizes2);	// Start prefetching next half 
		
	// Finish the first halfs
	SpParHelper::UnlockWindows(Aowner, rowwins1);
	SpParHelper::UnlockWindows(Bowner, colwins1);

	for(int i = 1; i < stages; ++i) 
	{
		SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>(*ARecv1, *BRecv1, false, true);

		if(!C_cont->isZero()) 
			tomerge.push_back(C_cont);

		bool remoteA = false;
		bool remoteB = false;

		delete ARecv1;		// free the memory of the previous first half
		delete BRecv1;

		SpParHelper::UnlockWindows(Aowner, rowwins2);	// Finish the second half
		SpParHelper::UnlockWindows(Bowner, colwins2);	

		Aowner = (i+Aoffset) % stages;		
		Bowner = (i+Boffset) % stages;

		// start fetching the current first half 
		SpParHelper::LockNFetch(ARecv1, Aowner, rowwins1, row_group, ARecvSizes1);
		SpParHelper::LockNFetch(BRecv1, Bowner, colwins1, col_group, BRecvSizes1);
	
		// while multiplying the already completed previous second halfs
		C_cont = MultiplyReturnTuples<SR>(*ARecv2, *BRecv2, false, true);	
		if(!C_cont->isZero()) 
			tomerge.push_back(C_cont);

		delete ARecv2;		// free memory of the previous second half
		delete BRecv2;

		// wait for the current first half to complte
		SpParHelper::UnlockWindows(Aowner, rowwins1);
		SpParHelper::UnlockWindows(Bowner, colwins1);
		
		// start prefetching the current second half 
		SpParHelper::LockNFetch(ARecv2, Aowner, rowwins2, row_group, ARecvSizes2);
		SpParHelper::LockNFetch(BRecv2, Bowner, colwins2, col_group, BRecvSizes2);
	}

	SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>(*ARecv1, *BRecv1, false, true);
	if(!C_cont->isZero()) 
		tomerge.push_back(C_cont);

	delete ARecv1;		// free the memory of the previous first half
	delete BRecv1;
	
	SpParHelper::UnlockWindows(Aowner, rowwins2);
	SpParHelper::UnlockWindows(Bowner, colwins2);

	C_cont = MultiplyReturnTuples<SR>(*ARecv2, *BRecv2, false, true);	
	if(!C_cont->isZero()) 
		tomerge.push_back(C_cont);		
		
	delete ARecv2;
	delete BRecv2;

	SpHelper::deallocate2D(ARecvSizes1, UDERA::esscount);
	SpHelper::deallocate2D(ARecvSizes2, UDERA::esscount);
	SpHelper::deallocate2D(BRecvSizes1, UDERB::esscount);
	SpHelper::deallocate2D(BRecvSizes2, UDERB::esscount);
			
	DER_promote * C = new DER_promote(MergeAll<SR>(tomerge, C_m, C_n), false, NULL);	// First get the result in SpTuples, then convert to UDER
	for(int i=0; i<tomerge.size(); ++i)
	{
		delete tomerge[i];
	}
	
	SpParHelper::FreeWindows(rowwins1);
	SpParHelper::FreeWindows(rowwins2);
	SpParHelper::FreeWindows(colwins1);
	SpParHelper::FreeWindows(colwins2);	

	(A.spSeq)->Merge(A1seq, A2seq);
	(B.spSeq)->Merge(B1seq, B2seq);	
	
	const_cast< UDERB* >(B.spSeq)->Transpose();	// transpose back to original

	return SpParMat<IU,N_promote,DER_promote> (C, GridC);		// return the result object
}

/**
 * Parallel A = B*C routine that uses one-sided MPI-2 features
 * Syncronization is through MPI_Win_Fence
 * Buggy as of September, 2009
 **/ 
template <typename SR, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB> 
SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDERA,UDERB>::T_promote> Mult_AnXBn_Fence
		(const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B )
{
	typedef typename promote_trait<NU1,NU2>::T_promote N_promote;
	typedef typename promote_trait<UDERA,UDERB>::T_promote DER_promote;
	
	if(A.getncol() != B.getnrow())
	{
		cout<<"Can not multiply, dimensions does not match"<<endl;
		MPI::COMM_WORLD.Abort(DIMMISMATCH);
		return SpParMat< IU,N_promote,DER_promote >();
	}

	int stages, Aoffset, Boffset; 	// stages = inner dimension of matrix blocks
	shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, Aoffset, Boffset);		
			
	ofstream oput;
	GridC->OpenDebugFile("deb", oput);
	const_cast< UDERB* >(B.spSeq)->Transpose();
	
	// set row & col window handles
	vector<MPI::Win> rowwindows, colwindows;
	vector<MPI::Win> rowwinnext, colwinnext;
	SpParHelper::SetWindows((A.commGrid)->GetRowWorld(), *(A.spSeq), rowwindows);
	SpParHelper::SetWindows((B.commGrid)->GetColWorld(), *(B.spSeq), colwindows);
	SpParHelper::SetWindows((A.commGrid)->GetRowWorld(), *(A.spSeq), rowwinnext);
	SpParHelper::SetWindows((B.commGrid)->GetColWorld(), *(B.spSeq), colwinnext);
	
	IU ** ARecvSizes = SpHelper::allocate2D<IU>(UDERA::esscount, stages);
	IU ** BRecvSizes = SpHelper::allocate2D<IU>(UDERB::esscount, stages);
	
	SpParHelper::GetSetSizes( *(A.spSeq), ARecvSizes, (A.commGrid)->GetRowWorld());
	SpParHelper::GetSetSizes( *(B.spSeq), BRecvSizes, (B.commGrid)->GetColWorld());
	
	UDERA * ARecv, * ARecvNext; 
	UDERB * BRecv, * BRecvNext;
	vector< SpTuples<IU,N_promote>  *> tomerge;

	// Prefetch first
	for(int j=0; j< rowwindows.size(); ++j)
		MPI_Win_fence(MPI_MODE_NOPRECEDE, rowwindows[j]);
	for(int j=0; j< colwindows.size(); ++j)
		MPI_Win_fence(MPI_MODE_NOPRECEDE, colwindows[j]);

	for(int j=0; j< rowwinnext.size(); ++j)
		MPI_Win_fence(MPI_MODE_NOPRECEDE, rowwinnext[j]);
	for(int j=0; j< colwinnext.size(); ++j)
		MPI_Win_fence(MPI_MODE_NOPRECEDE, colwinnext[j]);


	int Aownind = (0+Aoffset) % stages;		
	int Bownind = (0+Boffset) % stages;
	if(Aownind == (A.commGrid)->GetRankInProcRow())
	{	
		ARecv = A.spSeq;	// shallow-copy 
	}
	else
	{
		vector<IU> ess1(UDERA::esscount);		// pack essentials to a vector
		for(int j=0; j< UDERA::esscount; ++j)	
		{
			ess1[j] = ARecvSizes[j][Aownind];	
		}
		ARecv = new UDERA();	// create the object first	

		oput << "For A (out), Fetching " << (void*)rowwindows[0] << endl;
		SpParHelper::FetchMatrix(*ARecv, ess1, rowwindows, Aownind);	// fetch its elements later
	}
	if(Bownind == (B.commGrid)->GetRankInProcCol())
	{
		BRecv = B.spSeq;	// shallow-copy
	}
	else
	{
		vector<IU> ess2(UDERB::esscount);		// pack essentials to a vector
		for(int j=0; j< UDERB::esscount; ++j)	
		{
			ess2[j] = BRecvSizes[j][Bownind];	
		}	
		BRecv = new UDERB();

		oput << "For B (out), Fetching " << (void*)colwindows[0] << endl;
		SpParHelper::FetchMatrix(*BRecv, ess2, colwindows, Bownind);	// No lock version, only get !
	}

	int Aownprev = Aownind;
	int Bownprev = Bownind;
	
	for(int i = 1; i < stages; ++i) 
	{
		Aownind = (i+Aoffset) % stages;		
		Bownind = (i+Boffset) % stages;

		if(i % 2 == 1)	// Fetch RecvNext via winnext, fence on Recv via windows
		{	
			if(Aownind == (A.commGrid)->GetRankInProcRow())
			{	
				ARecvNext = A.spSeq;	// shallow-copy 
			}
			else
			{
				vector<IU> ess1(UDERA::esscount);		// pack essentials to a vector
				for(int j=0; j< UDERA::esscount; ++j)	
				{
					ess1[j] = ARecvSizes[j][Aownind];	
				}
				ARecvNext = new UDERA();	// create the object first	

				oput << "For A, Fetching " << (void*) rowwinnext[0] << endl;
				SpParHelper::FetchMatrix(*ARecvNext, ess1, rowwinnext, Aownind);
			}
		
			if(Bownind == (B.commGrid)->GetRankInProcCol())
			{
				BRecvNext = B.spSeq;	// shallow-copy
			}
			else
			{
				vector<IU> ess2(UDERB::esscount);		// pack essentials to a vector
				for(int j=0; j< UDERB::esscount; ++j)	
				{
					ess2[j] = BRecvSizes[j][Bownind];	
				}		
				BRecvNext = new UDERB();

				oput << "For B, Fetching " << (void*)colwinnext[0] << endl;
				SpParHelper::FetchMatrix(*BRecvNext, ess2, colwinnext, Bownind);	// No lock version, only get !
			}
		
			oput << "Fencing " << (void*) rowwindows[0] << endl;
			oput << "Fencing " << (void*) colwindows[0] << endl;
		
			for(int j=0; j< rowwindows.size(); ++j)
				MPI_Win_fence(MPI_MODE_NOSTORE, rowwindows[j]);		// Synch using "other" windows
			for(int j=0; j< colwindows.size(); ++j)
				MPI_Win_fence(MPI_MODE_NOSTORE, colwindows[j]);
	
			SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>(*ARecv, *BRecv, false, true);
			if(!C_cont->isZero()) 
				tomerge.push_back(C_cont);

			if(Aownprev != (A.commGrid)->GetRankInProcRow()) delete ARecv;
			if(Bownprev != (B.commGrid)->GetRankInProcCol()) delete BRecv;

			Aownprev = Aownind;
			Bownprev = Bownind; 
		}	
		else	// fetch to Recv via windows, fence on RecvNext via winnext
		{	
			
			if(Aownind == (A.commGrid)->GetRankInProcRow())
			{	
				ARecv = A.spSeq;	// shallow-copy 
			}
			else
			{
				vector<IU> ess1(UDERA::esscount);		// pack essentials to a vector
				for(int j=0; j< UDERA::esscount; ++j)	
				{
					ess1[j] = ARecvSizes[j][Aownind];	
				}
				ARecv = new UDERA();	// create the object first	

				oput << "For A, Fetching " << (void*) rowwindows[0] << endl;
				SpParHelper::FetchMatrix(*ARecv, ess1, rowwindows, Aownind);
			}
		
			if(Bownind == (B.commGrid)->GetRankInProcCol())
			{
				BRecv = B.spSeq;	// shallow-copy
			}
			else
			{
				vector<IU> ess2(UDERB::esscount);		// pack essentials to a vector
				for(int j=0; j< UDERB::esscount; ++j)	
				{
					ess2[j] = BRecvSizes[j][Bownind];	
				}		
				BRecv = new UDERB();

				oput << "For B, Fetching " << (void*)colwindows[0] << endl;
				SpParHelper::FetchMatrix(*BRecv, ess2, colwindows, Bownind);	// No lock version, only get !
			}
		
			oput << "Fencing " << (void*) rowwinnext[0] << endl;
			oput << "Fencing " << (void*) rowwinnext[0] << endl;
		
			for(int j=0; j< rowwinnext.size(); ++j)
				MPI_Win_fence(MPI_MODE_NOSTORE, rowwinnext[j]);		// Synch using "other" windows
			for(int j=0; j< colwinnext.size(); ++j)
				MPI_Win_fence(MPI_MODE_NOSTORE, colwinnext[j]);

			SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>(*ARecvNext, *BRecvNext, false, true);
			if(!C_cont->isZero()) 
				tomerge.push_back(C_cont);


			if(Aownprev != (A.commGrid)->GetRankInProcRow()) delete ARecvNext;
			if(Bownprev != (B.commGrid)->GetRankInProcCol()) delete BRecvNext;

			Aownprev = Aownind;
			Bownprev = Bownind; 
		}

	}

	if(stages % 2 == 1)	// fence on Recv via windows
	{
		oput << "Fencing " << (void*) rowwindows[0] << endl;
		oput << "Fencing " << (void*) colwindows[0] << endl;

		for(int j=0; j< rowwindows.size(); ++j)
			MPI_Win_fence(MPI_MODE_NOSUCCEED, rowwindows[j]);		// Synch using "prev" windows
		for(int j=0; j< colwindows.size(); ++j)
			MPI_Win_fence(MPI_MODE_NOSUCCEED, colwindows[j]);

		SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>(*ARecv, *BRecv, false, true);
		if(!C_cont->isZero()) 
			tomerge.push_back(C_cont);

		if(Aownprev != (A.commGrid)->GetRankInProcRow()) delete ARecv;
		if(Bownprev != (B.commGrid)->GetRankInProcRow()) delete BRecv;
	}
	else		// fence on RecvNext via winnext
	{
		oput << "Fencing " << (void*) rowwinnext[0] << endl;
		oput << "Fencing " << (void*) colwinnext[0] << endl;

		for(int j=0; j< rowwinnext.size(); ++j)
			MPI_Win_fence(MPI_MODE_NOSUCCEED, rowwinnext[j]);		// Synch using "prev" windows
		for(int j=0; j< colwinnext.size(); ++j)
			MPI_Win_fence(MPI_MODE_NOSUCCEED, colwinnext[j]);

		SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>(*ARecvNext, *BRecvNext, false, true);
		if(!C_cont->isZero()) 
			tomerge.push_back(C_cont);

		if(Aownprev != (A.commGrid)->GetRankInProcRow()) delete ARecvNext;
		if(Bownprev != (B.commGrid)->GetRankInProcRow()) delete BRecvNext;
	}
	for(int i=0; i< rowwindows.size(); ++i)
	{
		rowwindows[i].Free();
		rowwinnext[i].Free();
	}
	for(int i=0; i< colwindows.size(); ++i)
	{
		colwindows[i].Free();
		colwinnext[i].Free();
	}
	GridC->GetWorld().Barrier();

	
	IU C_m = A.spSeq->getnrow();
	IU C_n = B.spSeq->getncol();

	DER_promote * C = new DER_promote(MergeAll<SR>(tomerge, C_m, C_n), false, NULL);	// First get the result in SpTuples, then convert to UDER
	for(int i=0; i<tomerge.size(); ++i)
	{
		delete tomerge[i];
	}
	
	SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
	SpHelper::deallocate2D(BRecvSizes, UDERB::esscount);

	
	const_cast< UDERB* >(B.spSeq)->Transpose();	// transpose back to original
	
	return SpParMat<IU,N_promote,DER_promote> (C, GridC);			// return the result object
}



template <typename IU>
void RandPerm(SpParVec<IU,IU> & V, IU loclength)
{
	V.length = loclength;
	MPI::Intracomm DiagWorld = V.commGrid->GetDiagWorld();

	if(DiagWorld != MPI::COMM_NULL) // Diagonal processors only
	{
		pair<double,IU> * vecpair = new pair<double,IU>[loclength];

		int nproc = DiagWorld.Get_size();
		int diagrank = DiagWorld.Get_rank();

		long * dist = new long[nproc];
		dist[diagrank] = V.length;
		DiagWorld.Allgather(MPI::IN_PLACE, 0, MPI::DATATYPE_NULL, dist, 1, MPIType<long>());
		IU lengthuntil = accumulate(dist, dist+diagrank, 0);

  		MTRand M;	// generate random numbers with Mersenne Twister
		for(int i=0; i<loclength; ++i)
		{
			vecpair[i].first = M.rand();
			vecpair[i].second = i+1+lengthuntil;	// permutation indices are 1-based
		}

		// less< pair<T1,T2> > works correctly (sorts wrt first elements)	
    		psort::parallel_sort (vecpair, vecpair + loclength,  dist, DiagWorld);

		vector< IU > nind(loclength);
		vector< IU > nnum(loclength);
		for(int i=0; i<loclength; ++i)
		{
			nind[i] = i;
			nnum[i] = vecpair[i].second;
		}
		delete [] vecpair;
		delete [] dist;

		V.ind.swap(nind);
		V.num.swap(nnum);
	}
}
		
template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
DenseParVec<IU,typename promote_trait<NUM,NUV>::T_promote>  SpMV 
	(const SpParMat<IU,NUM,UDER> & A, const DenseParVec<IU,NUV> & x )
{
	typedef typename promote_trait<NUM,NUV>::T_promote T_promote;
	if(!(*A.commGrid == *x.commGrid)) 		
	{
		cout << "Grids are not comparable for SpMV" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
	}
	MPI::Intracomm DiagWorld = x.commGrid->GetDiagWorld();
	MPI::Intracomm ColWorld = x.commGrid->GetColWorld();
	MPI::Intracomm RowWorld = x.commGrid->GetRowWorld();
	int diaginrow = x.commGrid->GetDiagOfProcRow();
        int diagincol = x.commGrid->GetDiagOfProcCol();

	T_promote id = (T_promote) 0;	// do we need a better identity?
	DenseParVec<IU, T_promote> y ( x.commGrid, id);	
	IU ysize = A.getlocalrows();
	if(x.diagonal)
	{
		IU size = x.arr.size();
		ColWorld.Bcast(&size, 1, MPIType<IU>(), diagincol);
		ColWorld.Bcast(const_cast<NUV*>(&x.arr[0]), size, MPIType<NUV>(), diagincol); 

		T_promote * localy = new T_promote[ysize];
		fill_n(localy, ysize, id);		
		dcsc_gespmv<SR>(*(A.spSeq), &x.arr[0], localy);	

		// IntraComm::Reduce(sendbuf, recvbuf, count, type, op, root)
                RowWorld.Reduce(MPI::IN_PLACE, localy, ysize, MPIType<T_promote>(), SR::mpi_op(), diaginrow);
		y.arr.resize(ysize);
		copy(localy, localy+ysize, y.arr.begin());
		delete [] localy;
	}
	else
	{
		IU size;
		ColWorld.Bcast(&size, 1, MPIType<IU>(), diagincol);

		NUV * localx = new NUV[size];
		ColWorld.Bcast(localx, size, MPIType<NUV>(), diagincol); 
	
		T_promote * localy = new T_promote[ysize];		
		fill_n(localy, ysize, id);		

		dcsc_gespmv<SR>(*(A.spSeq), localx, localy);
		delete [] localx;

                RowWorld.Reduce(localy, NULL, ysize, MPIType<T_promote>(), SR::mpi_op(), diaginrow);	
		delete [] localy;
	}
	return y;
}
	

template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
SpParVec<IU,typename promote_trait<NUM,NUV>::T_promote>  SpMV 
	(const SpParMat<IU,NUM,UDER> & A, const SpParVec<IU,NUV> & x )
{
	typedef typename promote_trait<NUM,NUV>::T_promote T_promote;
	if(!(*A.commGrid == *x.commGrid)) 		
	{
		cout << "Grids are not comparable for SpMV" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
	}
	MPI::Intracomm DiagWorld = x.commGrid->GetDiagWorld();
	MPI::Intracomm ColWorld = x.commGrid->GetColWorld();
	MPI::Intracomm RowWorld = x.commGrid->GetRowWorld();
	int diaginrow = x.commGrid->GetDiagOfProcRow();
        int diagincol = x.commGrid->GetDiagOfProcCol();

	SpParVec<IU, T_promote> y ( x.commGrid);	// identity doesn't matter for sparse vectors
	IU ysize = A.getlocalrows();
	if(x.diagonal)
	{
		IU nnzx = x.getlocnnz();
		ColWorld.Bcast(&nnzx, 1, MPIType<IU>(), diagincol);
		ColWorld.Bcast(const_cast<IU*>(&x.ind[0]), nnzx, MPIType<IU>(), diagincol); 
		ColWorld.Bcast(const_cast<NUV*>(&x.num[0]), nnzx, MPIType<NUV>(), diagincol); 

		// define a SPA-like data structure
		T_promote * localy = new T_promote[ysize];
		bool * isthere = new bool[ysize];
		vector<IU> nzinds;	// nonzero indices		
		fill_n(isthere, ysize, false);

		// serial SpMV with sparse vector
		vector< IU > indy;
		vector< T_promote >  numy;
		dcsc_gespmv<SR>(*(A.spSeq), &x.ind[0], &x.num[0], nnzx, indy, numy);	

		int proccols = x.commGrid->GetGridCols();
		int * gsizes = new int[proccols];	// # of processor columns = number of processors in the RowWorld
		int mysize = indy.size();
		RowWorld.Gather(&mysize, 1, MPI::INT, gsizes, 1, MPI::INT, diaginrow);
		int maxnnz = std::accumulate(gsizes, gsizes+proccols, 0);
		int * dpls = new int[proccols]();	// displacements (zero initialized pid) 
		std::partial_sum(gsizes, gsizes+proccols-1, dpls+1);
		
		IU * indbuf = new IU[maxnnz];	
		T_promote * numbuf = new T_promote[maxnnz];

		// IntraComm::GatherV(sendbuf, int sentcnt, sendtype, recvbuf, int * recvcnts, int * displs, recvtype, root)
                RowWorld.Gatherv(&(indy[0]), mysize, MPIType<IU>(), indbuf, gsizes, dpls, MPIType<IU>(), diaginrow);
                RowWorld.Gatherv(&(numy[0]), mysize, MPIType<T_promote>(), numbuf, gsizes, dpls, MPIType<T_promote>(), diaginrow);

		for(int i=0; i< maxnnz; ++i)
		{
			if(!isthere[indbuf[i]])
			{
				localy[indbuf[i]] = numbuf[i];	// initial assignment
				nzinds.push_back(indbuf[i]);
				isthere[indbuf[i]] = true;
			} 
			else
			{
				localy[indbuf[i]] = SR::add(localy[indbuf[i]], numbuf[i]);	
			}
		}
		DeleteAll(gsizes, dpls, indbuf, numbuf,isthere);
		sort(nzinds.begin(), nzinds.end());
		
		int nnzy = nzinds.size();
		y.ind.resize(nnzy);
		y.num.resize(nnzy);
		for(int i=0; i< nnzy; ++i)
		{
			y.ind[i] = nzinds[i];
			y.num[i] = localy[nzinds[i]]; 	
		}
		y.length = ysize;
		delete [] localy;
	}
	else
	{
		IU nnzx;
		ColWorld.Bcast(&nnzx, 1, MPIType<IU>(), diagincol);

		IU * xinds = new IU[nnzx];
		NUV * xnums = new NUV[nnzx];
		ColWorld.Bcast(xinds, nnzx, MPIType<IU>(), diagincol); 
		ColWorld.Bcast(xnums, nnzx, MPIType<NUV>(), diagincol); 

		// serial SpMV with sparse vector
		vector< IU > indy;
		vector< T_promote >  numy;
		dcsc_gespmv<SR>(*(A.spSeq), xinds, xnums, nnzx, indy, numy);	

		int mysize = indy.size();
		RowWorld.Gather(&mysize, 1, MPI::INT, NULL, 1, MPI::INT, diaginrow);

		// IntraComm::GatherV(sendbuf, int sentcnt, sendtype, recvbuf, int * recvcnts, int * displs, recvtype, root)
                RowWorld.Gatherv(&(indy[0]), mysize, MPIType<IU>(), NULL, NULL, NULL, MPIType<IU>(), diaginrow);
                RowWorld.Gatherv(&(numy[0]), mysize, MPIType<T_promote>(), NULL, NULL, NULL, MPIType<T_promote>(), diaginrow);

		delete [] xinds;
		delete [] xnums;
	}
	return y;
}
	

template <typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB> 
SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDERA,UDERB>::T_promote> EWiseMult 
	(const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B , bool exclude)
{
	typedef typename promote_trait<NU1,NU2>::T_promote N_promote;
	typedef typename promote_trait<UDERA,UDERB>::T_promote DER_promote;

	if(*(A.commGrid) == *(B.commGrid))	
	{
		DER_promote * result = new DER_promote( EWiseMult(*(A.spSeq),*(B.spSeq),exclude) );
		return SpParMat<IU, N_promote, DER_promote> (result, A.commGrid);
	}
	else
	{
		cout << "Grids are not comparable elementwise multiplication" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
		return SpParMat< IU,N_promote,DER_promote >();
	}
}


/**
 * if exclude is true, then we prune all entries W[i] != zero from V 
 * if exclude is false, then we perform a proper elementwise multiplication
**/
template <typename IU, typename NU1, typename NU2>
SpParVec<IU,typename promote_trait<NU1,NU2>::T_promote> EWiseMult 
	(const SpParVec<IU,NU1> & V, const DenseParVec<IU,NU2> & W , bool exclude, NU2 zero)
{
	typedef typename promote_trait<NU1,NU2>::T_promote T_promote;

	if(*(V.commGrid) == *(W.commGrid))	
	{
		SpParVec< IU, T_promote> Product(V.commGrid);
		Product.length = V.length;
		if(Product.diagonal)
		{
			if(exclude)
			{
				IU size= V.ind.size();
				for(IU i=0; i<size; ++i)
				{
					if(W.arr[V.ind[i]] == zero) 	// keep only those
					{
						Product.ind.push_back(V.ind[i]);
						Product.num.push_back(V.num[i]);
					}
				}		
			}	
			else
			{
				IU size= V.ind.size();
				for(IU i=0; i<size; ++i)
				{
					if(W.arr[V.ind[i]] != zero) 	// keep only those
					{
						Product.ind.push_back(V.ind[i]);
						Product.num.push_back(V.num[i] * W.arr[V.ind[i]]);
					}
				}
			}
		}
		return Product;
	}
	else
	{
		cout << "Grids are not comparable elementwise multiplication" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
		return SpParVec< IU,T_promote>();
	}
}

#endif

