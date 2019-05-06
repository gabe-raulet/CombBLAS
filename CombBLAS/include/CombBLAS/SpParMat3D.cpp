/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */



#include "SpParMat3D.h"
#include "ParFriends.h"
#include "Operations.h"
#include "FileHeader.h"
extern "C" {
#include "mmio.h"
}
#include <sys/types.h>
#include <sys/stat.h>

#include <mpi.h>
#include <fstream>
#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>

namespace combblas
{
    template <class IT, class NT, class DER>
    SpParMat3D< IT,NT,DER >::SpParMat3D (const SpParMat< IT,NT,DER > & A2D, int nlayers, bool csplit, bool special): nlayers(nlayers), colsplit(csplit)
    {
        //colsplit = csplit;
        typedef typename DER::LocalIT LIT;
        auto commGrid2D = A2D.getcommgrid();
        int nprocs = commGrid2D->GetSize();
        if(colsplit) commGrid3D.reset(new CommGrid3D(commGrid2D->GetWorld(), nlayers, 0, 0, true, true));
        else commGrid3D.reset(new CommGrid3D(commGrid2D->GetWorld(), nlayers, 0, 0, false, true));

        DER* spSeq = A2D.seqptr(); // local submatrix
        //printf("myrank %d\trankInLayer %d\tnrow %d\tncol %d\tnnz %d\n", commGrid3D->myrank, commGrid3D->rankInLayer, spSeq->getnrow(), spSeq->getncol(), spSeq->getnnz());
        std::vector<DER> localChunks;
        int numChunks = (int)std::sqrt((float)nlayers);
        if(!colsplit) spSeq->Transpose();
        spSeq->ColSplit(numChunks, localChunks);
        if(!colsplit){
            for(int i = 0; i < localChunks.size(); i++) localChunks[i].Transpose();
        }

        // Some necessary processing before exchanging data
        int sqrtLayer = (int)std::sqrt((float)nlayers);
        std::vector<DER> sendChunks(nlayers);
        for(int i = 0; i < sendChunks.size(); i++){
            sendChunks[i] = DER(0, 0, 0, 0);
        }
        if(colsplit){
            for(int i = 0; i < localChunks.size(); i++){
                int rcvRankInFiber = ( ( commGrid3D->rankInFiber / sqrtLayer ) * sqrtLayer ) + i;
                sendChunks[rcvRankInFiber] = localChunks[i];
            }
        }
        else{
            for(int i = 0; i < localChunks.size(); i++){
                int rcvRankInFiber = ( ( commGrid3D->rankInFiber % sqrtLayer ) * sqrtLayer ) + i;
                sendChunks[rcvRankInFiber] = localChunks[i];
            }
        }
        MPI_Barrier(commGrid3D->GetWorld());

        IT datasize; NT x;
        std::vector<DER> recvChunks;

        //string prefix("3D-stdout-"); 
        //string proc = to_string(commGrid3D->myrank); 
        //string filename = prefix + proc;
        //FILE * fp;
        //fp = fopen(filename.c_str(), "a");
        //fprintf(fp, "[SENDING] --> \n");
        //for(int i = 0; i < sendChunks.size(); i++){
            //fprintf(fp, "myrank:%2d\tsendChunks[%d]\trows:%10d\tcols:%10d\tnnz:%10d\n", commGrid3D->myrank, i, sendChunks[i].getnrow(), sendChunks[i].getncol(), sendChunks[i].getnnz());
        //}
        SpecialExchangeData(sendChunks, commGrid3D->fiberWorld, datasize, x, commGrid3D->world3D, recvChunks);
        //fprintf(fp, "[RECEIVING] <-- \n");
        //for(int i = 0; i < recvChunks.size(); i++){
            //fprintf(fp, "myrank:%2d\trecvChunks[%d]\trows:%10d\tcols:%10d\tnnz:%10d\n", commGrid3D->myrank, i, recvChunks[i].getnrow(), recvChunks[i].getncol(), recvChunks[i].getnnz());
        //}
        IT concat_row = 0, concat_col = 0;
        for(int i  = 0; i < recvChunks.size(); i++){
            if(colsplit) recvChunks[i].Transpose();
            concat_row = std::max(concat_row, recvChunks[i].getnrow());
            concat_col = concat_col + recvChunks[i].getncol();
        }
        DER * localMatrix = new DER(0, concat_row, concat_col, 0);
        localMatrix->ColConcatenate(recvChunks);
        if(colsplit) localMatrix->Transpose();
        //fprintf(fp, "myrank:%2d\trankInFiber:%2d\trankInLayer:%2d\trankInSpecialWorld:%2d\tnrows:%2d\tncols:%2d\tnnz:%2d\n", 
                //commGrid3D->myrank, commGrid3D->rankInFiber, commGrid3D->rankInLayer, commGrid3D->rankInSpecialWorld, localMatrix->getnrow(), localMatrix->getncol(), localMatrix->getnnz());
        //fclose(fp);
        layermat = new SpParMat<IT, NT, DER>(localMatrix, commGrid3D->layerWorld);
    }
    
    template <class IT, class NT, class DER>
    SpParMat<IT, NT, DER> SpParMat3D<IT, NT, DER>::Convert2D(){
        DER * spSeq = layermat->seqptr();
        std::vector<DER> localChunks;
        int numChunks = (int)std::sqrt((float)nlayers);
        if(colsplit) spSeq->Transpose();
        spSeq->ColSplit(numChunks, localChunks);
        if(colsplit){
            for(int i = 0; i < localChunks.size(); i++) localChunks[i].Transpose();
        }
        std::vector<DER> sendChunks(nlayers);
        int sqrtLayer = (int)std::sqrt((float)nlayers);
        for(int i = 0; i < sendChunks.size(); i++){
            sendChunks[i] = DER(0, 0, 0, 0);
        }
        if(colsplit){
            for(int i = 0; i < localChunks.size(); i++){
                int rcvRankInFiber = ( ( commGrid3D->rankInFiber / sqrtLayer ) * sqrtLayer ) + i;
                sendChunks[rcvRankInFiber] = localChunks[i];
            }
        }
        IT datasize; NT x;
        std::vector<DER> recvChunks;

        SpecialExchangeData(sendChunks, commGrid3D->fiberWorld, datasize, x, commGrid3D->world3D, recvChunks);

        IT concat_row = 0, concat_col = 0;
        for(int i  = 0; i < recvChunks.size(); i++){
            if(!colsplit) recvChunks[i].Transpose();
            concat_row = std::max(concat_row, recvChunks[i].getnrow());
            concat_col = concat_col + recvChunks[i].getncol();
        }
        DER * localMatrix = new DER(0, concat_row, concat_col, 0);
        localMatrix->ColConcatenate(recvChunks);
        if(!colsplit) localMatrix->Transpose();
        //printf("myrank %d\trankInLayer %d\tnrow %d\tncol %d\tnnz %d\n", commGrid3D->myrank, commGrid3D->rankInLayer, localMatrix->getnrow(), localMatrix->getncol(), localMatrix->getnnz());
        std::shared_ptr<CommGrid> grid2d;
        grid2d.reset(new CommGrid(commGrid3D->GetWorld(), 0, 0));
        SpParMat<IT, NT, DER> mat2D(localMatrix, grid2d);
        return mat2D;
    }
    
    template <class IT, class NT, class DER>
    template <typename SR>
    SpParMat<IT, NT, DER> SpParMat3D< IT,NT,DER >::mult(SpParMat3D<IT, NT, DER> & M){
        SpParMat<IT, NT, DER>* Mlayermat = M.layermat;
        //CheckSpGEMMCompliance(*layermat, *Mlayermat);
        //printf("myrank %d\tA.rankInFiber %d\tA.rankInLayer %d\tB.rankInFiber %d\tB.rankInLayer %d\t:\t[%d x %d] X [%d x %d]\n", 
                //commGrid3D->myrank, commGrid3D->rankInFiber, commGrid3D->rankInLayer, M.commGrid3D->rankInFiber, M.commGrid3D->rankInLayer,
                //layermat->getnrow(), layermat->getncol(), Mlayermat->getnrow(), Mlayermat->getncol());
        typedef PlusTimesSRing<NT, NT> PTFF;
        //Mult_AnXBn_Synch<PTFF, NT, DER>(*layermat, *Mlayermat);
        SpParMat<IT, NT, DER> C3D_layer = Mult_AnXBn_DoubleBuff<PTFF, NT, DER>(*layermat, *Mlayermat);
        int nlayers;
        MPI_Comm_size(commGrid3D->fiberWorld, &nlayers);
        int sqrtLayers = (int)std::sqrt((float)nlayers);
        DER* C3D_localMat = C3D_layer.seqptr();
        //printf("myrank %d\trankInLayer %d\tnrow %d\tncol %d\tnnz %d\n", commGrid3D->myrank, commGrid3D->rankInLayer, C3D_localMat->getnrow(), C3D_localMat->getncol(), C3D_localMat->getnnz());
        vector<DER> chunkArr1;
        vector<DER> sendChunks(nlayers);
        C3D_localMat->ColSplit(sqrtLayers, chunkArr1);
        for(int i = 0; i < chunkArr1.size(); i++){
            vector<DER> chunkArr2;
            chunkArr1[i].Transpose();
            chunkArr1[i].ColSplit(sqrtLayers, chunkArr2);
            for(int j = 0; j < chunkArr2.size(); j++){
                chunkArr2[j].Transpose();
                sendChunks[i*sqrtLayers+j] = chunkArr2[j];
            }
        }
        vector<DER> rcvChunks;
        IT datasize;
        NT dummy;
        SpecialExchangeData( sendChunks, commGrid3D->fiberWorld, datasize, dummy, commGrid3D->fiberWorld, rcvChunks);
        DER * chunk = new DER(0, rcvChunks[0].getnrow(), rcvChunks[0].getncol(), 0);
        for(int i = 1; i < rcvChunks.size(); i++) *chunk += rcvChunks[i];
        std::shared_ptr<CommGrid> grid2d;
        grid2d.reset(new CommGrid(commGrid3D->GetWorld(), 0, 0));
        SpParMat<IT, NT, DER> C2D(chunk, grid2d);
        return C2D;
    }
    
    template <class IT, class NT, class DER>
    IT SpParMat3D< IT,NT,DER >::getnrow() const
    {
        IT totalrows_layer = layermat->getnrow();
        IT totalrows = 0;
        if(!colsplit)
        {
            MPI_Allreduce( &totalrows_layer, &totalrows, 1, MPIType<IT>(), MPI_SUM, commGrid3D->fiberWorld);
        }
        else
            totalrows = totalrows_layer;
        return totalrows;
    }
    
    
    template <class IT, class NT, class DER>
    IT SpParMat3D< IT,NT,DER >::getncol() const
    {
        IT totalcols_layer = layermat->getncol();
        IT totalcols = 0;
        if(!colsplit)
        {
            MPI_Allreduce( &totalcols_layer, &totalcols, 1, MPIType<IT>(), MPI_SUM, commGrid3D->fiberWorld);
        }
        else
            totalcols = totalcols_layer;
        return totalcols;
    }


    template <class IT, class NT, class DER>
    IT SpParMat3D< IT,NT,DER >::getnnz() const
    {
        IT totalnz_layer = layermat->getnnz();
        IT totalnz = 0;
        if(!colsplit)
        {
            MPI_Allreduce( &totalnz_layer, &totalnz, 1, MPIType<IT>(), MPI_SUM, commGrid3D->fiberWorld);
        }
        else
            totalnz = totalnz_layer;
        return totalnz;
    }


    template <class IT, class NT, class DER>
    vector<DER> SpecialExchangeData( std::vector<DER> & sendChunks, MPI_Comm World, IT& datasize, NT dummy, MPI_Comm secondaryWorld, vector<DER> & recvChunks){
        //int myrank;
        //MPI_Comm_rank(secondaryWorld, &myrank);
        int numChunks = sendChunks.size();

        MPI_Datatype MPI_tuple;
        MPI_Type_contiguous(sizeof(std::tuple<IT,IT,NT>), MPI_CHAR, &MPI_tuple);
        MPI_Type_commit(&MPI_tuple);

        int * sendcnt = new int[numChunks];
        int * sendprfl = new int[numChunks*3];
        int * sdispls = new int[numChunks]();
        int * recvcnt = new int[numChunks];
        int * recvprfl = new int[numChunks*3];
        int * rdispls = new int[numChunks]();

        IT totsend = 0;
        for(IT i=0; i<numChunks; ++i){
            sendprfl[i*3] = sendChunks[i].getnnz();
            sendprfl[i*3+1] = sendChunks[i].getnrow();
            sendprfl[i*3+2] = sendChunks[i].getncol();
            sendcnt[i] = sendprfl[i*3];
            totsend += sendcnt[i];
        }

        MPI_Alltoall(sendprfl, 3, MPI_INT, recvprfl, 3, MPI_INT, World);

        for(IT i = 0; i < numChunks; i++) recvcnt[i] = recvprfl[i*3];

        std::partial_sum(sendcnt, sendcnt+numChunks-1, sdispls+1);
        std::partial_sum(recvcnt, recvcnt+numChunks-1, rdispls+1);
        IT totrecv = std::accumulate(recvcnt,recvcnt+numChunks, static_cast<IT>(0));

        std::vector< std::tuple<IT,IT,NT> > sendTuples;
        for (int i = 0; i < numChunks; i++){
            for(typename DER::SpColIter colit = sendChunks[i].begcol(); colit != sendChunks[i].endcol(); ++colit){
                for(typename DER::SpColIter::NzIter nzit = sendChunks[i].begnz(colit); nzit != sendChunks[i].endnz(colit); ++nzit){
                    NT val = nzit.value();
                    sendTuples.push_back(std::make_tuple(nzit.rowid(), colit.colid(), nzit.value()));
                }
            }
        }
        //if(myrank == 12){
            //int i = 1;
            //printf("[SENDING] Processor: %d, Chunk: %d\n", myrank, i);
            //printf("nnz: %d, mdim: %d, ndim: %d\n",sendprfl[i*3], sendprfl[i*3+1], sendprfl[i*3+2]);
            //printf("first tuple: < %lld, %lld, %lf >\n", 
                    //get<0>(sendTuples[sdispls[i]]), 
                    //get<1>(sendTuples[sdispls[i]]), 
                    //get<2>(sendTuples[sdispls[i]]));
            //printf("last tuple: < %lld, %lld, %lf >\n", 
                    //get<0>(sendTuples[sdispls[i]+sendcnt[i]-1]), 
                    //get<1>(sendTuples[sdispls[i]+sendcnt[i]-1]), 
                    //get<2>(sendTuples[sdispls[i]+sendcnt[i]-1]));
            ////for(int j = sdispls[i]; j < sdispls[i]+sendcnt[i]; j++){
                ////cout << get<0>(sendTuples[j]) << " " << get<1>(sendTuples[j]) << " " << get<2>(sendTuples[j]) << endl;
            ////}
        //}
        std::tuple<IT,IT,NT>* recvTuples = new std::tuple<IT,IT,NT>[totrecv];
        MPI_Alltoallv(sendTuples.data(), sendcnt, sdispls, MPI_tuple, recvTuples, recvcnt, rdispls, MPI_tuple, World);

        DeleteAll(sendcnt, sendprfl, sdispls);
        sendTuples.clear();
        sendTuples.shrink_to_fit();

        //std::vector< std::tuple<IT,IT,NT> > recvTuples(totrecv);
        //MPI_Alltoallv(sendTuples.data(), sendcnt, sdispls, MPI_tuple, recvTuples.data(), recvcnt, rdispls, MPI_tuple, World);
        tuple<IT, IT, NT> ** tempTuples = new tuple<IT, IT, NT>*[numChunks];
        //vector<DER> recvChunks;
        for (int i = 0; i < numChunks; i++){
            tempTuples[i] = new tuple<IT, IT, NT>[recvcnt[i]];
            memcpy(tempTuples[i], recvTuples+rdispls[i], recvcnt[i]*sizeof(tuple<IT, IT, NT>));
            recvChunks.push_back(DER(SpTuples<IT, NT>(recvcnt[i], recvprfl[i*3+1], recvprfl[i*3+2], tempTuples[i]), false));
        }

        //if(myrank == 6){
            //int i = 2;
            //printf("[RECEIVING] Processor: %d, Chunk: %d\n", myrank, i);
            //printf("nnz: %d, mdim: %d, ndim: %d\n",recvprfl[i*3], recvprfl[i*3+1], recvprfl[i*3+2]);
            //printf("first tuple: < %lld, %lld, %lf >\n", 
                    //get<0>(recvTuples[rdispls[i]]), 
                    //get<1>(recvTuples[rdispls[i]]), 
                    //get<2>(recvTuples[rdispls[i]]));
            //printf("last tuple: < %lld, %lld, %lf >\n", 
                    //get<0>(recvTuples[rdispls[i]+recvcnt[i]-1]), 
                    //get<1>(recvTuples[rdispls[i]+recvcnt[i]-1]), 
                    //get<2>(recvTuples[rdispls[i]+recvcnt[i]-1]));
            ////for(int j = rdispls[i]; j < rdispls[i]+recvcnt[i]; j++){
                ////cout << get<0>(recvTuples[j]) << " " << get<1>(recvTuples[j]) << " " << get<2>(recvTuples[j]) << endl;
            ////}
        //}
        DeleteAll(recvcnt, recvprfl, rdispls, recvTuples); // Free all memory
        //for(int i = 0; i < numChunks; i++){
            //delete[] tempTuples[i];
        //}
        //delete[] tempTuples;
        MPI_Type_free(&MPI_tuple);
        return recvChunks;
    }
}
