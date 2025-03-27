/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

namespace {
  template<typename T, typename RedOp, typename Proto, bool isNetOffload = false>
  __device__ __forceinline__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclRing *ring = &ncclShmem.channel.ring;
    const int *ringRanks = ring->userRanks;
    const int nranks = ncclShmem.comm.nRanks;
    ssize_t count, partOffset, partCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count, &partOffset, &partCount, &chunkCount);
    ssize_t offset;
    ssize_t dataOffset;
    int nelem;
    int rankDest;
    int workNthreads;
    T *inputBuf = (T*)work->sendbuff;
    T *outputBuf = (T*)work->recvbuff;

    // If isNetOffload == true, we only use 1 warp to drive Ring algo/network communication
    // and the rest of warps proceed to copy src data into dst buffer in parallel when AG
    // is not in-place.
    if (isNetOffload) {
      workNthreads = WARP_SIZE;
      chunkCount = NCCL_MAX_NET_SIZE;
    } else {
      workNthreads = nthreads;
    }

    if (tid < workNthreads) {
      // Coverity reports that the callee treats &ring->next as an array.  However, due to the use of
      // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
      // coverity[callee_ptr_arith:FALSE]
      Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0, isNetOffload> prims
        (tid, workNthreads, &ring->prev, &ring->next, inputBuf, outputBuf, work->redOpArg, 0, 0, 0, work, NULL, isNetOffload ? NCCL_MAX_NET_SIZE : 0);
      for (size_t elemOffset = 0; elemOffset < partCount; elemOffset += chunkCount) {
        /////////////// begin AllGather steps ///////////////
        nelem = min(chunkCount, partCount - elemOffset);
        dataOffset = partOffset + elemOffset;

        // step 0: push data to next GPU
        rankDest = ringRanks[0];
        offset = dataOffset + rankDest * count;

        if ((inputBuf + dataOffset == outputBuf + offset) || isNetOffload) { // In place or onePPN
          prims.directSend(dataOffset, offset, nelem);
        } else {
          prims.directCopySend(dataOffset, offset, nelem);
        }

        // k-2 steps: copy to next GPU
        for (int j = 1; j < nranks - 1; ++j) {
          rankDest = ringRanks[nranks - j];
          offset = dataOffset + rankDest * count;
          prims.directRecvCopyDirectSend(offset, offset, nelem);
        }

        // Make final copy from buffer to dest.
        rankDest = ringRanks[1];
        offset = dataOffset + rankDest * count;

        // Final wait/copy.
        prims.directRecv(offset, offset, nelem);
      }
    } else if (inputBuf != outputBuf + ringRanks[0] * count) {
      inputBuf = inputBuf + partOffset;
      outputBuf = outputBuf + partOffset + ringRanks[0] * count;
      reduceCopy<COLL_UNROLL, RedOp, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs=*/0>
        (tid - workNthreads, nthreads - workNthreads, work->redOpArg, &work->redOpArg, false, 1, (void**)&inputBuf, 1, (void**)&outputBuf, partCount);
    }
    // we have to wait for all warps before we can proceed to the next work;
    // otherwise, we can have contention if next work will use the outputBuf
    // in this work. We use bar 14 to avoid conflicts with prims barrier and
    // __syncthread().
    if (isNetOffload) barrier_sync(14, nthreads);
  }
}

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    bool isNetOffload = work->isOneRPN && work->netRegUsed;
    if (isNetOffload)
      runRing<T, RedOp, ProtoSimple<1, 1>, true>(tid, nthreads, work);
    else
      runRing<T, RedOp, ProtoSimple<ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS>, false>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<1, 1>;
    const int nranks = ncclShmem.comm.nRanks;
    const int rank = ncclShmem.comm.rank;
    size_t count, channelOffset, channelCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count, &channelOffset, &channelCount, &chunkCount);

    T *inputBuf = (T*)work->sendbuff;
    T *outputBuf = (T*)work->recvbuff;
    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0> prims
      (tid, nthreads, NULL, NULL, inputBuf, outputBuf, work->redOpArg, 0*Proto::MaxGroupWidth, 0, 0, nullptr, nullptr, 0, primsModePatAg);

    PatAGAlgorithm<T> patAlgo(chunkCount*sizeof(T), NCCL_STEPS, channelOffset, channelOffset + channelCount, count, chunkCount, rank, nranks);
    int last = 0;
    while (!last) {
      int recvDim, sendDim, recvOffset, sendOffset, recvStepOffset, postRecv, postSend, nelem;
      size_t inpIx, outIx;
      patAlgo.getNextOp(recvDim, sendDim, inpIx, outIx, recvOffset, sendOffset, recvStepOffset, nelem, postRecv, postSend, last);
      prims.patCopy(recvDim, sendDim, inpIx, outIx, recvOffset, sendOffset, recvStepOffset, nelem, postRecv, postSend);
    }
  }
};
