/******************************************************************************
 * 
 * Copyright 2010-2012 Duane Merrill
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 ******************************************************************************/

/******************************************************************************
 * CTA-processing functionality for radix sort upsweep reduction kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>

#include <b40c/radix_sort/sort_utils.cuh>

namespace b40c {
namespace radix_sort {
namespace upsweep {

/**
 * Radix sort upsweep reduction CTA
 */
template <
	typename KernelPolicy,
	typename SizeT,
	typename KeyType>
struct Cta
{
	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

	typedef typename KeyTraits<KeyType>::IngressOp 			IngressOp;
	typedef typename KeyTraits<KeyType>::ConvertedKeyType 	ConvertedKeyType;

	// Integer type for digit counters (to be packed into words of PackedCounters)
	typedef unsigned char DigitCounter;

	// Integer type for packing DigitCounters into columns of shared memory banks
	typedef typename util::If<
		(KernelPolicy::SMEM_8BYTE_BANKS),
		unsigned long long,
		unsigned int>::Type PackedCounter;

	enum {
		CURRENT_BIT 				= KernelPolicy::CURRENT_BIT,
		CURRENT_PASS 				= KernelPolicy::CURRENT_PASS,
		RADIX_BITS					= KernelPolicy::RADIX_BITS,
		RADIX_DIGITS 				= 1 << RADIX_BITS,

		// Direction of flow though ping-pong buffers: (FLOP_TURN) ? (d_keys1 --> d_keys0) : (d_keys0 --> d_keys1)
		FLOP_TURN					= KernelPolicy::CURRENT_PASS & 0x1,

		LOG_THREADS 				= KernelPolicy::LOG_THREADS,
		THREADS						= 1 << LOG_THREADS,

		LOG_WARP_THREADS 			= CUB_LOG_WARP_THREADS(__CUB_CUDA_ARCH__),
		WARP_THREADS				= 1 << LOG_WARP_THREADS,

		LOG_WARPS					= LOG_THREADS - LOG_WARP_THREADS,
		WARPS						= 1 << LOG_WARPS,

		LOG_LOAD_VEC_SIZE  			= KernelPolicy::LOG_LOAD_VEC_SIZE,
		LOAD_VEC_SIZE				= 1 << LOG_LOAD_VEC_SIZE,

		LOG_LOADS_PER_TILE 			= KernelPolicy::LOG_LOADS_PER_TILE,
		LOADS_PER_TILE				= 1 << LOG_LOADS_PER_TILE,

		LOG_THREAD_ELEMENTS			= LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE,
		THREAD_ELEMENTS				= 1 << LOG_THREAD_ELEMENTS,

		LOG_TILE_ELEMENTS 			= LOG_THREAD_ELEMENTS + LOG_THREADS,
		TILE_ELEMENTS				= 1 << LOG_TILE_ELEMENTS,

		BYTES_PER_COUNTER			= sizeof(DigitCounter),
		LOG_BYTES_PER_COUNTER		= util::Log2<BYTES_PER_COUNTER>::VALUE,

		PACKING_RATIO				= sizeof(PackedCounter) / sizeof(DigitCounter),
		LOG_PACKING_RATIO			= util::Log2<PACKING_RATIO>::VALUE,

		LOG_COUNTER_LANES 			= CUB_MAX(0, RADIX_BITS - LOG_PACKING_RATIO),
		COUNTER_LANES 				= 1 << LOG_COUNTER_LANES,

		// To prevent counter overflow, we must periodically unpack and aggregate the
		// digit counters back into registers.  Each counter lane is assigned to a
		// warp for aggregation.

		LOG_LANES_PER_WARP			= CUB_MAX(0, LOG_COUNTER_LANES - LOG_WARPS),
		LANES_PER_WARP 				= 1 << LOG_LANES_PER_WARP,

		// Unroll tiles in batches without risk of counter overflow
		UNROLL_COUNT				= 127 / THREAD_ELEMENTS,
		UNROLLED_ELEMENTS 			= UNROLL_COUNT * TILE_ELEMENTS,
	};



	/**
	 * Shared storage for radix distribution sorting upsweep
	 */
	struct SmemStorage
	{
		union {
			unsigned char	counter_base[1];
			DigitCounter 	digit_counters[COUNTER_LANES][THREADS][PACKING_RATIO];
			PackedCounter 	packed_counters[COUNTER_LANES][THREADS];
			SizeT 			digit_partials[RADIX_DIGITS][WARP_THREADS + 1];
		};
	};


	//---------------------------------------------------------------------
	// Thread fields
	//---------------------------------------------------------------------

	// Shared storage for this CTA
	SmemStorage 		&smem_storage;

	// Thread-local counters for periodically aggregating composite-counter lanes
	SizeT 				local_counts[LANES_PER_WARP][PACKING_RATIO];

	// Input and output device pointers
	ConvertedKeyType	*d_in_keys;
	SizeT				*d_spine;

	// Bit-twiddling operator needed to make keys suitable for radix sorting
	IngressOp			ingress_op;

	int 				warp_id;
	int 				warp_tid;

	DigitCounter		*base_counter;


	//---------------------------------------------------------------------
	// Helper structure for templated iteration
	//---------------------------------------------------------------------

	// Iterate
	template <int COUNT, int MAX>
	struct Iterate
	{
		// BucketKeys
		static __device__ __forceinline__ void BucketKeys(
			Cta &cta,
			ConvertedKeyType keys[THREAD_ELEMENTS])
		{
			cta.Bucket(keys[COUNT]);

			// Next
			Iterate<COUNT + 1, MAX>::BucketKeys(cta, keys);
		}

		// ProcessTiles
		static __device__ __forceinline__ void ProcessTiles(Cta &cta, SizeT cta_offset)
		{
			cta.ProcessFullTile(cta_offset);

			// Next
			Iterate<COUNT + 1, MAX>::ProcessTiles(cta, cta_offset + TILE_ELEMENTS);
		}
	};

	// Terminate
	template <int MAX>
	struct Iterate<MAX, MAX>
	{
		// BucketKeys
		static __device__ __forceinline__ void BucketKeys(Cta &cta, ConvertedKeyType keys[THREAD_ELEMENTS]) {}

		// ProcessTiles
		static __device__ __forceinline__ void ProcessTiles(Cta &cta, SizeT cta_offset) {}
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 	&smem_storage,
		SizeT 			*d_spine,
		KeyType 		*d_keys0,
		KeyType 		*d_keys1) :
			smem_storage(smem_storage),
			d_in_keys(reinterpret_cast<ConvertedKeyType*>(FLOP_TURN ? d_keys1 : d_keys0)),
			d_spine(d_spine),
			warp_id(threadIdx.x >> LOG_WARP_THREADS),
			warp_tid(util::LaneId())
	{
		base_counter = smem_storage.digit_counters[warp_id][warp_tid];
	}


	/**
	 * Decode a key and increment corresponding smem digit counter
	 */
	__device__ __forceinline__ void Bucket(ConvertedKeyType key)
	{
		// Compute byte offset of smem counter.  Add in thread column.
		unsigned int byte_offset = (threadIdx.x << (LOG_PACKING_RATIO + LOG_BYTES_PER_COUNTER));

		// Perform transform op
		ConvertedKeyType converted_key = ingress_op(key);

		// Add in sub-counter byte_offset
		byte_offset = Extract<
			CURRENT_BIT,
			LOG_PACKING_RATIO,
			LOG_BYTES_PER_COUNTER>(
				converted_key,
				byte_offset);

		// Add in row byte_offset
		byte_offset = Extract<
			CURRENT_BIT + LOG_PACKING_RATIO,
			LOG_COUNTER_LANES,
			LOG_THREADS + (LOG_PACKING_RATIO + LOG_BYTES_PER_COUNTER)>(
				converted_key,
				byte_offset);

		// Increment counter
		DigitCounter *counter = (DigitCounter*) (smem_storage.counter_base + byte_offset);
		(*counter)++;
	}


	/**
	 * Reset composite counters
	 */
	__device__ __forceinline__ void ResetDigitCounters()
	{
		#pragma unroll
		for (int LANE = 0; LANE < COUNTER_LANES; LANE++)
		{
			smem_storage.packed_counters[LANE][threadIdx.x] = 0;
		}
	}


	/**
	 * Reset the unpacked counters in each thread
	 */
	__device__ __forceinline__ void ResetUnpackedCounters()
	{
		#pragma unroll
		for (int LANE = 0; LANE < LANES_PER_WARP; LANE++)
		{
			#pragma unroll
			for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO; UNPACKED_COUNTER++)
			{
				local_counts[LANE][UNPACKED_COUNTER] = 0;
			}
		}
	}


	/**
	 * Extracts and aggregates the digit counters for each counter lane
	 * owned by this warp
	 */
	__device__ __forceinline__ void UnpackDigitCounts()
	{
		if (warp_id < COUNTER_LANES)
		{
			#pragma unroll
			for (int LANE = 0; LANE < LANES_PER_WARP; LANE++)
			{
				const int COUNTER_LANE = LANE * WARPS;

				#pragma unroll
				for (int PACKED_COUNTER = 0; PACKED_COUNTER < THREADS; PACKED_COUNTER += WARP_THREADS)
				{
					#pragma unroll
					for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO; UNPACKED_COUNTER++)
					{
						const int OFFSET = (((COUNTER_LANE * THREADS) + PACKED_COUNTER) * PACKING_RATIO) + UNPACKED_COUNTER;
						local_counts[LANE][UNPACKED_COUNTER] += *(base_counter + OFFSET);
					}
				}
			}
		}
	}


	/**
	 * Places unpacked counters into smem for final digit reduction
	 */
	__device__ __forceinline__ void ReduceUnpackedCounts()
	{
		// Place unpacked digit counters in shared memory
		if (warp_id < COUNTER_LANES)
		{
			#pragma unroll
			for (int LANE = 0; LANE < LANES_PER_WARP; LANE++)
			{
				const int COUNTER_LANE = LANE * WARPS;
				int digit_row = (COUNTER_LANE + warp_id) << LOG_PACKING_RATIO;

				#pragma unroll
				for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO; UNPACKED_COUNTER++)
				{
					smem_storage.digit_partials[digit_row + UNPACKED_COUNTER][warp_tid]
						  = local_counts[LANE][UNPACKED_COUNTER];
				}
			}
		}

		__syncthreads();

		// Rake-reduce and write out the bin_count reductions
		if (threadIdx.x < RADIX_DIGITS)
		{
			SizeT bin_count = util::reduction::SerialReduce<WARP_THREADS>::Invoke(
				smem_storage.digit_partials[threadIdx.x]);

			int spine_bin_offset = (gridDim.x * threadIdx.x) + blockIdx.x;

			util::io::ModifiedStore<KernelPolicy::STORE_MODIFIER>::St(
				bin_count,
				d_spine + spine_bin_offset);
		}
	}


	/**
	 * Processes a single, full tile
	 */
	__device__ __forceinline__ void ProcessFullTile(SizeT cta_offset)
	{
		// Tile of keys
		ConvertedKeyType keys[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Read tile of keys
		util::io::LoadTile<
			LOG_LOADS_PER_TILE,
			LOG_LOAD_VEC_SIZE,
			THREADS,
			KernelPolicy::LOAD_MODIFIER,
			false>::LoadValid(
				(ConvertedKeyType (*)[LOAD_VEC_SIZE]) keys,
				d_in_keys,
				cta_offset);

		// Prevent bucketing from being hoisted (otherwise we don't get the desired outstanding loads)
		if (LOADS_PER_TILE > 1) __syncthreads();

		// Bucket tile of keys
		Iterate<0, THREAD_ELEMENTS>::BucketKeys(*this, (ConvertedKeyType*) keys);
	}


	/**
	 * Processes a single load (may have some threads masked off)
	 */
	__device__ __forceinline__ void ProcessPartialTile(
		SizeT cta_offset,
		const SizeT &out_of_bounds)
	{
		// Process partial tile if necessary using single loads
		cta_offset += threadIdx.x;
		while (cta_offset < out_of_bounds)
		{
			// Load and bucket key
			ConvertedKeyType key = d_in_keys[cta_offset];
			Bucket(key);
			cta_offset += THREADS;
		}
	}


	/**
	 * Process work range of tiles
	 */
	__device__ __forceinline__ void ProcessWorkRange(
		util::CtaWorkLimits<SizeT> &work_limits)
	{
		// Reset digit counters in smem and unpacked counters in registers
		ResetDigitCounters();
		ResetUnpackedCounters();

		SizeT cta_offset = work_limits.offset;

		// Unroll batches of full tiles
		while (cta_offset + UNROLLED_ELEMENTS < work_limits.out_of_bounds)
		{
			Iterate<0, UNROLL_COUNT>::ProcessTiles(*this, cta_offset);
			cta_offset += UNROLLED_ELEMENTS;

			__syncthreads();

			// Aggregate back into local_count registers to prevent overflow
			UnpackDigitCounts();

			__syncthreads();

			// Reset composite counters in lanes
			ResetDigitCounters();
		}

		// Unroll single full tiles
		while (cta_offset < work_limits.guarded_offset)
		{
			ProcessFullTile(cta_offset);
			cta_offset += TILE_ELEMENTS;
		}

		// Process partial tile if necessary
		ProcessPartialTile(cta_offset, work_limits.out_of_bounds);

		__syncthreads();

		// Aggregate back into local_count registers
		UnpackDigitCounts();

		__syncthreads();

		// Final raking reduction of counts by bin, output to spine.
		ReduceUnpackedCounts();
	}
};



} // namespace upsweep
} // namespace radix_sort
} // namespace b40c

