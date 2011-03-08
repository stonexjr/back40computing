/******************************************************************************
 * 
 * Copyright 2010-2011 Duane Merrill
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
 * "Metatype" for guiding copy granularity configuration
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/basic_utils.cuh>
#include <b40c/util/data_movement_load.cuh>
#include <b40c/util/data_movement_store.cuh>
#include <b40c/copy/problem_type.cuh>

namespace b40c {
namespace copy {

/**
 * Copy kernel granularity configuration meta-type.  Parameterizations of this
 * type encapsulate our kernel-tuning parameters (i.e., they are reflected via
 * the static fields).
 *
 * Kernels can be specialized for problem-type, SM-version, etc. by parameterizing
 * them with different performance-tuned parameterizations of this type.  By
 * incorporating this type into the kernel code itself, we guide the compiler in
 * expanding/unrolling the kernel code for specific architectures and problem
 * types.
 */
template <
	// ProblemType type parameters
	typename ProblemType,

	// Machine parameters
	int CUDA_ARCH,

	// Tunable parameters
	int _MAX_CTA_OCCUPANCY,
	int _LOG_THREADS,
	int _LOG_LOAD_VEC_SIZE,
	int _LOG_LOADS_PER_TILE,
	util::ld::CacheModifier _READ_MODIFIER,
	util::st::CacheModifier _WRITE_MODIFIER,
	bool _WORK_STEALING,
	int _LOG_SCHEDULE_GRANULARITY>

struct KernelConfig : ProblemType
{
	typedef ProblemType ProblemType;
	typedef typename ProblemType::T T;

	static const util::ld::CacheModifier READ_MODIFIER 		= _READ_MODIFIER;
	static const util::st::CacheModifier WRITE_MODIFIER 	= _WRITE_MODIFIER;

	static const bool WORK_STEALING		= _WORK_STEALING;

	enum {

		LOG_THREADS 					= _LOG_THREADS,
		THREADS							= 1 << LOG_THREADS,

		LOG_LOAD_VEC_SIZE  				= _LOG_LOAD_VEC_SIZE,
		LOAD_VEC_SIZE					= 1 << LOG_LOAD_VEC_SIZE,

		LOG_LOADS_PER_TILE 				= _LOG_LOADS_PER_TILE,
		LOADS_PER_TILE					= 1 << LOG_LOADS_PER_TILE,

		LOG_WARPS						= LOG_THREADS - B40C_LOG_WARP_THREADS(CUDA_ARCH),
		WARPS							= 1 << LOG_WARPS,

		LOG_TILE_ELEMENTS_PER_THREAD	= LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE,
		TILE_ELEMENTS_PER_THREAD		= 1 << LOG_TILE_ELEMENTS_PER_THREAD,

		LOG_TILE_ELEMENTS 				= LOG_TILE_ELEMENTS_PER_THREAD + LOG_THREADS,
		TILE_ELEMENTS					= 1 << LOG_TILE_ELEMENTS,

		LOG_SCHEDULE_GRANULARITY		= _LOG_SCHEDULE_GRANULARITY,
		SCHEDULE_GRANULARITY			= 1 << LOG_SCHEDULE_GRANULARITY,

		THREAD_OCCUPANCY				= B40C_SM_THREADS(CUDA_ARCH) >> LOG_THREADS,
		CTA_OCCUPANCY  					= B40C_MIN(_MAX_CTA_OCCUPANCY, B40C_MIN(B40C_SM_CTAS(CUDA_ARCH), THREAD_OCCUPANCY))
	};

	static const bool VALID				= (CTA_OCCUPANCY > 0);
};


} // namespace copy
} // namespace b40c

