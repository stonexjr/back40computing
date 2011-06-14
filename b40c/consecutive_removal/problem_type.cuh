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
 * Consecutive Removal Problem Type
 ******************************************************************************/

#pragma once

namespace b40c {
namespace consecutive_removal {


/**
 * Type of consecutive removal problem
 */
template <
	typename _T,
	typename _SizeT>
struct ProblemType
{
	typedef _T 				T;
	typedef _SizeT 			SizeT;
	typedef SizeT 			SpineType;
	typedef int 			SpineSizeT;
};


} // namespace consecutive_removal
} // namespace b40c

