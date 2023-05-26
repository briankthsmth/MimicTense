//
//  Copyright 2023 Brian Keith Smith
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//
//  Created by Brian Smith on 1/25/23.
//

import Foundation

extension Tensor {
    /// Subscripting method for tensors.
    ///
    ///  - Parameters:
    ///    - range: The range of the sub-tensor to create.
    ///
    ///  - Returns: A new Tensor for the given range.
    public subscript(range: Swift.Range<Index>) -> Tensor {
        let subshape = shape[1...]
        let subshapeSize = subshape.isEmpty ? 1 : subshape.reduce(1, *)
        let rangeShape = [range.count] + shape[1...]
        let startIndex = range.lowerBound * subshapeSize * dataType.memoryLayoutSize
        let endIndex = range.upperBound * subshapeSize * dataType.memoryLayoutSize
        let rangeData = data[startIndex ..< endIndex]
        return Tensor(shape: rangeShape,
               data: Array(rangeData),
               dataType: dataType,
               featureChannelPosition: featureChannelPosition)
    }
    
    /// Subscripting method using a closed range.
    ///
    /// - Parameters:
    ///   - range: A closed range to extract from the tensor.
    ///
    /// - Returns: A new Tensor for the closed range.
    public subscript(range: ClosedRange<Index>) -> Tensor {
        self[Swift.Range(range)]
    }
    
    ///  Subscripting method for a single index.
    ///
    ///  - Parameters:
    ///    - index: The index to extract from the tensor
    ///
    ///  - Returns:A new Tensor with the sub-tensor for the given index.
    public subscript(index: Index) -> Tensor {
        self[index ... index]
    }
}
