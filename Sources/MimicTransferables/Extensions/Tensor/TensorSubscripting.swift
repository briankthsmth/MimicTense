//
//  File.swift
//  
//
//  Created by Brian Smith on 1/25/23.
//

import Foundation

extension Tensor {
    /// Subscripting method for tensors.
    ///
    ///  - Parameters:
    ///  - range: The range of the sub-tensor to create.
    ///
    ///  - Returns: A sub-tensor for the given range.
    public subscript(range: Range<Index>) -> Tensor {
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

    public subscript(range: ClosedRange<Index>) -> Tensor {
        self[Range(range)]
    }
}
