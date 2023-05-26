//
//  Copyright 2022 Brian Keith Smith
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
//  Created by Brian Smith on 7/12/22.
//

import Foundation

///  A container for array of tensors that is transferable to other services.
public struct TensorArray: Transferable {
    public typealias Index = Int
    
    public let tensors: [Tensor]

    public var isEmpty: Bool {
        tensors.isEmpty
    }
    
    public var count: Int {
        return tensors.count
    }
    
    public var endIndex: Int {
        guard !isEmpty else { return -1 }
        return rank > 1 ? tensors.reduce(0) { $0 + $1.shape[0] } : count
    }
    
    public var rank: Int {
        guard !isEmpty else { return -1 }
        return tensors[0].shape.count
    }
    
    public var shape: [Int] {
        guard !isEmpty else { return [] }
        return tensors[0].shape
    }
    
    public var dataType: DataType {
        tensors[0].dataType
    }

    public init(tensors: [Tensor]) {
        self.tensors = tensors
    }
    
    public subscript(range: Swift.Range<Index>) -> Tensor {
        guard rank >= 0 else {
            fatalError("Out of bounds.")
        }
        if rank == 4 {
            guard range.upperBound <= endIndex else { fatalError() }
            
            var tensorRanges = [(Int, Swift.Range<Index>)]()
            var currentIndex: Int = 0
            tensors
                .enumerated()
                .forEach { index, tensor in
                    let tensorEndIndex = tensor.shape[0] + currentIndex
                    let tensorStartIndex = currentIndex
                    if (tensorStartIndex ..< tensorEndIndex).overlaps(range) {
                        let lowerBound = range.lowerBound >= tensorStartIndex ?
                            range.lowerBound - currentIndex : tensorStartIndex - currentIndex
                        let upperBound = range.upperBound < tensorEndIndex ?
                            range.upperBound - currentIndex : tensorEndIndex - currentIndex
                        tensorRanges.append((index, lowerBound ..< upperBound))
                    }
                    currentIndex = tensor.shape[0]
                }
            return tensorRanges.reduce(into: Tensor(shape: [], dataType: dataType)) { tensor, element in
                tensor.append(tensors[element.0][element.1])
            }
        } else {
            let subArray = tensors[range]
            return subArray
                .reduce(into: Tensor(shape: [], dataType: dataType)) {
                    $0.append($1)
                }
        }
    }

    public subscript(range: ClosedRange<Index>) -> Tensor {
        self[Swift.Range(range)]
    }
}
