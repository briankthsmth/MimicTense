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
//  Created by Brian Smith on 4/19/22.
//

import Foundation
import MimicTransferables

/// Structure to represent tensors.
public struct Tensor<NativeType: NeuralNativeType> {
    /// Enumeration types that can be thrown as errors.
    public enum ThrowableError: Error {
        case uncovertableData
    }
    
    /// Enumeration for the rank of the tensor.
    public enum Rank {
        /// Indicates a rank 0 (scalar) tensor.
        case scalar
        /// Indicates a rank 1 (vector) tensor.
        case vector
        /// Indicates a rank 2 (matrix) tensor.
        case matrix
        /// Indicates a rank 3 tensor.
        case rank3
        /// Indicates a rank 4 tensor.
        case rank4
        
        init(_ shapeCount: Int) {
            switch shapeCount {
            case 0:
                self = .scalar
            case 1:
                self = .vector
            case 2:
                self = .matrix
            case 3:
                self = .rank3
            case 4:
                self = .rank4
            default:
                fatalError("Tensor with unsupported rank.")
            }
        }
    }
    
    /// The shape of the tensor represented by an array of Int. This can only contain up to four elements.
    public let shape: [Int]
    /// Optional value to set a randomizer type to use to initialize the tensor data.
    public let randomizer: Randomizer?
    
    /// The tensor's rank.
    public var rank: Rank {
        Rank(shape.count)
    }
    
    /// Initializer to create a placeholder tensor.
    ///
    /// Creates a tensor object with only the shape and no data which can be used as
    ///  a placeholder for describing a tensor. If a randomizer type is supplied, this could
    ///  generate data depending on the randomizer.
    ///
    ///  - Parameters:
    ///    - shape: The shape as an array containing an element for each deminsion in the Tensor. An empty array signifies a scalar.
    ///    - randomizer: The random method type to use to initialize the tensor data.
    public init(shape: [Int], randomizer: Randomizer? = nil) {
        assert(shape.count <= 4, "Only Tensors up to rank 4 supported.")
        self.shape = shape
        self.randomizer = randomizer
        data = []
    }
    
    public init(_ data: NativeType) {
        self.init(shape: [], data: [data])
    }
        
    public init(_ data: [NativeType]) {
        self.init(shape: [data.count], data: data)
    }
    
    public init(_ data: [[NativeType]]) {
        let shape = [data.count,
                     data.first?.count ?? 0]
        let data = Array(data.joined())
        self.init(shape: shape, data: data)
    }
    
    public init(_ data: [[[NativeType]]]) {
        let shape = [data.count,
                     data.first?.count ?? 0,
                     data.first?.count ?? 0]
        let data = Array(data.joined().joined())
        self.init(shape: shape, data: data)
    }
    
    public init(_ data: [[[[NativeType]]]])  {
        let shape = [data.count,
                     data.first?.count ?? 0,
                     data.first?.first?.count ?? 0,
                     data.first?.first?.first?.count ?? 0]
        let data = Array(data.joined().joined().joined())
        self.init(shape: shape, data: data)
    }
    
    public subscript (_ index: Int) -> Tensor {
        switch rank {
        case .rank4, .rank3, .matrix:
            let newShape = shape[1...]
            let subarraySize = newShape.reduce(1, *)
            let subarrayStart = index * subarraySize
            let subarrayEnd = subarrayStart + subarraySize
            let subdata = data[subarrayStart ..< subarrayEnd]
            return Tensor(shape: Array(newShape), data: Array(subdata))
        case .vector:
            return Tensor(data[index])
        case .scalar:
            return Tensor(data[0])
        }
    }
    
    public var rank0Data: NativeType? {
        guard shape.count == 0 else { return nil }
        return data.first
    }
    
    public var rank1Data: [NativeType]? {
        guard shape.count == 1 else { return nil }
        return data
    }
    
    public var rank2Data: [[NativeType]]? {
        guard shape.count == 2 else { return nil }
        return data.divided(by: shape[1])
    }
    
    public var rank3Data: [[[NativeType]]]? {
        guard shape.count == 3 else { return nil }
        return data.divided(by: shape[2]).divided(by: shape[1])
    }
    
    public var rank4Data: [[[[NativeType]]]]? {
        guard shape.count == 4 else { return nil }
        return data.divided(by: shape[3]).divided(by: shape[2]).divided(by: shape[1])
    }
        
    
    internal let data: [NativeType]
    
    internal init(shape: [Int], data: [NativeType], randomizer: Randomizer? = nil) {
        self.shape = shape
        self.data = data
        self.randomizer = randomizer
    }
    
}

extension Tensor: Equatable {
    public static func == (lhs: Tensor, rhs: Tensor) -> Bool {
        return lhs.shape == rhs.shape &&
        lhs.rank0Data == rhs.rank0Data &&
        lhs.rank1Data == rhs.rank1Data &&
        lhs.rank2Data == rhs.rank2Data &&
        lhs.rank3Data == rhs.rank3Data &&
        lhs.rank4Data == rhs.rank4Data
    }
    
    public func isEqual(_ other: Tensor, accuracy: NativeType) -> Bool where NativeType == Float {
        guard
            shape == other.shape,
            data.count == other.data.count
        else {
            return false
        }
        if data.isEmpty && other.data.isEmpty { return true }
        return zip(data, other.data).reduce(true) { partialResult, elements in
            partialResult && (abs(elements.0 - elements.1) < accuracy)
        }
    }
}

extension Array {
    fileprivate func divided(by stride: Int) -> [[Element]] {
        let divisions = count / stride
        let range = 0 ..< divisions
        return range.map {
            let start = $0 * stride
            let end = start + stride
            return Array(self[start ..< end])
        }
    }
}
