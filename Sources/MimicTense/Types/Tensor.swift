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

public struct Tensor<NativeType: NeuralNativeType> {
    public enum ThrowableError: Error {
        case uncovertableData
    }
    
    // TODO: handle feature channel position in shape. Perhaps, create a "Shape" type.
    public let shape: [Int]
    
    public init(shape: [Int]) {
        self.shape = shape
        data = nil
    }
    
    public init(_ data: NativeType) {
        shape = []
        self.data = data
    }
        
    public init(_ data: [NativeType]) {
        shape = [data.count]
        self.data = data
    }
    
    public init(_ data: [[NativeType]]) {
        shape = [data.count,
                 data.first?.count ?? 0]
        self.data = data
    }
    
    public init(_ data: [[[NativeType]]]) {
        shape = [data.count,
                 data.first?.count ?? 0,
                 data.first?.count ?? 0]
        self.data = data
    }
    
    public init(_ data: [[[[NativeType]]]])  {
        shape = [data.count,
                 data.first?.count ?? 0,
                 data.first?.first?.count ?? 0,
                 data.first?.first?.first?.count ?? 0]
        self.data = data
    }
    
    public subscript (_ index: Int) -> Tensor {
        switch shape.count {
        case 4:
            let data = data as! [[[[NativeType]]]]
            let subdata = data[index]
            return Tensor(subdata)
        case 3:
            let data = data as! [[[NativeType]]]
            let subdata = data[index]
            return Tensor(subdata)
        case 2:
            let data = data as! [[NativeType]]
            let subdata = data[index]
            return Tensor(subdata)
        case 1:
            let data = data as! [NativeType]
            let subdata = data[index]
            return Tensor(subdata)
        case 0:
            return Tensor(data as! NativeType)
        default:
            fatalError()
        }
    }
    
    public var rank0Data: NativeType? {
        data as? NativeType
    }
    
    public var rank1Data: [NativeType]? {
        data as? [NativeType]
    }
    
    public var rank2Data: [[NativeType]]? {
        data as? [[NativeType]]
    }
    
    public var rank3Data: [[[NativeType]]]? {
        data as? [[[NativeType]]]
    }
    
    public var rank4Data: [[[[NativeType]]]]? {
        data as? [[[[NativeType]]]]
    }
    
    private let data: Any?
    
    private func flatten() -> [NativeType]? {
        switch shape.count {
        case 0:
            guard let data = rank0Data else { return nil }
            return [data]
        case 1:
            guard let data = rank1Data else { return nil }
            return data
        case 2:
            guard let data = rank2Data else { return nil }
            return Array(data.joined())
        case 3:
            guard let data = rank3Data else { return nil }
            return Array(data.joined().joined())
        case 4:
            guard let data = rank4Data else { return nil }
            return Array(data.joined().joined().joined())
        default:
            return nil
        }
    }
}

extension Tensor {
    static func make<NativeType: NeuralNativeType>(from transferableTensor: MimicTransferables.Tensor) throws -> Tensor<NativeType>
    {
        guard
            MimicTransferables.DataType(NativeType.self) == transferableTensor.dataType,
            let tensorData = transferableTensor.extract(NativeType.self)
        else {
            throw ThrowableError.uncovertableData
        }
        switch tensorData {
        case let scalar as NativeType:
            return Tensor<NativeType>(scalar)
        case let vector as [NativeType]:
            return Tensor<NativeType>(vector)
        case let matrix as [[NativeType]]:
            return Tensor<NativeType>(matrix)
        case let rank3Array as [[[NativeType]]]:
            return Tensor<NativeType>(rank3Array)
        case let rank4Array as [[[[NativeType]]]]:
            return Tensor<NativeType>(rank4Array)
        default:
            throw ThrowableError.uncovertableData
        }
    }
    
    func makeTransferable() throws -> MimicTransferables.Tensor {
        switch data {
        case let scalar as NativeType:
            return MimicTransferables.Tensor(scalar)
        case let vector as [NativeType]:
            return MimicTransferables.Tensor(vector)
        case let matrix as [[NativeType]]:
            return MimicTransferables.Tensor(matrix)
        case let rank3Array as [[[NativeType]]]:
            return MimicTransferables.Tensor(rank3Array)
        case let rank4Array as [[[[NativeType]]]]:
            return MimicTransferables.Tensor(rank4Array)
        case .none:
            return MimicTransferables.Tensor(shape: shape, dataType: DataType(NativeType.self))
        default:
            throw ThrowableError.uncovertableData
        }
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
        guard shape == other.shape else { return false }
        if data == nil && other.data == nil { return true }
        guard
            let vector = flatten(),
            let otherVector = other.flatten()
        else {
            return false
        }
        return zip(vector, otherVector).reduce(true) { partialResult, elements in
            partialResult && (abs(elements.0 - elements.1) < accuracy)
        }
    }
}
