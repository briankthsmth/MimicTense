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
//  Created by Brian Smith on 5/20/22.
//

import Foundation
import MimicCore

public struct Tensor: Equatable, Transferable {
    public typealias Index = Int
    
    public private(set) var shape: [Int]
    public private(set) var data: [UInt8]
    public let dataType: DataType
    public private(set) var featureChannelPosition: FeatureChannelPosition
    public private(set) var randomInitializerType: RandomInitializerType?
    
    public var featureChannelCount: Int {
        guard shape.count == 4 else { return 0 }
        return featureChannelPosition == .first ? shape[1] : shape[3]
    }
    
    // MARK: Verification
    /// Test to determine if tensor is a scalar(rank 0).
    public var isScalar: Bool {
        shape.count == 0 && data.count > 0
    }
    
    ///  Total count of data needed to support the shape.
    public var shapeCount: Int {
        shape.reduce(1, *)
    }
    
    /// Byte count needed for tensor data based on the shape and data type.
    public var shapeByteCount: Int {
        shapeCount * dataType.memoryLayoutSize
    }
    
    // MARK: Initialization
    public init(shape: [Int],
                data: [UInt8] = [],
                dataType: DataType,
                featureChannelPosition: FeatureChannelPosition = .last,
                randomInitializerType: RandomInitializerType? = nil
    ) {
        self.shape = shape
        self.data = data
        self.dataType = dataType
        self.featureChannelPosition = shape.count == 4 ? featureChannelPosition : .notApplicable
        self.randomInitializerType = randomInitializerType
    }
    
    public init<NativeType: SupportedNativeType>(_ value: NativeType) {
        self.init(shape: [], value: [value])
    }
    
    public init<NativeType: SupportedNativeType>(_ value: [NativeType]) {
        self.init(shape: [value.count], value: value)
    }
    
    public init<NativeType: SupportedNativeType>(_ value: [[NativeType]]) {
        let shape = [value.count,
                     value.first?.count ?? 0]
        self.init(shape: shape, value: Array(value.joined()))
    }
    
    public init<NativeType: SupportedNativeType>(_ value: [[[NativeType]]]) {
        let shape = [value.count,
                     value.first?.count ?? 0,
                     value.first?.first?.count ?? 0]
        self.init(shape: shape, value: Array(value.joined().joined()))
    }
    
    public init<NativeType: SupportedNativeType>(_ value: [[[[NativeType]]]],
                                                 featureChannelPosition: FeatureChannelPosition = .last)
    {
        let shape = [value.count,
                     value.first?.count ?? 0,
                     value.first?.first?.count ?? 0,
                     value.first?.first?.first?.count ?? 0]
        self.init(shape: shape,
                  value: Array(value.joined().joined().joined()),
                  featureChannelPosition: featureChannelPosition)
    }
    
    public init(_ original: Tensor,
                shape: [Int]? = nil,
                data: [UInt8]? = nil,
                dataType: DataType? = nil,
                featureChannelPosition: FeatureChannelPosition? = nil,
                randomInitializerType: RandomInitializerType? = nil)
    {
        self.shape = shape ?? original.shape
        self.data = data ?? original.data
        self.dataType = dataType ?? original.dataType
        self.featureChannelPosition = featureChannelPosition ?? original.featureChannelPosition
        self.randomInitializerType = randomInitializerType ?? original.randomInitializerType
    }
    
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

    public func extractScalar<NativeType: SupportedNativeType>(_ nativeType: NativeType.Type = NativeType.self) -> NativeType? {
        guard
            DataType(nativeType) == dataType,
            data.count == MemoryLayout<NativeType>.size,
            shape.isEmpty
        else {
            return nil
        }
        return data.withUnsafeBytes { $0.load(as: nativeType) }
    }
    
    public func extract<NativeType: SupportedNativeType>(_ nativeType: NativeType.Type = NativeType.self) -> Any? {
        let dataCount = shape.reduce(1, *) * MemoryLayout<NativeType>.size
        guard
            DataType(nativeType) == dataType,
            data.count == dataCount,
            dataCount > 0
        else {
            return nil
        }
        
        var vector = [NativeType](repeating: NativeType(), count: dataCount)
        let _ = vector.withUnsafeMutableBytes { pointer in
            data.copyBytes(to: pointer, count: dataCount)
        }
        
        if shape.isEmpty { return vector[0] }
        
        var multiArray: Array<Any> = vector
        shape
            .reversed()
            .forEach { count in
                multiArray = multiArray
                    .enumerated()
                    .reduce(into: [[Any]]()) { partialResult, tuple in
                        let index = tuple.offset / count
                        var subArray = partialResult.count > index ? partialResult.remove(at: index) : []
                        subArray.append(tuple.element)
                        partialResult.append(subArray)
                    }
            }
        return multiArray[0]
    }
       
    public mutating func append(_ tensor: Tensor) {
        let fatalErrorMessage = "Incompatable tensor shapes."
        
        switch (shape.count, tensor.shape.count, data.isEmpty, tensor.data.isEmpty) {
        case (0, 0, true, false), (0, 0, false, true):
            shape = []
        case (0, 0, false, false):
            shape = [2]
        case (0, 1...3, true, false):
            shape = [1] + tensor.shape
        case (0, 4, true, false):
            shape = tensor.shape
            featureChannelPosition = tensor.featureChannelPosition
        case (1, 1, true, _):
            guard shape[0] == tensor.shape[0] else {
                fatalError(fatalErrorMessage)
            }
            shape = [1, shape[0]]
        case (1, 1, false, _):
            guard shape[0] == tensor.shape[0] else {
                fatalError(fatalErrorMessage)
            }
            shape = [2, shape[0]]
        case (2, 1, _, _):
            guard shape[1] == tensor.shape[0] else {
                fatalError()
            }
            shape = [shape[0] + 1, shape[1]]
        case (4, 4, true, _):
            guard shape[1..<4] == tensor.shape[1..<4] else {
                fatalError()
            }
            shape = [tensor.shape[0]] + shape[1..<4]
        case (4, 4, false, _):
            guard shape[1..<4] == tensor.shape[1..<4] else {
                fatalError()
            }
            shape = [shape[0] + tensor.shape[0]] + shape[1..<4]
        default:
            fatalError(fatalErrorMessage)
        }
        data = data + tensor.data
    }
    
    public func appended(_ tensor: Tensor) -> Tensor {
        var base = self
        base.append(tensor)
        return base
    }
    
    // MARK: Private Interface
    private init<NativeType: SupportedNativeType>(shape: [Int],
                                                  value: [NativeType],
                                                  featureChannelPosition: FeatureChannelPosition = .last )
    {
        self.init(shape: shape,
                  data: value.makeBuffer(),
                  dataType: DataType(NativeType.self),
                  featureChannelPosition: featureChannelPosition)
    }
}
