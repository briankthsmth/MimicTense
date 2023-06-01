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

/// Transferable type for tensors.
public struct Tensor: Equatable, Transferable {
    /// Represents the integer index into the tensor data.
    public typealias Index = Int
    /// Shape of the tensor as an array of integer values.
    public internal(set) var shape: [Int]
    /// The data as an array of UInt8 elements.
    public internal(set) var data: [UInt8]
    /// The data's type.
    public let dataType: DataType
    /// Enumeration to specify how to interpert a feature channel.
    public internal(set) var featureChannelPosition: FeatureChannelPosition
    /// Descripter for generating random data.
    public internal(set) var randomDescriptor: RandomDescriptor?
    
    public var featureChannelCount: Int {
        guard shape.count == 4 else { return 0 }
        return featureChannelPosition == .first ? shape[1] : shape[3]
    }
    
    /// Initializer
    ///
    ///
    /// - Parameters:
    ///   - shape: The shape as an array of Int values.
    ///   - data: The data in buffer of UInt8 values. The default is an empty array
    ///   for a placeholder tensor or when initializing with random data.
    ///   - dataType: The data's type.
    ///   - featureChannelPosition: Specifies position of a feature channel in the shape.
    ///   The default is set to last position.
    ///   - randomDescriptor: A descriptor for generating random numbers from a distribution.
    ///   
    public init(shape: [Int],
                data: [UInt8] = [],
                dataType: DataType,
                featureChannelPosition: FeatureChannelPosition = .last,
                randomDescriptor: RandomDescriptor? = nil
    ) {
        self.shape = shape
        self.data = data
        self.dataType = dataType
        self.featureChannelPosition = shape.count == 4 ? featureChannelPosition : .notApplicable
        self.randomDescriptor = randomDescriptor
    }
    
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
}
