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
//  Created by Brian Smith on 5/26/23.
//

import Foundation
import MimicCore

/// A transferable range for a given data type.
public struct Range: Transferable, Equatable {
    
    /// The type of range.
    public enum Kind: Transferable {
        /// Half open range from the minimum up to, not including, the maximum.
        case halfOpen
        /// Closed range from the minimum to, including, the maximum.
        case closed
    }
    
    /// The range kind.
    public let kind: Kind
    /// The minimum as an array of UInt8 values.
    public let lowerBound: [UInt8]
    /// The maximum as an array of UInt8 values.
    public let upperBound: [UInt8]
    /// The data type for the minimum and maximum values.
    public let dataType: DataType
    
    /// Initializer to create a range.
    ///
    /// - Parameters:
    ///   - range: The type of range.
    ///   - lowerBound: The lower bound for the range.
    ///   - upperBound: The upper bound for the range.
    ///
    public init<NativeType: SupportedNativeType>(kind: Kind, lowerBound: NativeType, upperBound: NativeType) {
        self.kind = kind
        self.lowerBound = lowerBound.makeBuffer()
        self.upperBound = upperBound.makeBuffer()
        self.dataType = DataType(NativeType.self)
    }
    
    /// Initializer with a swift Range.
    ///
    /// - Parameters:
    ///   - range: A Swift Range.
    ///
    public init<NativeType: SupportedNativeType>(_ range: Swift.Range<NativeType>) {
        self.init(kind: .halfOpen, lowerBound: range.lowerBound, upperBound: range.upperBound)
    }
    
    /// Initializer with a swift ClosedRange
    ///
    /// - Parameters:
    ///   - range: A Swift ClosedRange
    ///
    public init<NativeType: SupportedNativeType>(_ range: ClosedRange<NativeType>) {
        self.init(kind: .closed, lowerBound: range.lowerBound, upperBound: range.upperBound)
    }
}
