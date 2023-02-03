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

extension ArithmeticOperation: Codable {}

public struct Layer: Transferable {
    public enum Kind: String, Transferable {
        case arithmetic
        case convolution
        case fullyConnected
    }
    
    public struct KernelSize: Transferable, Equatable {
        public let height: Int
        public let width: Int
        
        public init(height: Int, width: Int) {
            self.height = height
            self.width = width
        }
    }
    
    // MARK: General properties
    public let label: String?
    public let dataType: DataType
    public let kind: Kind
    
    // MARK: Arithmetic layer properties
    public let arithmeticOperation: ArithmeticOperation?
    
    // MARK: Convolution layer properties
    public let kernelSize: KernelSize?
    
    // MARK: Convolution/fully connected layer properties
    public let inputFeatureChannelCount: Int?
    public let outputFeatureChannelCount: Int?
    public let weights: Tensor?
    public let biases: Tensor?
    
    public init(
        label: String? = nil,
        kind: Kind,
        dataType: DataType,
        arithmeticOperation: ArithmeticOperation? = nil,
        kernelSize: KernelSize? = nil,
        inputFeatureChannelCount: Int? = nil,
        outputFeatureChannelCount: Int? = nil,
        weights: Tensor? = nil,
        biases: Tensor? = nil
    )
    {
        self.label = label
        self.kind = kind
        self.dataType = dataType
        self.arithmeticOperation = arithmeticOperation
        self.kernelSize = kernelSize
        self.inputFeatureChannelCount = inputFeatureChannelCount
        self.outputFeatureChannelCount = outputFeatureChannelCount
        self.weights = weights
        self.biases = biases
    }
}
