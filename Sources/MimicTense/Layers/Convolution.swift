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
import MimicCore

public struct Convolution<NativeType: NeuralNativeType>: Layerable {
    public let identifier = LayerIdentifier(kind: .convolution)
    
    public let inputs: Inputs<NativeType>?
    public let weights: Tensor<NativeType>?
    public var biases: Tensor<NativeType>?
    
    public let kernelSize: KernelSize?
    public let inputFeatureChannelCount: Int?
    public let outputFeatureChannelCount: Int?
        
    public let arithmeticOperation: ArithmeticOperation? = nil
    
    public init(kernelSize: KernelSize,
                inputFeatureChannelCount: Int,
                outputFeatureChannelCount: Int,
                weights: Tensor<NativeType>? = nil,
                _ makeInputs: (() -> Inputs<NativeType>)? = nil)
    {
        self.kernelSize = kernelSize
        self.inputFeatureChannelCount = inputFeatureChannelCount
        self.outputFeatureChannelCount = outputFeatureChannelCount
        self.weights = weights
        self.inputs = makeInputs?()
    }
}
