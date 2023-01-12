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
//  Created by Brian Smith on 6/21/22.
//

import Foundation
import MLCompute
import MimicCore
import MimicTransferables

protocol MlComputeLayerConvertable {
    var label: String? { get }
    var dataType: DataType { get }
    var kind: Layer.Kind { get }
    var arithmeticOperation: ArithmeticOperation? { get }
    // Convolution layer properties
    var kernelSize: Layer.KernelSize? { get }
    var inputFeatureChannelCount: Int? { get }
    var outputFeatureChannelCount: Int? { get }
    var weights: Tensor? { get }
    var biases: Tensor? { get }
    
}

extension Layer: MlComputeLayerConvertable {}

extension MlComputeLayerConvertable {
    func makeMlComputeLayer() throws -> MLCLayer? {
        switch kind {
        case .arithmetic:
            return makeMlcArithmeticLayer()
        case .convolution:
            return try makeMlcConvolutionLayer()
        case .fullyConnected:
            return try makeMlcFullyConnectedLayer()
        }
    }    
}
