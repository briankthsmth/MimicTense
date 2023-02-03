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
//  Created by Brian Smith on 1/26/23.
//

import Foundation
import MLCompute
import MimicTransferables

extension MLCLayer {
    /// Factory method to make transferable Layer object from a MLCLayer object.
    ///
    /// - Returns: A Layer object.
    func makeLayer() throws -> Layer {
        switch self {
        case let fullyConnectedLayer as MLCFullyConnectedLayer:
            // MLCompute wants a batch count of 1 in weights and biases, which seems
            // unnessary. So, they are reshape to exclude it.
            let weights = fullyConnectedLayer.weights.makeTensor().map {
                Tensor($0, shape: Array($0.shape[1...]))
            }
            let biases = fullyConnectedLayer.biases?.makeTensor().map {
                Tensor($0, shape: Array($0.shape[1...]))
            }
            return Layer(label: fullyConnectedLayer.label,
                         kind: .fullyConnected,
                         dataType: weights.dataType,
                         inputFeatureChannelCount: fullyConnectedLayer.descriptor.inputFeatureChannelCount,
                         outputFeatureChannelCount: fullyConnectedLayer.descriptor.outputFeatureChannelCount,
                         weights: weights,
                         biases: biases)
        case let convolutionLayer as MLCConvolutionLayer:
            // MLCompute wants the weights in the shape in form
            //  [ batchSize = 1, output channels * feature channels, kernel height, kernel width]
            // The code reshapes this to,
            //  [ output channels, feature channels, kernel height, kernel width]
            let weights = convolutionLayer.weights.makeTensor().map {
                Tensor($0, shape: [
                    convolutionLayer.descriptor.outputFeatureChannelCount,
                    convolutionLayer.descriptor.inputFeatureChannelCount,
                    convolutionLayer.descriptor.kernelSizes.height,
                    convolutionLayer.descriptor.kernelSizes.width
                ],
                       featureChannelPosition: .first)
            }
            let biases = convolutionLayer.biases?.makeTensor().map {
                Tensor($0, shape: Array($0.shape[1...]))
            }
            
            return Layer(label: convolutionLayer.label,
                         kind: .convolution,
                         dataType: .float32,
                         kernelSize: Layer.KernelSize(height: convolutionLayer.descriptor.kernelSizes.height,
                                                      width: convolutionLayer.descriptor.kernelSizes.width),
                         inputFeatureChannelCount: convolutionLayer.descriptor.inputFeatureChannelCount,
                         outputFeatureChannelCount: convolutionLayer.descriptor.outputFeatureChannelCount,
                         weights: weights,
                         biases: biases)
            
        default:
            throw ComputeEngineError.layerConversion
        }
    }
}
