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
//  Created by Brian Smith on 7/15/22.
//

import Foundation
import MLCompute
import MimicTransferables

extension MlComputeLayerConvertable {
    func makeMlcFullyConnectedLayer() throws -> MLCLayer? {
        guard
            let weights = weights,
            let inputFeatureChannelCount = inputFeatureChannelCount,
            let outputFeatureChannelCount = outputFeatureChannelCount,
            let weightsDescriptor = MLCTensorDescriptor(shape: [1, outputFeatureChannelCount, inputFeatureChannelCount],
                                                        dataType: dataType.mlcDataType)
        else {
            return nil
        }
        let descriptor = MLCConvolutionDescriptor(kernelSizes: (inputFeatureChannelCount, outputFeatureChannelCount),
                                                  inputFeatureChannelCount: inputFeatureChannelCount,
                                                  outputFeatureChannelCount: outputFeatureChannelCount)
        let platformWeights = try MlComputeWeightsFactory(weightsTensor: weights,
                                                          weightsDescriptor: weightsDescriptor) ()
        
        let biasesReshape: [Int]?
        if let biases = biases {
            biasesReshape =  [1] + biases.shape
        } else {
            biasesReshape = nil
        }
        
        let platformLayer = MLCFullyConnectedLayer(weights: platformWeights,
                                                   biases: biases?.makeMlcTensor(reshape: biasesReshape),
                                                   descriptor: descriptor)
        if let label = label {
            platformLayer?.label = label
        }
        return platformLayer
    }
}
