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
//  Created by Brian Smith on 6/6/22.
//

import Foundation
import MLCompute
import MimicTransferables

extension MlComputeLayerConvertable {
    func makeMlcConvolutionLayer() throws -> MLCLayer? {
        guard
            let weights = weights,
            let kernelSize = kernelSize,
            let inputFeatureChannelCount = inputFeatureChannelCount,
            let outputFeatureChannelCount = outputFeatureChannelCount,
            let weightsDescriptor = MLCTensorDescriptor(convolutionWeightsWithWidth: kernelSize.width,
                                                        height: kernelSize.height,
                                                        inputFeatureChannelCount: inputFeatureChannelCount,
                                                        outputFeatureChannelCount: outputFeatureChannelCount,
                                                        dataType: dataType.mlcDataType)
        else {
            return nil
        }
        
        
        
        let platformWeightsTensor = try MlComputeWeightsFactory(weightsTensor: weights,
                                                            weightsDescriptor: weightsDescriptor) ()
        let convolutionDescriptor = MLCConvolutionDescriptor(kernelSizes: (kernelSize.height, kernelSize.width),
                                                             inputFeatureChannelCount: inputFeatureChannelCount, outputFeatureChannelCount: outputFeatureChannelCount)
        return MLCConvolutionLayer(weights: platformWeightsTensor,
                                   biases: nil,
                                   descriptor: convolutionDescriptor)
    }
}
