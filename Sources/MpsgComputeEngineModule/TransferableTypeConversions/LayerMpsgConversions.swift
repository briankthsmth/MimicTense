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
//  Created by Brian Smith on 6/2/23.
//

import Foundation
import MetalPerformanceShadersGraph

import MimicTransferables
import MimicComputeEngineModule

extension Layer {
    typealias TensorPair = (placeholder: MPSGraphTensor, data: MPSGraphTensorData)
    
    func addAdditionLayer(to graph: MPSGraph, inputs: [MPSGraphTensor]) throws -> MPSGraphTensor {
        guard inputs.count == 2 else { throw ComputeEngineLayerInputsError() }
        return graph.addition(inputs[0], inputs[1], name: nil)
    }
    
    func addConvolutionLayer(to graph: MPSGraph,
                             device: MPSGraphDevice,
                             inputs: [MPSGraphTensor]) throws -> (output: MPSGraphTensor, weightsPair: TensorPair)
    {
        guard let descriptor = MPSGraphConvolution2DOpDescriptor(strideInX: 1,
                                                                 strideInY: 1,
                                                                 dilationRateInX: 1,
                                                                 dilationRateInY: 1,
                                                                 groups: 1,
                                                                 paddingStyle: .TF_SAME,
                                                                 dataLayout: .NHWC,
                                                                 weightsLayout: .HWIO),
              let weights = weights,
              let input = inputs.first
        else {
            throw ComputeEngineLayerInputsError()
        }
        
        let mpsgWeights = weights.makeMpsgTensor(for: graph)
        let mpsgWeightsData = try weights.makeMpsgTensorData(for: device)
        let mpsgOutput = graph.convolution2D(input,
                                             weights: mpsgWeights,
                                             descriptor: descriptor,
                                             name: nil)
        return (mpsgOutput, (mpsgWeights, mpsgWeightsData))
    }
}
