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
//
//  Created by Brian Smith on 9/29/22.
//

import Foundation
import MLCompute
import MimicTransferables

protocol MlComputeGraphTransformable: PlatformGraphTransformable
    where PlatformGraph == MLCGraph, PlatformTensor == MLCTensor {}

extension MlComputeGraphTransformable {
    static func makePlatformGraph(from graph: Graph) throws -> GraphTransformProducts<PlatformGraph, PlatformTensor>
    {
        let platformGraph = MLCGraph()
        var inputs = [MLCTensor]()
        
        let layersOutput: MLCTensor? = try graph
            .layers
            .enumerated()
            .reduce(nil) { reduceTensor, arguments in
                var layerInputs = [MLCTensor]()
                if let sourceTensor = reduceTensor {
                    layerInputs = [sourceTensor]
                }
                if let layerInputTensors = graph.layerInputTensors(at: arguments.offset) {
                    var platformInputTensors = layerInputTensors.map { $0.makeMlcTensor() }
                    inputs.append(contentsOf: platformInputTensors)
                    
                    if graph.featureChannelPosition == .last {
                        platformInputTensors = try platformInputTensors.map {
                            guard
                                let transposeLayer = MLCTransposeLayer(dimensions: [0, 3, 1, 2]),
                                let outputTensor = platformGraph.node(with: transposeLayer, source: $0)
                            else {
                                throw ComputeEngineError.layerConversion
                            }
                            return outputTensor
                        }
                    }
                    layerInputs.append(contentsOf: platformInputTensors)
                }
                guard let mlcLayer = try arguments.element.makeMlComputeLayer() else { throw ComputeEngineError.layerConversion }
                guard let sourceTensor = platformGraph.node(with: mlcLayer, sources: layerInputs) else {
                    throw ComputeEngineError.layerConversion
                }
                return sourceTensor
            }
        
        guard var output = layersOutput else { throw ComputeEngineError.layerConversion }
        
        if graph.featureChannelPosition == .last {
            guard
                let transposeLayer = MLCTransposeLayer(dimensions: [0, 2, 3, 1]),
                let tensor = platformGraph.node(with: transposeLayer, source: output)
            else {
                throw ComputeEngineError.layerConversion
            }
            output = tensor
        }
                
        return GraphTransformProducts(graph: platformGraph,
                                      inputs: inputs,
                                      output: output)
    }
}
