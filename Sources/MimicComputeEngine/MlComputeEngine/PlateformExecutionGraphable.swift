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

protocol PlateformExecutionGraphable {
}

extension PlateformExecutionGraphable {
    static func makePlatformGraphs(from graphs: [Graph]) throws -> (graphs: [MLCGraph], inputs: [MLCTensor], outputs: [MLCTensor]) {
        var platformGraphs = [MLCGraph]()
        var inputTensors = [MLCTensor]()
        var outputTensors = [MLCTensor]()
        
        try graphs
            .forEach { graph in
                let platformGraph = MLCGraph()
                
                var sourceTensors: [MLCTensor] = []
                try graph
                    .layers
                    .enumerated()
                    .forEach { index, layer in
                        if let layerInputTensors = graph.layerInputTensors(at: index) {
                            var platformInputTensors = layerInputTensors.map { $0.makeMlcTensor() }
                            inputTensors.append(contentsOf: platformInputTensors)
                            if graph.featureChannelPosition == .last {
                                platformInputTensors = platformInputTensors.map {
                                    platformGraph.node(with: MLCTransposeLayer(dimensions: [0, 3, 1, 2])!,
                                                       source: $0)!
                                }
                            }
                            sourceTensors.append(contentsOf: platformInputTensors)
                        }
                        guard let mlcLayer = try layer.makeMlComputeLayer() else { throw ComputeEngineError.layerConversion }
                        let sourceTensor = platformGraph.node(with: mlcLayer, sources: sourceTensors)!
                        sourceTensors = [sourceTensor]
                    }
                
                if graph.featureChannelPosition == .last {
                    sourceTensors = [platformGraph.node(with: MLCTransposeLayer(dimensions: [0, 2, 3, 1])!,
                                                        source: sourceTensors.first!)!]
                }
                
                platformGraphs.append(platformGraph)
                outputTensors.append(contentsOf: sourceTensors)
            }
        return (platformGraphs, inputTensors, outputTensors)
    }
}
