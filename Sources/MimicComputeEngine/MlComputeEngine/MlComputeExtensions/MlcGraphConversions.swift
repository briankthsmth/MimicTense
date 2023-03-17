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
//  Created by Brian Smith on 2/1/23.
//

import Foundation
import MLCompute
import MimicTransferables

extension MLCGraph {
    /// Factory method to make a transferable Graph object from a MLCGraph object.
    ///
    ///  If the original Graph object is provided, it will be used to extract layer labels as MLCompute modifes them.
    ///
    /// - Parameters:
    ///    - originalGraph: The original Graph used to create the MLCGraph
    ///
    /// - Returns:A Graph object.
    func makeGraph(against originalGraph: Graph? = nil) throws -> Graph {
        guard let firstLayer = layers.first else {
            throw ComputeEngineError.layerConversion
        }
        let featureChannelPosition: FeatureChannelPosition
        let platformLayers: [MLCLayer]
        if
            layers.count > 2,
            let firstTranspose = layers.first as? MLCTransposeLayer,
            let lastTranspose = layers.last as? MLCTransposeLayer,
            firstTranspose.dimensions == [0, 3, 1, 2],
            lastTranspose.dimensions == [0, 2, 3, 1]
        {
            featureChannelPosition = .last
            platformLayers = Array(layers[1 ..< layers.endIndex - 1])
        } else {
            featureChannelPosition = .first
            platformLayers = layers
        }
    
        let layers = try platformLayers
            .enumerated()
            .map {
                try $1.makeLayer(against: originalGraph?.layers[$0])
            }
        let sourceTensors = sourceTensors(for: firstLayer).map{ $0.makeTensor() }
        return Graph(kind: .sequential,
                     dataType: .float32,
                     inputTensors: [sourceTensors],
                     layers: layers,
                     featureChannelPosition: featureChannelPosition)
    }
}
