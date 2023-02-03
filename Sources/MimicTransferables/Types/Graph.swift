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


public struct Graph: Transferable {
    public enum Kind: String, Codable {
        case sequential
    }
    
    /// The kind of graph this represents.
    public let kind: Kind
    /// The type of data used in the graph.
    public let dataType: DataType
    ///  Arrays of placeholder tensors that represent the inputs to the graph.
    public let inputTensors: [[Tensor]]
    /// The layers for the graph in the order they are connected.
    public let layers: [Layer]
    /// The position in the shape array that represents the feature channel.
    public let featureChannelPosition: FeatureChannelPosition
    
    public init(
        kind: Kind,
        dataType: DataType,
        inputTensor: Tensor,
        layers: [Layer],
        featureChannelPosition: FeatureChannelPosition
    ) {
        self.kind = kind
        self.dataType = dataType
        self.inputTensors = [[inputTensor]]
        self.layers = layers
        self.featureChannelPosition = featureChannelPosition
    }
    
    public init(
        kind: Kind,
        dataType: DataType,
        inputTensors: [[Tensor]],
        layers: [Layer],
        featureChannelPosition: FeatureChannelPosition
    ) {
        self.kind = kind
        self.dataType = dataType
        self.inputTensors = inputTensors
        self.layers = layers
        self.featureChannelPosition = featureChannelPosition
    }
}

extension Graph {
    public func layerInputTensors(at layerIndex: Int) -> [Tensor]? {
        guard layerIndex < inputTensors.count else { return nil }
        return inputTensors[layerIndex]
    }
}
