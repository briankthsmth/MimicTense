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
//  Created by Brian Smith on 1/19/23.
//

import Foundation
import MimicTransferables

public struct LinearModel: TestModel {
    public struct Constant {
        public static let slope: Float = 0.47
        public static let intercept: Float = 0.3
        public static let batchSize = 2
        public static let inputChannels = 1
        public static let outputChannels = 1
        public static let layerLabel = "TestLayer"
    }
    
    
    public let samples: [[Float]] = [
        [1.5],
        [2.3],
        [-4.4],
        [5.2],
        [-0.8],
        [2.7],
        [0.12],
        [-3.12],
        [2.8],
        [4.2]
    ]
    public let labels: [Float]
    public let graph: Graph
    public let dataSet: DataSet
    
    public init() {
        self.labels = samples.map {
            $0[0] * Constant.slope + Constant.intercept
        }
        
        let randomDescriptor = RandomDescriptor(type: .uniform, range: Float(-1)...Float(1))
        let weights = Tensor(shape: [Constant.outputChannels, Constant.inputChannels],
                                   dataType: .float32,
                                   randomDescriptor: randomDescriptor)
        let biases = Tensor([Float](repeating: 0, count: Constant.outputChannels))
        let layer = Layer(label: Constant.layerLabel,
                          kind: .fullyConnected,
                          dataType: .float32,
                          inputFeatureChannelCount: Constant.inputChannels,
                          outputFeatureChannelCount: Constant.outputChannels,
                          weights: weights,
                          biases: biases)
        graph = Graph(kind: .sequential,
                       dataType: .float32,
                       inputTensor: Tensor(shape: [Constant.batchSize, Constant.inputChannels],
                                           dataType: .float32),
                       layers: [layer],
                       featureChannelPosition: .notApplicable)
        
        dataSet = DataSet(inputTensor: Tensor(samples),
                          labels: Tensor(labels),
                          batchSize: Constant.batchSize)
    }
}
