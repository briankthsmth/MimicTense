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
//  Created by Brian Smith on 4/22/23.
//

import Foundation
import MimicTransferables

public struct FullyConnectedNodeAdditionModel: TestModel {
    public struct Constant {
        public static let inputs = [[[Float]]]([[[2, 2]]])
        public static let weights = [[Float]]([[1, 0.5]])
        public static let biases = [Float]([2])
        public static let inputFeatureChannels = 2
        public static let outputFeatureChannels = 1
        public static let batchSize = 1
    }
    public let graph: Graph
    public let dataSet: DataSet
    public let labels: Tensor
    
    public init() {
        let weights = Tensor(Constant.weights)
        let biases = Tensor(Constant.biases)
        let layer = Layer(kind: .fullyConnected,
                          dataType: .float32,
                          inputFeatureChannelCount: Constant.inputFeatureChannels,
                          outputFeatureChannelCount: Constant.outputFeatureChannels,
                          weights: weights,
                          biases: biases)
        graph = Graph(kind: .sequential,
                          dataType: .float32,
                      inputTensor: Tensor(shape: [Constant.outputFeatureChannels, Constant.inputFeatureChannels], dataType: .float32),
                          layers: [layer],
                          featureChannelPosition: .notApplicable)

        let inputTensor = Tensor([[Float]]([[2, 2]]))
        dataSet = DataSet(inputTensor: inputTensor, batchSize: 1)
        
        labels = Tensor([[Float]]([[5]]))
    }
}
