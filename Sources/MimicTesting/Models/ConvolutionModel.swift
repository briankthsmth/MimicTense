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
//  Created by Brian Smith on 4/20/23.
//

import Foundation
import MimicTransferables

public struct ConvolutionModel: TestModel {
    public struct Constant {
        public static let batchSize: Int = 1
        public static let dataTensorWithFeatureChannelLast = Tensor([[[[Float]]]]([[
            [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]],
            [[0.4, 0.4, 0.4], [0.5, 0.5, 0.5], [0.6, 0.6, 0.6]]
        ]]))
        public static let labelsForFeatureChannelLast = Tensor([[[[Float]]]]([[
            [[0.3], [0.6], [0.9]],
            [[1.2], [1.5], [1.8]]
        ]]))
        public static let dataTensorWithFeatureChannelFirst = Tensor([[[[Float]]]]([[
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]
            ],
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]
            ],
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]
            ],
        ]]), featureChannelPosition: .first)
        public static let labelsForFeatureChannelFirst = Tensor([[[[Float]]]]([[
            [
                [0.3, 0.6, 0.9],
                [1.2, 1.5, 1.8]
            ]
        ]]))
    }
    
    public let graph: Graph
    public let dataSet: DataSet
    public let labels: Tensor
    
    public init(featureChannelPosition: FeatureChannelPosition) {
        let input = featureChannelPosition == .first ? Constant.dataTensorWithFeatureChannelFirst :
        Constant.dataTensorWithFeatureChannelLast
        dataSet = DataSet(inputTensor: input, batchSize: Constant.batchSize)
        
        labels = featureChannelPosition == .first ? Constant.labelsForFeatureChannelFirst :
        Constant.labelsForFeatureChannelLast
        
        let weightsTensor = Tensor(Float(1))
        let layer = Layer(kind: .convolution,
                          dataType: input.dataType,
                          kernelSize: Layer.KernelSize(height: 1, width: 1),
                          inputFeatureChannelCount: input.featureChannelCount,
                          outputFeatureChannelCount: 1,
                          weights: weightsTensor)
        graph = Graph(kind: .sequential,
                      dataType: input.dataType,
                      inputTensor: Tensor(shape: input.shape,
                                          dataType: input.dataType,
                                          featureChannelPosition: input.featureChannelPosition),
                      layers: [layer],
                      featureChannelPosition: input.featureChannelPosition)
    }
}
