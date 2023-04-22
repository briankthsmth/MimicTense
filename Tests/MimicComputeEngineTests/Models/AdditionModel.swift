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
//  Created by Brian Smith on 6/21/22.
//

import Foundation
import MimicTransferables

struct AdditionModel: TestModel {
    struct Constant {
        static let dataType: DataType = .float32
        static let inputFeatureChannels: Int = 3
        static let outputFeatureChannels: Int = 3
        static let inputs: [[[Float]]] = [
            [[1, 2, 3], [4, 5, 6], [13, 14, 15]],
            [[7, 8, 9], [10, 11, 12], [16, 17, 18]]
        ]
        static let batchSize = 1
    }
    
    let graph: Graph
    let dataSet: DataSet
    let labels: Tensor
    
    init() {
        let layer = Layer(kind: .arithmetic, dataType: Constant.dataType, arithmeticOperation: .add)
        let inputShape = [Constant.batchSize , Constant.inputFeatureChannels]
        graph = Graph(
            kind: .sequential,
            dataType: Constant.dataType,
            inputTensors: [
                [Tensor(shape: inputShape, dataType: Constant.dataType, featureChannelPosition: .notApplicable),
                 Tensor(shape: inputShape, dataType: Constant.dataType, featureChannelPosition: .notApplicable)]
            ],
            layers: [layer],
            featureChannelPosition: .notApplicable)
        
        dataSet = DataSet(inputTensors: Constant.inputs.map { Tensor($0) },
                          batchSize: Constant.batchSize)
        labels = Tensor(zip(Constant.inputs[0], Constant.inputs[1]).map {
            zip($0.0, $0.1).reduce(into: []) { $0.append($1.0 + $1.1) }
        })
    }
}
