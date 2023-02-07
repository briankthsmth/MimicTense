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

struct ArithmeticModel: TestModel {
    struct Constant {
        static let dataType: DataType = .float32
        static let vector1: [Float] = [3, 5, 10]
        static let vector2: [Float] = [4, 6, 11]
        static let inputs: [[[Float]]] = [
            [[1, 2, 3], [4, 5, 6], [13, 14, 15]],
            [[7, 8, 9], [10, 11, 12], [16, 17, 18]]
        ]
        static let batchSize = 1
    }
    
    let resultVector: [Float]
    let graphs: [Graph]
    let dataSet: DataSet
    let tensorData1: Tensor
    let tensorData2: Tensor
    
    init() {
        resultVector = zip(Constant.vector1, Constant.vector2).reduce(into: []) { $0.append($1.0 + $1.1) }
        
        let vector1 = Constant.vector1
        tensorData1 = Tensor(shape: [3],
                             data: vector1.makeBuffer(),
                             dataType: .float32,
                             featureChannelPosition: .notApplicable)
        
        let vector2 = Constant.vector2
        tensorData2 = Tensor(shape: [3],
                             data: vector2.makeBuffer(),
                             dataType: .float32,
                             featureChannelPosition: .notApplicable)
        
        let layer = Layer(kind: .arithmetic, dataType: tensorData1.dataType, arithmeticOperation: .add)
        
        graphs = [Graph(
            kind: .sequential,
            dataType: tensorData1.dataType,
            inputTensors: [
                [Tensor(shape: [1, Constant.vector1.count], dataType: tensorData1.dataType, featureChannelPosition: .notApplicable),
                 Tensor(shape: [1, Constant.vector2.count], dataType: tensorData2.dataType, featureChannelPosition: .notApplicable)]
            ],
            layers: [layer],
            featureChannelPosition: .notApplicable)]
        
        dataSet = DataSet(inputTensors: Constant.inputs.map { $0.map { Tensor($0) } },
                                    batchSize: Constant.batchSize)
    }
    
    func inputs(at index: Int) -> [Tensor] {
        [Tensor(Constant.inputs[0][index]), Tensor(Constant.inputs[1][index])]
    }
    
    func expectedVector(at index: Int) -> [Float] {
        zip(Constant.inputs[0][index], Constant.inputs[1][index])
            .reduce(into: []) { $0.append($1.0 + $1.1) }
    }
}
