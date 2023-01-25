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
//  Created by Brian Smith on 1/18/23.
//

import XCTest
import MimicTransferables

final class DataSetBatchTests: XCTestCase {
    let testTensors = [
        [
            Tensor([Float]([2, 1])), 
            Tensor([Float]([4, 3])),
            Tensor([Float]([6, 5])),
            Tensor([Float]([8, 7]))
        ],
        [
            Tensor([Float]([8, 9])),
            Tensor([Float]([10, 11])),
            Tensor([Float]([12, 13])),
            Tensor([Float]([14, 15]))
        ]
    ]
    var dataSet: DataSet!
    
    override func setUpWithError() throws {
        dataSet = DataSet(inputTensors: testTensors, labels: testTensors, batchSize: 2)
    
    }
    
    func testMakeBatch() throws {
        let batchTensors = dataSet.makeBatch(at: 1)
        let expectedTensors = [dataSet.tensors[0][2...3], dataSet.tensors[1][2...3]]
        
        XCTAssertEqual(batchTensors, expectedTensors)
    }
    
    func testMakeBatchWithSingleTensor() throws {
        let tensor = Tensor([[Float]]([
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
        ]))
        let dataSet = DataSet(inputTensor: tensor, batchSize: 2)
        let batchTensor = dataSet.makeBatch(at: 1)
        let expectedTensors = [tensor[2...3]]
        
        XCTAssertEqual(batchTensor, expectedTensors)
    }
    
    func testMakeBatchLabels() throws {
        let batchLabels = dataSet.makeBatchLabels(at: 0)
        let expectedLabels = [dataSet.tensors[0][0...1], dataSet.tensors[1][0...1]]
        
        XCTAssertEqual(batchLabels, expectedLabels)
    }
}
