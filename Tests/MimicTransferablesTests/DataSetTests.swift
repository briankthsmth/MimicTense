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
//  Created by Brian Smith on 6/14/22.
//

import XCTest
import MimicTransferables

class DataSetTests: XCTestCase {
    
    func testLoopOnSingleInputTensor() throws {
        let inputTensor = Tensor(Float(8))
        runLoop(inputTensors: [[inputTensor]], batchSize: 1) { batchInputTensors, batchIndex in
            XCTAssertEqual(batchInputTensors.count, 1)
            XCTAssertEqual(batchInputTensors[0], inputTensor)
        }
    }
    
    func testLoopWithMultipleInputTensors() throws {
        let inputTensors = [
            [Tensor([Float]([3, 2])), Tensor([Float]([4, 7]))],
            [Tensor([Float]([8, 9])), Tensor([Float]([10, 11]))]
        ]
        runLoop(inputTensors: inputTensors, batchSize: 1) { batchInputTensors, batchIndex in
            XCTAssertEqual(batchInputTensors.count, inputTensors[batchIndex].count)
            XCTAssertEqual(batchInputTensors[0], Tensor(shape: [2], dataType: .float32).appended(inputTensors[0][batchIndex]))
            XCTAssertEqual(batchInputTensors[1], Tensor(shape: [2], dataType:.float32).appended(inputTensors[1][batchIndex]))
        }
    }
    
    func testLoopWithRank4TensorSingleBatch() throws {
        let rank4Array = [[[[Float]]]]([[
            [[1, 2, 3], [4, 5, 6]]
        ]])
        let inputTensor = Tensor(rank4Array)
        runLoop(inputTensors: [[inputTensor]], batchSize: 1) { batchInputTensors, batchIndex in
            XCTAssertEqual(batchInputTensors.count, 1)
            XCTAssertEqual(batchInputTensors[0], inputTensor)
        }
    }
    
    func testLoopWithRank4TensorMultipleBatchs() throws {
        let rank4Array = [[[[Float]]]]([
            [[[1]]],
            [[[2]]],
            [[[3]]],
            [[[4]]],
            [[[5]]]
        ])
        let inputTensor = Tensor(rank4Array)
        var finalBatchIndex = -1
        runLoop(inputTensors: [[inputTensor]], batchSize: 2) { batchInputTensors, batchIndex in
            XCTAssertEqual(batchInputTensors.count, 1)
            XCTAssertEqual(batchInputTensors[0].shape[0], 2)
            finalBatchIndex = batchIndex
        }
        XCTAssertEqual(finalBatchIndex, 1)
    }
    
    func testMakeBatch() throws {
        let inputTensors = [
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

        let dataSet = DataSet(inputTensors: inputTensors, batchSize: 2)
        let batchTensors = dataSet.makeBatch(at: 1)
        let expectedTensors = [dataSet.tensors[0][2...3], dataSet.tensors[1][2...3]]
        
        XCTAssertEqual(batchTensors, expectedTensors)
    }
    
    func runLoop(inputTensors: [[Tensor]], batchSize: Int, testBody: ([Tensor], Int) -> Void) {
        var loopCount = 0
        let dataSet = DataSet(inputTensors: inputTensors, batchSize: batchSize)
            .forEachBatch { tensors in
                testBody(tensors, loopCount)
                loopCount += 1
            }
        XCTAssertEqual(loopCount, dataSet.tensors[0].endIndex / batchSize)
    }
}
