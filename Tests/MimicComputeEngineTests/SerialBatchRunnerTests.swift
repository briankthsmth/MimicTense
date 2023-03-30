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
//  Created by Brian Smith on 7/26/22.
//

import XCTest
import MimicTransferables
@testable import MimicComputeEngine

final class SerialBatchRunnerTests: XCTestCase {
    final class TestOperation: Operational {
        func execute(batch: Int, dataSet: DataSet) async throws -> Tensor? {
            await withCheckedContinuation({ continuation in
                continuation.resume(returning: dataSet.makeBatch(at: batch).first)
            })
        }
        
        func compile(device: MimicTransferables.DeviceType) {}
        func execute(dataSet: MimicTransferables.DataSet) -> [MimicTransferables.Tensor] { [] }
        func retrieveGraph() throws -> MimicTransferables.Graph { throw ComputeEngineError.layerConversion }
    }
    
    func testNext() async throws {
        let inputTensors = [
            Tensor([[Float]]([
                [1],
                [2],
                [3]
            ]))
        ]
        let expectedTensors = [
            Tensor([[Float]]([[1]])),
            Tensor([[Float]]([[2]])),
            Tensor([[Float]]([[3]]))
        ]
        let dataSet = DataSet(inputTensors: inputTensors, batchSize: 1)
        let batchRunner = SerialBatchRunner(dataSet: dataSet, operation: TestOperation())
        var batchIndex = 0
        while let output = try await batchRunner.next() {
            XCTAssertEqual(output, expectedTensors[batchIndex])
            batchIndex += 1
        }
        XCTAssertEqual(batchIndex, 3)
    }
}
