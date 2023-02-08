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

import XCTest
import Distributed
import MimicComputeEngine
import MimicTransferables

final class ComputeEngineServiceTests: XCTestCase {
    
    var actorSystem = LocalTestingDistributedActorSystem()
    var service: ComputeEngineService!
    
    override func setUpWithError() throws {
        service = ComputeEngineService(actorSystem: actorSystem)
    }
    
    func testInferenceExecution() async throws {
        let model = ArithmeticModel()
        try await runExecutionTest(for: model, kind: .inference, device: .gpu, testHandler: { batch, outputs in
            XCTAssertEqual(outputs.count, 1)
            let result = try XCTUnwrap(outputs.first)
            assertEqual(resultTensor: result, expectedVector: model.expectedVector(at: batch))
        })
    }
    
    func testTrainingExecution() async throws {
        let model = LinearModel()
        let kind = Session.Kind.training(lossFunction: .meanSquaredError,
                                         optimizer: .rootMeanSquare(learningRate: 0.01))
        try await runExecutionTest(for: model, kind: kind, device: .cpu) { _, outputs in
            XCTAssertEqual(outputs.count, 1)
            let result = try XCTUnwrap(outputs.first)
            XCTAssertEqual(result.shape, [2, 1])
        }
    }
    
    func runExecutionTest(for model: TestModel, kind: Session.Kind, device: DeviceType, testHandler: (Int, [Tensor]) throws -> Void) async throws {
        let session = try await service.makeSession(kind: kind,
                                                    graphs: model.graphs,
                                                    dataSet: model.dataSet)
        try await session.compile(device: device)
        var batch = 0
        while let outputs = try await session.executeNext() {
            try testHandler(batch, outputs)
            batch += 1
        }
        XCTAssertEqual(batch, model.dataSet.batchCount)
    }
}
