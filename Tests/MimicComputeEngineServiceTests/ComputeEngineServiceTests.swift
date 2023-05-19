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
import MimicComputeEngineService
import MimicTransferables
import MimicTesting


final class ComputeEngineServiceTests: XCTestCase {
    
    var actorSystem = LocalTestingDistributedActorSystem()
    var service: ComputeEngineService!
    
    override func setUpWithError() throws {
        service = ComputeEngineService()
    }
    
    func testInferenceExecution() async throws {
        let model = AdditionModel()
        try await runExecutionTest(for: model, kind: .inference, device: .gpu, testHandler: { batch, output in
            let result = try XCTUnwrap(output)
            try assertEqual(result, model.labels[batch], accuracy: Float(0.01))
        })
    }
    
    func testTrainingExecution() async throws {
        let model = LinearModel()
        let kind = Session.Kind.training(lossFunction: .meanSquaredError,
                                         optimizer: .rootMeanSquare(learningRate: 0.01))
        try await runExecutionTest(for: model, kind: kind, device: .cpu) { _, output in
            let result = try XCTUnwrap(output)
            XCTAssertEqual(result.shape, [2, 1])
        }
    }
    
    func runExecutionTest(for model: TestModel, kind: Session.Kind, device: DeviceType, testHandler: (Int, Tensor) throws -> Void) async throws {
        let session = try await service.makeSession(kind: kind,
                                                    graph: model.graph,
                                                    dataSet: model.dataSet)
        try await session.compile(device: device)
        var batch = 0
        while let output = try await session.executeNext() {
            try testHandler(batch, output)
            batch += 1
        }
        XCTAssertEqual(batch, model.dataSet.batchCount)
    }
}
