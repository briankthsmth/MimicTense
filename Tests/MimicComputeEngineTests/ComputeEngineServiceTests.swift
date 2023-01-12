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
    
    let model = ArithmeticModel()
    var actorSystem = LocalTestingDistributedActorSystem()
    var service: ComputeEngineService!
    
    override func setUpWithError() throws {
        service = ComputeEngineService(actorSystem: actorSystem)
    }
    
    func testExecuteNext() async throws {
        let session = try await service.makeSession(kind: .inference,
                                                    graphs: model.graphs,
                                                    dataSet: model.multiBatchDataSet)
        try await session.compile(device: .gpu)
        var resultIndex = 0
        while let outputs = try await session.executeNext() {
            XCTAssertEqual(outputs.count, 1)
            try XCTSkipUnless(outputs.count > 0, "Can not continue testing with empty array.")
            assertEqual(resultTensor: outputs[0], expectedVector: model.expectedVector(at: resultIndex))
            resultIndex += 1
        }
        XCTAssertEqual(resultIndex, model.multiBatchDataSet.batchCount)
    }
}
