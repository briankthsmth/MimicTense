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
//  Created by Brian Smith on 6/15/22.
//

import XCTest
import MimicTransferables
@testable import MimicComputeEngine

class ArthmeticLayerInferenceGraphTests: XCTestCase {
    let model = ArithmeticModel()
    
    func testExecute() async throws {
        let inferencGraph = try MlComputeInferenceGraph(graphs: [model.graph])
        try inferencGraph.compile(device: .gpu)
        let results = try await inferencGraph.execute(inputs: model.inputs(at: 0),
                                                       batchSize: ArithmeticModel.Constant.batchSize)
        let expectedVector = model.expectedVector(at: 0)
        
        XCTAssertEqual(results[0].shape[1], expectedVector.count)
        XCTAssertEqual(results[0].dataType, ArithmeticModel.Constant.dataType)
        assertEqual(resultTensor: results[0], expectedVector: expectedVector)
    }
}
