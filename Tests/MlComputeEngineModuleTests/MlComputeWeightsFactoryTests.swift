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
//
//  Created by Brian Smith on 10/5/22.
//

import XCTest
import MLCompute
import MimicTransferables
import MimicComputeEngineModule
@testable import MlComputeEngineModule

final class MlComputeWeightsFactoryTests: XCTestCase {
    let shape = [1, 3, 1]
    var weightsDescriptor: MLCTensorDescriptor!
    
    override func setUpWithError() throws {
        weightsDescriptor = try XCTUnwrap(MLCTensorDescriptor(shape: shape,
                                                              dataType: .float32))
    }
    
    func testWithFillDataWeights() throws {
        let fillDataWeightTensor = Tensor(Float(4))
        let tensorFactory = MlComputeWeightsFactory(weightsTensor: fillDataWeightTensor,
                                                    weightsDescriptor: weightsDescriptor)
        let platformTensor = try tensorFactory()
        
        XCTAssertEqual(platformTensor.descriptor, weightsDescriptor)
        XCTAssertEqual(platformTensor.copyToArray(), [Float](arrayLiteral: 4, 4, 4))
    }
    
    func testWithUniformRandomInitializerWeights() throws {
        let randomDescriptor = RandomDescriptor(type: .uniform, range: Float(-1)..<Float(1))
        let randomWeightTensor = Tensor(shape: shape,
                                        dataType: .float32,
                                        randomDescriptor: randomDescriptor)
        let tensorFactory = MlComputeWeightsFactory(weightsTensor: randomWeightTensor,
                                                    weightsDescriptor: weightsDescriptor)
        let platformTensor = try tensorFactory()
        let weightsVector: [Float] = platformTensor.copyToArray()
        
        XCTAssertEqual(weightsVector.count, 3)
        try XCTSkipIf(weightsVector.count != 3)
        let notEqual = notEqual(weightsVector[0], weightsVector[1]) +
        notEqual(weightsVector[0], weightsVector[2]) +
        notEqual(weightsVector[1], weightsVector[2])
        XCTAssertGreaterThanOrEqual(notEqual, 2)
    }
    
    func testWithExistingWeights() throws {
        let weightsVector = [Float](arrayLiteral: 1.2, 0.4, -1.1)
        let weights = Tensor([[Float]]([weightsVector]))
        
        let tensorFactory = MlComputeWeightsFactory(weightsTensor: weights,
                                                    weightsDescriptor: weightsDescriptor)
        let platformTensor = try tensorFactory()
        
        XCTAssertEqual(platformTensor.copyToArray(), weightsVector)
    }
    
    func testInvalidWeightDataException() throws {
        let weightsVector = [Float](arrayLiteral: 0.3, 0.5, 0.6, 0.7)
        let weights = Tensor([weightsVector, weightsVector])
        
        let tensorFactory = MlComputeWeightsFactory(weightsTensor: weights,
                                                    weightsDescriptor: weightsDescriptor)
        XCTAssertThrowsError(try tensorFactory()) {
            XCTAssertEqual($0 as! ComputeEngineError, ComputeEngineError.invalidWeights)
        }
    }
}

func notEqual(_ x1: Float, _ x2: Float) -> Int {
    abs(x1 - x2) > 0.001 ? 1 : 0
}
