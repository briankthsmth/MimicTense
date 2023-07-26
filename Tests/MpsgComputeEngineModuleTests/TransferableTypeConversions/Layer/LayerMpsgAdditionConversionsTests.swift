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
//
//  Created by Brian Smith on 6/2/23.
//

import XCTest

import MimicTesting
import MimicTransferables
@testable import MpsgComputeEngineModule

final class LayerMpsgAdditionConversionsTests: XCTestCase {
    struct Constant {
        struct Addition {
            static let firstOperand: Float = 4
            static let secondOperand: Float = 3
        }
    }
    
    var testRunner: MpsgGraphTestRunner!
    
    override func setUpWithError() throws {
        testRunner = try MpsgGraphTestRunner()
    }
    
    func testConvertAdditionLayer() throws {
        let layer = Layer(kind: .arithmetic, dataType: .float32, arithmeticOperation: .add)
        let firstOperand = Tensor(Constant.Addition.firstOperand)
        let secondOperand = Tensor(Constant.Addition.secondOperand)
        let outputTensor = try testRunner.run(inputs: [firstOperand, secondOperand]) {
            let output = try layer.addAdditionLayer(to: $2, inputs: $0)
            return (output, nil)
        }
        
        XCTAssertEqual(outputTensor, Tensor([Constant.Addition.firstOperand + Constant.Addition.secondOperand]))
    }
    
    func testAdditionLayerInputsErrors() throws {
        let layer = Layer(kind: .arithmetic, dataType: .float32, arithmeticOperation: .add)
        XCTAssertThrowsError(try layer.addAdditionLayer(to: testRunner.graph, inputs: []))
        
        let singleOperand = testRunner.graph.placeholder(shape: [], name: nil)
        XCTAssertThrowsError(try layer.addAdditionLayer(to: testRunner.graph, inputs: [singleOperand]))
    }    
}
