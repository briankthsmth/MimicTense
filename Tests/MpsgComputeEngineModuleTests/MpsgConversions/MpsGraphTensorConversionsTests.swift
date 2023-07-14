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
//  Created by Brian Smith on 6/29/23.
//

import XCTest
import MetalPerformanceShadersGraph

import MimicTransferables

@testable import MpsgComputeEngineModule

final class MpsGraphTensorConversionsTests: XCTestCase {
    var graph: MPSGraph!
    
    override func setUpWithError() throws {
        graph = MPSGraph()
    }
    
    func testMakeTensor() throws {
        let mpsgTensor = graph.placeholder(shape: [3, 5, 1], dataType: .float32, name: nil)
        let tensor = try mpsgTensor.makeTensor()
        
        XCTAssertEqual(tensor.shape, mpsgTensor.shape?.mapToInts())
        XCTAssertEqual(tensor.dataType, DataType(mpsgTensor.dataType))
    }
    
    func testMakeTensorWithoutShape() throws {
        let mpsgTensor = graph.placeholder(shape: nil, dataType: .float32, name: nil)
        XCTAssertThrowsError(try mpsgTensor.makeTensor())
    }
    
    func testMakeTensorWithUnsupportedType() throws {
        let mpsgTensor = graph.placeholder(shape: [1, 2], dataType: .bool, name: nil)
        XCTAssertThrowsError(try mpsgTensor.makeTensor())
    }
}
