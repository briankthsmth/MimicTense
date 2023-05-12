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
//  Created by Brian Smith on 1/31/23.
//

import XCTest
import MimicTransferables
@testable import MimicComputeEngine

final class MlComputeGraphTransformableTests: XCTestCase {
    struct MockExecutionGraph: MlComputeGraphTransformable {}
    
    func testMakePlatformGraph() throws {
        let graph = Graph(kind: .sequential,
                          dataType: .float32,
                          inputTensor: Tensor(shape: [1, 5], dataType: .float32),
                          layers: [
                            Layer(label: "layer1",
                                  kind: .fullyConnected,
                                  dataType: .float32,
                                  inputFeatureChannelCount: 5,
                                  outputFeatureChannelCount: 3,
                                  weights: Tensor(shape: [5, 3], dataType: .float32, randomInitializerType: .uniform),
                                  biases: Tensor(shape: [3],
                                                 data: [Float](arrayLiteral: 0.2, 0.1, 0.2).makeBuffer(),
                                                 dataType: .float32)),
                            Layer(label: "layer2",
                                  kind: .fullyConnected,
                                  dataType: .float32,
                                  inputFeatureChannelCount: 3,
                                  outputFeatureChannelCount: 1,
                                  weights: Tensor(shape: [3, 1], dataType: .float32, randomInitializerType: .uniform),
                                  biases: Tensor(shape: [1],
                                                 data: [Float](arrayLiteral: 0.4).makeBuffer(),
                                                 dataType: .float32))
                          ],
                          featureChannelPosition: .first)
        let product = try MockExecutionGraph.makePlatformGraph(from: graph)
        
        XCTAssertEqual(product.graph.layers.count, 2)
    }
}
