//
//  PlatformExecutionGraphableTests.swift
//  
//
//  Created by Brian Smith on 1/31/23.
//

import XCTest
import MimicTransferables
@testable import MimicComputeEngine

final class PlatformExecutionGraphableTests: XCTestCase {
    struct MockExecutionGraph: PlatformExecutionGraphable {}
    
    func testMakePlatformGraphs() throws {
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
        let product = try MockExecutionGraph.makePlatformGraphs(from: [graph])
        
        XCTAssertEqual(product.graphs.count, 1)
        XCTAssertEqual(product.graphs.first?.layers.count, 2)
    }
}
