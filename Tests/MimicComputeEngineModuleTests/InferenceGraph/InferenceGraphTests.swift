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
//  Created by Brian Smith on 4/14/23.
//

import XCTest
import Collections
import MimicTransferables
import MimicTesting
@testable import MimicComputeEngineModule
import MlComputeEngineModule
import MpsgComputeEngineModule

final class InferenceGraphTests: XCTestCase {
    struct Constant {
        static let floatAccuracy: Float = 0.001
    }
    
    typealias Factory = (Graph) throws -> InferenceGraphable
    
    struct Configuration {
        let name: String
        let device: DeviceType
        let factory: Factory
    }
    
    static var configurations: Deque<Configuration> = [
        Configuration(name: "MlComputeInferenceGraphTests",
                      device: .gpu,
                      factory: { graph in
                          try MlComputeInferenceGraph(graph: graph)
                      }),
        Configuration(name: "MpsgInferenceGraphTests",
                      device: .gpu,
                      factory: { graph in
                          try MpsgInferenceGraph(graph: graph)
                      })
    ]
    
    var configuration: Configuration!
    
    override class func setUp() {
        print("Testing configuration \(configurations.first?.name ?? "none").")
    }
    
    override class func tearDown() {
        configurations.removeFirst()
    }
    
    override func setUpWithError() throws {
        configuration = Self.configurations.first
    }
        
    func testAdditionModel() async throws {
        let model = AdditionModel()
        let inferencGraph = try configuration.factory(model.graph)
        try inferencGraph.compile(device: configuration.device)
        let result = try await inferencGraph.execute(inputs: model.dataSet.makeBatch(at: 0),
                                                       batchSize: AdditionModel.Constant.batchSize)

        try assertEqual(result, model.labels[0], accuracy: Constant.floatAccuracy)
    }
    
    func testFullyConnectedNodeAdditionModel() async throws {
        let model = FullyConnectedNodeAdditionModel()
        let inferenceGraph = try configuration.factory(model.graph)
        try inferenceGraph.compile(device: configuration.device)
        let result = try await inferenceGraph.execute(inputs: model.dataSet.makeBatch(at: 0),
                                                      batchSize: FullyConnectedNodeAdditionModel.Constant.batchSize)
        
        try assertEqual(result, model.labels[0], accuracy: Constant.floatAccuracy)
    }
    
    func testConvolutionModelWithFeatureChannelFirst() async throws {
        try await performConvolutionTest(model: ConvolutionModel(featureChannelPosition: .first))
    }
    
    func testConvolutionModelWithFeatureChannelLast() async throws {
        try await performConvolutionTest(model: ConvolutionModel(featureChannelPosition: .last))
    }

    private func performConvolutionTest(model: ConvolutionModel) async throws {
        let dataSet = model.dataSet
        
        let inferenceGraph = try configuration.factory(model.graph)
        try inferenceGraph.compile(device: configuration.device)
        let result = try await inferenceGraph.execute(inputs: dataSet.makeBatch(at: 0),
                                                      batchSize: ConvolutionModel.Constant.batchSize)
        
        try assertEqual(result, model.labels, accuracy: Constant.floatAccuracy)
    }

    override class var defaultTestSuite: XCTestSuite {
        let testSuite = XCTestSuite(name: "InferenceGraphTests")
        for _ in 0 ..< configurations.count {
            testSuite.addTest(XCTestSuite(forTestCaseClass: InferenceGraphTests.self))
        }
        return testSuite
    }
}
