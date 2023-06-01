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
//  Created by Brian Smith on 7/15/22.
//

import XCTest
import MLCompute
import MimicTransferables
@testable import MlComputeEngineModule

final class FullyConnectedLayerConversionTests: XCTestCase {

    func testConversionWithoutWeights() throws {
        let layer = Layer(kind: .fullyConnected, dataType: .float32)
        let mlcLayer = try layer.makeMlcFullyConnectedLayer()
        XCTAssertNil(mlcLayer)
    }
    
    func testConversion() throws {
        let weightsVector = [Float]([1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7])
        let weightsData = weightsVector.makeBuffer()
        let weights = Tensor(shape: [2, 7], data: weightsData, dataType: .float32)
        let biasesVector = [Float]([0.5, 0.5])
        let biasesData = biasesVector.makeBuffer()
        let biases = Tensor(shape: [2], data: biasesData, dataType: .float32)
        let layer = Layer(kind: .fullyConnected,
                          dataType: .float32,
                          inputFeatureChannelCount: 7,
                          outputFeatureChannelCount: 2,
                          weights: weights,
                          biases: biases)
        let mlcLayer = try XCTUnwrap(layer.makeMlComputeLayer() as? MLCFullyConnectedLayer)
        XCTAssertEqual(mlcLayer.descriptor.inputFeatureChannelCount, 7)
        XCTAssertEqual(mlcLayer.descriptor.outputFeatureChannelCount, 2)
        XCTAssertEqual(mlcLayer.descriptor.kernelSizes.height, 7)
        XCTAssertEqual(mlcLayer.descriptor.kernelSizes.width, 2)
        XCTAssertEqual(mlcLayer.weights.descriptor.shape, [1] + weights.shape)
        XCTAssertEqual(mlcLayer.weights.copyDataToBuffer(), weightsData)
        XCTAssertEqual(mlcLayer.biases?.descriptor.shape, [1] + biases.shape)
        XCTAssertEqual(mlcLayer.biases?.copyDataToBuffer(), biasesData)
    }
    
    func testWithFillDataWeights() throws {
        let weightsVector = [Float]([3])
        let weightsData = weightsVector.makeBuffer()
        let weights = Tensor(shape: [], data: weightsData, dataType: .float32)
        let layer = Layer(kind: .fullyConnected,
                          dataType: .float32,
                          inputFeatureChannelCount: 7,
                          outputFeatureChannelCount: 2,
                          weights: weights)
        let mlcLayer = try XCTUnwrap(layer.makeMlComputeLayer() as? MLCFullyConnectedLayer)
        XCTAssertEqual(mlcLayer.weights.descriptor.shape, [1, 2, 7])
        XCTAssertEqual(mlcLayer.weights.copyToArray().first, weightsVector.first)
    }
    
    func testWithRandomWeightsInitializer() throws {
        let randomDescriptor = RandomDescriptor(type: .uniform, range: Float(-1)..<Float(1))
        let weights = Tensor(shape: [2, 7], dataType: .float32, randomDescriptor: randomDescriptor)
        let layer = Layer(kind: .fullyConnected,
                          dataType: .float32,
                          inputFeatureChannelCount: 7,
                          outputFeatureChannelCount: 2,
                          weights: weights)
        let mlcLayer = try XCTUnwrap(layer.makeMlComputeLayer() as? MLCFullyConnectedLayer)
        XCTAssertEqual(mlcLayer.weights.descriptor.shape, [1, 2, 7])
        XCTAssertEqual(mlcLayer.weights.data?.count, weights.shapeByteCount)
    }
}
