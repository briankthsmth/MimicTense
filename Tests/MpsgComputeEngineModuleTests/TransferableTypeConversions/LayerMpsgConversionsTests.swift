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
import MetalPerformanceShadersGraph
import MimicTesting
import MimicTransferables
@testable import MpsgComputeEngineModule

final class LayerMpsgConversionsTests: XCTestCase {
    struct Constant {
        struct Addition {
            static let firstOperand: Float = 4
            static let secondOperand: Float = 3
        }
    }
    
    enum TestError: Error {
        case noOutputTensorData
    }
    
    var device: MTLDevice!
    var graph: MPSGraph!
    
    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        graph = MPSGraph()
    }
    
    func testConvertAdditionLayer() throws {
        let layer = Layer(kind: .arithmetic, dataType: .float32, arithmeticOperation: .add)
        let firstOperand = Tensor(Constant.Addition.firstOperand)
        let secondOperand = Tensor(Constant.Addition.secondOperand)
        let outputTensor = try run(inputs: [firstOperand, secondOperand]) {
            let output = try layer.addAdditionLayer(to: graph, inputs: $0)
            return (output, nil)
        }
        
        XCTAssertEqual(outputTensor, Tensor([Constant.Addition.firstOperand + Constant.Addition.secondOperand]))
    }
    
    func testAdditionLayerInputsErrors() throws {
        let layer = Layer(kind: .arithmetic, dataType: .float32, arithmeticOperation: .add)
        XCTAssertThrowsError(try layer.addAdditionLayer(to: graph, inputs: []))
        
        let singleOperand = graph.placeholder(shape: [], name: nil)
        XCTAssertThrowsError(try layer.addAdditionLayer(to: graph, inputs: [singleOperand]))
    }
    
    func testConvertConvolutionLayer() throws {
        let layer = ConvolutionLayerTestData.layer

        let outputTensor = try run(inputs: [ConvolutionLayerTestData.input]) { inputs in
            let tensors = try layer.addConvolutionLayer(to: graph, device: MPSGraphDevice(mtlDevice: device), inputs: inputs)
            return (tensors.output, tensors.weightsPair)
        }
        
        XCTAssertEqual(outputTensor, ConvolutionLayerTestData.output)
    }
    
    func testConvertFullConnectedLayer() throws {
        
    }
    
    func run(inputs: [Tensor], layerFactory: ([MPSGraphTensor]) throws -> (output: MPSGraphTensor, weightsPair: Layer.TensorPair?)) throws -> Tensor {
        let graphDevice = try XCTUnwrap(MPSGraphDevice(mtlDevice: device))
        
        let mpsgInputs = inputs.map { $0.makeMpsgTensor(for: graph) }
        let tensors = try layerFactory(mpsgInputs)
        
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
        try inputs.enumerated().forEach {
            feeds[mpsgInputs[$0.offset]] = try $0.element.makeMpsgTensorData(for: graphDevice)
        }
        if let weightsPair = tensors.weightsPair {
            feeds[weightsPair.placeholder] = weightsPair.data
        }
        
        
        let commandQueue = try XCTUnwrap(device.makeCommandQueue())
        let result = graph.run(with: commandQueue,
                               feeds: feeds,
                               targetTensors: [tensors.output],
                               targetOperations: nil)
        
        guard let tensorData = result[tensors.output] else { throw  TestError.noOutputTensorData }
        
        return try tensorData.makeTensor()
    }
}
