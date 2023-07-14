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
    var device: MTLDevice!
    var graph: MPSGraph!
    
    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        graph = MPSGraph()
    }
    
    func testConvertAdditionLayer() throws {
        let layer = Layer(kind: .arithmetic, dataType: .float32, arithmeticOperation: .add)
        let firstOperand = graph.placeholder(shape: [], dataType: .float32, name: "firstOperand")
        let secondOperand = graph.placeholder(shape: [], dataType: .float32, name: "secondOperand")
        let outputTensor = try layer.addAdditionLayer(to: graph, inputs: [firstOperand, secondOperand])
        
        XCTAssertEqual(outputTensor.shape, [])
        
        let graphDevice = MPSGraphDevice(mtlDevice: device)
        let firstOperandArray = [Float](arrayLiteral: Constant.Addition.firstOperand)
        let firstOperandData = MPSGraphTensorData(device: graphDevice,
                                                  data: Data(bytes: firstOperandArray, count: firstOperandArray.count * Float.memoryLayoutSize),
                                                  shape: [],
                                                  dataType: .float32)
        let secondOperandArray = [Float](arrayLiteral: Constant.Addition.secondOperand)
        let secondOperandData = MPSGraphTensorData(device: graphDevice,
                                                   data: Data(bytes: secondOperandArray, count: secondOperandArray.count * Float.memoryLayoutSize),
                                                   shape: [],
                                                   dataType: .float32)
        let commandQueue = try XCTUnwrap(device.makeCommandQueue())
        let result = graph.run(with: commandQueue,
                               feeds:
                                [
                                    firstOperand : firstOperandData,
                                    secondOperand : secondOperandData
                                ],
                               targetTensors: [outputTensor],
                               targetOperations: nil)
        let resultOutputData = try XCTUnwrap(result[outputTensor])
        var resultValue: Float = 0
        resultOutputData.mpsndarray().readBytes(&resultValue, strideBytes: nil)
        
        XCTAssertEqual(resultValue, Constant.Addition.firstOperand + Constant.Addition.secondOperand)
    }
    
    func testAdditionLayerInputsErrors() throws {
        let layer = Layer(kind: .arithmetic, dataType: .float32, arithmeticOperation: .add)
        XCTAssertThrowsError(try layer.addAdditionLayer(to: graph, inputs: []))
        
        let singleOperand = graph.placeholder(shape: [], name: nil)
        XCTAssertThrowsError(try layer.addAdditionLayer(to: graph, inputs: [singleOperand]))
    }
    
    func testConvertConvolutionLayer() throws {
        let layer = ConvolutionLayerTestData.layer

        let outputTensor = try run(inputs: []) { inputs in
            try layer.addConvolutionLayer(to: graph, inputs: inputs)
        }
        
        XCTAssertEqual(outputTensor.shape, [7,14,4])
    }
    
    func testConvertFullConnectedLayer() throws {
        
    }
    
    func run(inputs: [Tensor], layerFactory: ([MPSGraphTensor]) throws -> MPSGraphTensor) throws -> Tensor {
        let graphDevice = try XCTUnwrap(MPSGraphDevice(mtlDevice: device))
        
        let mpsgInputs = inputs.map { $0.makeMpsgTensor(for: graph) }
        let mpsgOutputTensor = try layerFactory(mpsgInputs)
        
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
        try inputs.enumerated().forEach {
            feeds[mpsgInputs[$0.offset]] = try $0.element.makeMpsgTensorData(for: graphDevice)
        }
        
        let commandQueue = try XCTUnwrap(device.makeCommandQueue())
        let result = graph.run(with: commandQueue,
                               feeds: feeds,
                               targetTensors: [mpsgOutputTensor],
                               targetOperations: nil)
        
        return Tensor(shape: [], dataType: .float32)
    }
}
