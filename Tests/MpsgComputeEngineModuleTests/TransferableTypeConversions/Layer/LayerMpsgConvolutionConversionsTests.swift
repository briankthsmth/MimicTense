//
//  LayerMpsgConvolutionConversionsTests.swift
//  
//
//  Created by Brian Smith on 7/25/23.
//

import XCTest
import MetalPerformanceShadersGraph

import MimicTesting
import MimicTransferables
import MimicComputeEngineModule
@testable import MpsgComputeEngineModule

final class LayerMpsgConvolutionConversionsTests: XCTestCase {
    var testRunner: MpsgGraphTestRunner!
    
    override func setUpWithError() throws {
        testRunner = try MpsgGraphTestRunner()
    }
    
    func testConvertConvolutionLayer() throws {
        let layer = ConvolutionLayerTestData.layer

        let outputTensor = try testRunner.run(inputs: [ConvolutionLayerTestData.input])
        { inputs, device, graph in
            let tensors = try layer.addConvolutionLayer(
                to: graph,
                device: MPSGraphDevice(mtlDevice: device),
                inputs: inputs)
            return (tensors.output, tensors.weightsPair)
        }
        
        XCTAssertEqual(outputTensor, ConvolutionLayerTestData.output)
    }
    
    func testConvolutionLayerInputsErrors() throws {
        XCTAssertThrowsError(try ConvolutionLayerTestData.layer.addConvolutionLayer(
            to: testRunner.graph,
            device: MPSGraphDevice(mtlDevice: testRunner.device),
            inputs: [])
        ) {
            XCTAssertNotNil($0 as? ComputeEngineLayerInputsError, "Error is not ComputeEngineLayerInputsError")
        }
    }
}
