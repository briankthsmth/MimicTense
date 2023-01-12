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
//  Created by Brian Smith on 6/10/22.
//

import XCTest
import MLCompute
import MimicTransferables
@testable import MimicComputeEngine

class ConvolutionLayerConversionTests: XCTestCase {
    struct Constant {
        struct LayerWithFill {
            static let kernelSizeWidth = 2
            static let kernelSizeHeight = 3
            static let inputFeatureChannelCount = 3
            static let outputFeatureChannelCount = 64
            static let weightTensor = Tensor(Float(1))
            static let dataType = DataType.float32
            
            static var weightsCount: Int {
                kernelSizeWidth * kernelSizeHeight * inputFeatureChannelCount * outputFeatureChannelCount
            }
            
            static var weightsShape: [Int] {
                [1, inputFeatureChannelCount * outputFeatureChannelCount, kernelSizeHeight, kernelSizeWidth]
            }
        }
    }
    
    var mlcConvolutionLayer: MLCConvolutionLayer?
    
    override func setUpWithError() throws {
        let convolutionLayer = Layer(kind: .convolution,
                                     dataType: Constant.LayerWithFill.dataType,
                                     kernelSize: Layer.KernelSize(height: Constant.LayerWithFill.kernelSizeHeight,
                                                                  width: Constant.LayerWithFill.kernelSizeWidth),
                                     inputFeatureChannelCount: Constant.LayerWithFill.inputFeatureChannelCount,
                                     outputFeatureChannelCount: Constant.LayerWithFill.outputFeatureChannelCount,
                                     weights: Constant.LayerWithFill.weightTensor)
        mlcConvolutionLayer = try convolutionLayer.makeMlComputeLayer() as? MLCConvolutionLayer
    }
    
    func testWeightsWithFill() throws {
        let layer = try XCTUnwrap(mlcConvolutionLayer)
        var weightsData = [Float](repeating: Constant.LayerWithFill.weightTensor.extractScalar() ?? 1,
                                count: Constant.LayerWithFill.weightsCount)
        let tensorData = MLCTensorData(bytesNoCopy: &weightsData,
                                       length: weightsData.count * MemoryLayout<Float>.size)
        let expectedTensor = MLCTensor(shape: Constant.LayerWithFill.weightsShape,
                                   data: tensorData,
                                   dataType: .float32)
        XCTAssertTrue(layer.weights.compare(to: expectedTensor),
                      "\(layer.weights) is not equal to \(expectedTensor)")
    }
    
    func testKernelSize() throws {
        let layer = try XCTUnwrap(mlcConvolutionLayer)
        XCTAssertEqual(layer.descriptor.kernelSizes.width, Constant.LayerWithFill.kernelSizeWidth)
        XCTAssertEqual(layer.descriptor.kernelSizes.height, Constant.LayerWithFill.kernelSizeHeight)
    }
    
    func testFeatureChannels() throws {
        let layer = try XCTUnwrap(mlcConvolutionLayer)
        XCTAssertEqual(layer.descriptor.inputFeatureChannelCount, Constant.LayerWithFill.inputFeatureChannelCount)
        XCTAssertEqual(layer.descriptor.outputFeatureChannelCount, Constant.LayerWithFill.outputFeatureChannelCount)
    }
}
