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
//
//  Created by Brian Smith on 9/26/22.
//

import XCTest
@testable import MimicComputeEngine
import MimicTransferables

final class MlComputeTrainingGraphTests: XCTestCase {
    struct Constant {
        static let learningRate: Float = 0.01
    }
    
    let model = LinearModel()
    var trainingGraph: MlComputeTrainingGraph!
    
    override func setUpWithError() throws {
        trainingGraph = try MlComputeTrainingGraph(graphs: model.graphs,
                                                   lossLabelTensors: [Tensor(shape: [
                                                    LinearModel.Constant.batchSize,
                                                    LinearModel.Constant.outputChannels
                                                   ],
                                                                             dataType: .float32)],
                                                   lossFunction: .meanSquaredError,
                                                   optimizer: .rootMeanSquare(learningRate: Constant.learningRate))
        
        try trainingGraph.compile(device: .cpu)
    }
    
    func testRetrieveTrainedLayer() async throws {
        try await train()
        let layerLabel = try XCTUnwrap(model.graphs.first?.layers.first?.label)
        let layer = try trainingGraph.retrieveLayer(by: layerLabel)
        let weights = try XCTUnwrap(layer.weights?.extract(Float.self) as? [[Float]])
        let biases = try XCTUnwrap(layer.biases?.extract(Float.self) as? [Float])
        XCTAssertEqual(weights.first?.first ?? 0, LinearModel.Constant.slope, accuracy: 0.01)
        XCTAssertEqual(biases.first ?? 0, LinearModel.Constant.intercept, accuracy: 0.01)
    }
        
    func testRetrieveGraphs() async throws {
        try await train()
        
        let graphs = try trainingGraph.retrieveGraphs()
        XCTAssertEqual(graphs.count, model.graphs.count)
        let layers = try XCTUnwrap(graphs.first?.layers)
        let modelLayers = try XCTUnwrap(model.graphs.first?.layers)
        XCTAssertEqual(layers.count, modelLayers.count)
        let layer = try XCTUnwrap(layers.first)
        let modelLayer = try XCTUnwrap(modelLayers.first)
        let weights = try XCTUnwrap(layer.weights?.extract(Float.self) as? [[Float]])
        let biases = try XCTUnwrap(layer.biases?.extract(Float.self) as? [Float])
        XCTAssertEqual(layer.kind, modelLayer.kind)
        XCTAssertEqual(weights.first?.first ?? 0, LinearModel.Constant.slope, accuracy: 0.01)
        XCTAssertEqual(biases.first ?? 0, LinearModel.Constant.intercept, accuracy: 0.01)
    }
    
    func train() async throws {
        for _ in 0 ..< 20 {
            for batch in 0 ..< model.dataSet.batchCount {
                let batchSamples = model.dataSet.makeBatch(at: batch)
                let batchLabels = try XCTUnwrap(model.dataSet.makeBatchLabels(at: batch))
                let _ = try await trainingGraph.execute(inputs: batchSamples,
                                                        lossLables: batchLabels,
                                                        batchSize: LinearModel.Constant.batchSize)
            }
        }
    }
}
