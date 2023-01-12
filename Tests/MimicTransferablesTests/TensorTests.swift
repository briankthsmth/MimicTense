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
//  Created by Brian Smith on 5/23/22.
//

import XCTest
import MimicTransferables

class TensorTests: XCTestCase {
    let scalar: Float = 8
    var scalarTensor: Tensor!
    let vector: [Float] = [1, 2 , 3, 4]
    var vectorTensor: Tensor!
    let matrix: [[Float]] = [[1,2], [3, 4]]
    var matrixTensor: Tensor!
    let rank3Array: [[[Float]]] = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    var rank3Tensor: Tensor!
    
    let featureChannelLastArray = [[[[Float]]]]([
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]]
        ],
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]]
        ]
    ])
    var rank4TensorFeatureChannelLast: Tensor!
    
    let featureChannelFirstArray = [[[[Float]]]]([
        [
            [
                [1, 4],
                [7, 10]
            ],
            [
                [2, 5],
                [8, 11]
            ],
            [
                [3, 6],
                [9, 12]
            ]
        ],
        [
            [
                [1, 4],
                [7, 10]
            ],
            [
                [2, 5],
                [8, 11]
            ],
            [
                [3, 6],
                [9, 12]
            ]
        ]
    ])
    var rank4TensorFeatureChannelFirst: Tensor!

    override func setUpWithError() throws {
        scalarTensor = Tensor(scalar)
        vectorTensor = Tensor(vector)
        matrixTensor = Tensor(matrix)
        rank3Tensor = Tensor(rank3Array)
        
        rank4TensorFeatureChannelLast = Tensor(featureChannelLastArray)
        rank4TensorFeatureChannelFirst = Tensor(featureChannelFirstArray, featureChannelPosition: .first)
    }
    
    func testInitWithScalar() throws {
        let tensor = Tensor(scalar)
        XCTAssertEqual(tensor, scalarTensor)
    }
    
    func testInitWithVector() throws {
        let vector = [Float]([3, 4, 6, 7])
        let vectorData = vector.makeBuffer()
        let vectorTensor = Tensor(vector)
        let expectTensor = Tensor(shape: [vector.count], data: vectorData, dataType: .float32)
        XCTAssertEqual(vectorTensor, expectTensor)
    }
    
    func testInitWithRank4Tensor() throws {
        let tensor = Tensor([[[[Float]]]]([
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]]
            ],
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]]
            ]
        ]))
        let flattenedArray: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3 , 4, 5, 6, 7, 8, 9, 10, 11, 12]
        let expectedData = flattenedArray.makeBuffer()
        let expectedTensor = Tensor(shape: [2, 2, 2, 3], data: expectedData, dataType: .float32)
        XCTAssertEqual(tensor, expectedTensor)
    }
    
    func testScalarValue() throws {
        // test data conversion
        XCTAssertEqual(Tensor(scalar).extractScalar(), scalar)
    }
    
    func testFeatureChannelCount() throws {
        // Test scalar
        XCTAssertEqual(Tensor(shape: [], dataType: .float32).featureChannelCount, 0,
                       "Scalar should not have a channel count.")
        
        // Test rank 4 tensor
        XCTAssertEqual(rank4TensorFeatureChannelFirst.featureChannelCount,
                       featureChannelFirstArray[0].count)
        XCTAssertEqual(rank4TensorFeatureChannelLast.featureChannelCount,
                       featureChannelLastArray[0][0][0].count)
    }
    
    func testDefaultFeatureChannelPosition() {
        XCTAssertEqual(Tensor(shape: [], dataType: .float32).featureChannelPosition, .notApplicable)
        XCTAssertEqual(Tensor(shape: [1], dataType: .float32).featureChannelPosition, .notApplicable)
        XCTAssertEqual(Tensor(shape: [1, 3], dataType: .float32).featureChannelPosition, .notApplicable)
        XCTAssertEqual(Tensor(shape: [1, 3, 4], dataType: .float32).featureChannelPosition, .notApplicable)
        XCTAssertEqual(Tensor(shape: [1, 2, 3, 4], dataType: .float32).featureChannelPosition, .last)
    }
    
    func testExtractData() {
        XCTAssertEqual(scalarTensor.extract(Float.self) as? Float, scalar)
        XCTAssertEqual(vectorTensor.extract(Float.self) as? [Float], vector)
        XCTAssertEqual(matrixTensor.extract(Float.self) as? [[Float]], matrix)
        XCTAssertEqual(rank3Tensor.extract(Float.self) as? [[[Float]]], rank3Array)
        XCTAssertEqual(rank4TensorFeatureChannelLast.extract(Float.self) as? [[[[Float]]]], featureChannelLastArray)
    }
    
    func testAppendVectorTensors() {
        var baseTensor = Tensor(shape: [], dataType: .float32)
        let vectorTensor1 = Tensor([Float]([1, 2, 3]))
        baseTensor.append(vectorTensor1)
        XCTAssertEqual(baseTensor.shape, [1,3])
        XCTAssertEqual(baseTensor.data, vectorTensor1.data)
        
        let vectorTensor2 = Tensor([Float]([4, 5, 6]))
        baseTensor.append(vectorTensor2)
        XCTAssertEqual(baseTensor.shape, [2,3])
        XCTAssertEqual(baseTensor.data, vectorTensor1.data + vectorTensor2.data)
    }
    
    func testAppendRank4Tensors() {
        var baseTensor = Tensor(shape: [],
                                dataType: rank4TensorFeatureChannelFirst.dataType)
        baseTensor.append(rank4TensorFeatureChannelFirst)
        XCTAssertEqual(baseTensor, rank4TensorFeatureChannelFirst)
        
        let originalBatchSize = baseTensor.shape[0]
        baseTensor.append(rank4TensorFeatureChannelFirst)
        XCTAssertEqual(baseTensor.shape[0], originalBatchSize + rank4TensorFeatureChannelFirst.shape[0])
        XCTAssertEqual(baseTensor.data, rank4TensorFeatureChannelFirst.data + rank4TensorFeatureChannelFirst.data)
    }
    
    func testSubscriptVectorInRange() {
        let tensor = Tensor([Float]([1, 2, 3, 4, 5]))
        let range = 1 ..< 4
        let closedRange = 1 ... 3
        let expectedTensor = Tensor([Float]([2, 3, 4]))
        XCTAssertEqual(tensor[range], expectedTensor)
        XCTAssertEqual(tensor[closedRange], expectedTensor)
    }
    
    func testSubscriptMatrixInRange() {
        let tensor = Tensor([[Float]]([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ]))
        let range = 2..<4
        let closedRange = 2...3
        let expectedTensor = Tensor([[Float]]([
            [7, 8, 9],
            [10, 11, 12]
        ]))
        XCTAssertEqual(tensor[range], expectedTensor)
        XCTAssertEqual(tensor[closedRange], expectedTensor)
    }
    
    func testShapeCount() {
        let tensor = Tensor(shape: [3, 2], dataType: .float32)
        XCTAssertEqual(tensor.shapeCount, 3 * 2)
    }
    
    func testShapeByteCount() {
        let tensor = Tensor(shape: [4, 5], dataType: .float32)
        XCTAssertEqual(tensor.shapeByteCount, 4 * 5 * MemoryLayout<Float>.size)
    }
}
