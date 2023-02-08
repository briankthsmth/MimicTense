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
//  Created by Brian Smith on 1/25/23.
//

import Foundation
import MimicTransferables

struct TensorTestData {
    let scalar: Float = 8
    let scalarTensor: Tensor
    let vector: [Float] = [1, 2 , 3, 4]
    let vectorTensor: Tensor
    let matrix: [[Float]] = [[1,2], [3, 4]]
    let matrixTensor: Tensor
    let rank3Array: [[[Float]]] = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    let rank3Tensor: Tensor
    
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
    let rank4TensorFeatureChannelLast: Tensor
    
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
    let rank4TensorFeatureChannelFirst: Tensor

    init() {
        scalarTensor = Tensor(scalar)
        vectorTensor = Tensor(vector)
        matrixTensor = Tensor(matrix)
        rank3Tensor = Tensor(rank3Array)
        
        rank4TensorFeatureChannelLast = Tensor(featureChannelLastArray)
        rank4TensorFeatureChannelFirst = Tensor(featureChannelFirstArray, featureChannelPosition: .first)
    }
}
