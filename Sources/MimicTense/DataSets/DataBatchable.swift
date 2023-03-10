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
//  Created by Brian Smith on 7/8/22.
//

import Foundation
import MimicTransferables

public protocol DataBatchable {
    associatedtype NativeType: NeuralNativeType
    
    var batchSize: Int { get }
    var inputTensors: [Tensor<NativeType>] { get }
    var labels: Tensor<NativeType>? { get }
}

extension DataBatchable {
    func makeTransferable() throws -> MimicTransferables.DataSet {
        MimicTransferables.DataSet(inputTensors: try inputTensors.map { try $0.makeTransferable() },
                                   labels: try labels?.makeTransferable(),
                                   batchSize: batchSize)
    }
}
