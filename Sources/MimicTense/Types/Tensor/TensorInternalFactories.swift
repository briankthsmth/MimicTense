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
//  Created by Brian Smith on 2/16/23.
//

import Foundation
import MimicTransferables

extension Tensor {
    static func make<NativeType: NeuralNativeType>(from transferableTensor: MimicTransferables.Tensor) throws -> Tensor<NativeType>
    {
        guard
            MimicTransferables.DataType(NativeType.self) == transferableTensor.dataType,
            let tensorData = transferableTensor.extract(NativeType.self)
        else {
            throw ThrowableError.uncovertableData
        }
        switch tensorData {
        case let scalar as NativeType:
            return Tensor<NativeType>(scalar)
        case let vector as [NativeType]:
            return Tensor<NativeType>(vector)
        case let matrix as [[NativeType]]:
            return Tensor<NativeType>(matrix)
        case let rank3Array as [[[NativeType]]]:
            return Tensor<NativeType>(rank3Array)
        case let rank4Array as [[[[NativeType]]]]:
            return Tensor<NativeType>(rank4Array)
        default:
            throw ThrowableError.uncovertableData
        }
    }
    
    func makeTransferable() throws -> MimicTransferables.Tensor {
        return MimicTransferables.Tensor(shape: shape,
                                         data: data.makeBuffer(),
                                         dataType: DataType(NativeType.self),
                                         randomInitializerType: randomizer?.makeTransferable())
    }
    
    static func makeFillUniform(shape: [Int], in range: ClosedRange<NativeType>) -> MimicTense.Tensor<NativeType> {
        let dataCount = shape.reduce(1, *)
        var data: [NativeType] = []
        for _ in 0 ..< dataCount {
            data.append(NativeType.random(in: range))
        }
        return Tensor(shape: shape, data: data)
    }
}
