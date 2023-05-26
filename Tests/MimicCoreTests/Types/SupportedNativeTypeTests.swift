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
//  Created by Brian Smith on 2/28/23.
//

import XCTest
import MimicCore

final class SupportedNativeTypeTests: XCTestCase {
    struct Mock<NativeType: SupportedNativeType> {
        let value: NativeType
    }
    
    struct Constant {
        static let floatValue: Float = 4
        static let bufferedFloat = [UInt8](arrayLiteral: 0, 0, 128, 64)
    }
    
    func testMultiplicationWithFloats() {
        let mock1 = Mock(value: Float(2))
        let mock2 = Mock(value: Float(3))
        XCTAssertEqual(mock1.value * mock2.value, 6, accuracy: 0.001)
    }
    
    func testMakeBufferFromFloat() throws {
        let buffer = Constant.floatValue.makeBuffer()
        XCTAssertEqual(buffer, Constant.bufferedFloat)
    }
    
    func testMakeFloatFromBuffer() throws {
        let value = try Float.makeValue(from: Constant.bufferedFloat)
        XCTAssertEqual(value, Constant.floatValue)
        XCTAssertThrowsError(try Float.makeValue(from: [UInt8](arrayLiteral: 0, 0)))
    }
}
