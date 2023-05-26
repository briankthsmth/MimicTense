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
//  Created by Brian Smith on 5/24/23.
//

import XCTest
import MimicCore
import MimicTransferables

final class DataTypeRandomTests: XCTestCase {
    func testRandomFloatInRange() throws {
        let range = Float(1) ..< Float(2)
        for _ in 0 ..< 30 {
            let randomFloat = try DataType.float32.random(in: range)
            XCTAssertTrue(range.contains(randomFloat))
        }
    }
    
    func testRandomFloatInClosedRange() throws {
        let range = Float(1) ... Float(10)
        for _ in 0 ..< 30 {
            let bufferedFloat = try DataType.float32.random(in: range)
            let randomFloat = try Float.makeValue(from: bufferedFloat)
            XCTAssertTrue(range.contains(randomFloat))
        }
    }    
}
