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
//  Created by Brian Smith on 5/26/23.
//

import XCTest
import MimicTransferables

final class RangeTests: XCTestCase {
    struct Constant {
        static let minimumFloat: Float = 1
        static let maximumFloat: Float = 10
    }
    
    func testCreateHalfOpenRange() throws {
        let nativeRange = Constant.minimumFloat ..< Constant.maximumFloat
        let range = Range(nativeRange)
        XCTAssertEqual(range.kind, .halfOpen)
        XCTAssertEqual(range.lowerBound, Constant.minimumFloat.makeBuffer())
        XCTAssertEqual(range.upperBound, Constant.maximumFloat.makeBuffer())
        XCTAssertEqual(range.dataType, .float32)
    }
    
    func testCreateClosedRange() throws {
        let nativeRange = Constant.minimumFloat ... Constant.maximumFloat
        let range = Range(nativeRange)
        XCTAssertEqual(range.kind, .closed)
        XCTAssertEqual(range.lowerBound, Constant.minimumFloat.makeBuffer())
        XCTAssertEqual(range.upperBound, Constant.maximumFloat.makeBuffer())
        XCTAssertEqual(range.dataType, .float32)
    }
}
