//
//  RandomDescriptorTests.swift
//  
//
//  Created by Brian Smith on 5/26/23.
//

import XCTest
import MimicTransferables

final class RandomDescriptorTests: XCTestCase {
    func testInitializer() {
        let range = Range(Float(1) ... Float(10))
        let descriptor = RandomDescriptor(type: .uniform, range: range)
        
        XCTAssertEqual(descriptor.type, .uniform)
        XCTAssertEqual(descriptor.range, range)
    }

    func testInitializerFromSwiftRange() {
        let swiftRange = Float(1)..<Float(3)
        let descriptor = RandomDescriptor(type: .uniform, range: swiftRange)
        
        XCTAssertEqual(descriptor.type, .uniform)
        XCTAssertEqual(descriptor.range, Range(swiftRange))
    }

    func testInitializerFromSwiftClosedRange() {
        let swiftRange = Float(1)...Float(3)
        let descriptor = RandomDescriptor(type: .uniform, range: swiftRange)
        
        XCTAssertEqual(descriptor.type, .uniform)
        XCTAssertEqual(descriptor.range, Range(swiftRange))
    }
}
