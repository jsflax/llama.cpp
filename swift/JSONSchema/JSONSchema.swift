import Foundation

public struct JSONSchema : Codable {
    public struct Items : Codable {
        let type: String
        let `enum`: [String]?
        let properties: [String: Property]?
        
        public init(type: String, `enum`: [String]?, properties: [String: Property]?) {
            self.type = type
            self.enum = `enum`
            self.properties = properties
        }
    }
    public struct Property : Codable {
        let type: String
        let items: Items?
        let description: String?
        
        public init(type: String, items: Items?, description: String?) {
            self.type = type
            self.items = items
            self.description = description
        }
    }
    let type: String
    let items: Items?
    let properties: [String : Property]?
    
    public init(type: String, items: Items?, properties: [String : Property]?) {
        self.type = type
        self.items = items
        self.properties = properties
    }
}


public struct _JSONFunctionSchema: Codable {
    public typealias Items = JSONSchema.Items
//    public struct Items: Codable {
//        let type: String
//        let `enum`: [String]?
//        let properties: [Property]?
//        
//        public init(type: String, `enum`: [String]?, properties: [Property]?) {
//            self.type = type
//            self.enum = `enum`
//        }
//    }

    public struct Property: Codable {
        let type: String
        let items: Items?
        let `enum`: [String]?
        let description: String?
        
        public init(type: String.Type, description: String?) {
            self.type = "string"
            self.description = description
            self.items = nil
            self.enum = nil
        }
        
        public init(type: Int.Type, description: String?) {
            self.type = "integer"
            self.description = description
            self.items = nil
            self.enum = nil
        }
        
        
        public init(type: Double.Type, description: String?) {
            self.type = "number"
            self.description = description
            self.items = nil
            self.enum = nil
        }
        
        public init(type: Bool.Type, description: String?) {
            self.type = "boolean"
            self.description = description
            self.items = nil
            self.enum = nil
        }
        
        public init(type: UUID.Type, description: String?) {
            self.type = "string"
            self.description = description
            self.items = nil
            self.enum = nil
        }
        
        public init(type: Date.Type, description: String?) {
            self.type = "string"
            self.description = description
            self.items = nil
            self.enum = nil
        }
        
        public init<T: JSONSchemaConvertible>(type: Array<T>.Type, description: String?) {
            self.type = "array"
            self.description = description
            self.items = Array<T>.items
            self.enum = nil
        }
        public init<T: JSONSchemaConvertible>(type: T.Type, description: String?) {
            self.type = "object"
            self.description = description
            self.items = T.items
            self.enum = nil
        }
        public init<T: CaseIterable>(type: T.Type, description: String?) where T: RawRepresentable,
        T: StringProtocol {
            self.type = "string"
            self.enum = Array(type.allCases.map { $0.rawValue as! String })
            self.description = description
            self.items = nil
        }
    }
    
    
    public struct Parameters: Codable {
        public let properties: [String: Property]
        public let required: [String]
        public var type = "object"
        
        public init(properties: [String : Property], required: [String]) {
            self.properties = properties
            self.required = required
        }
    }
    
    let name: String
    let description: String
    let parameters: Parameters
    
    public init(name: String, description: String, parameters: Parameters) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

public enum JSONDecodingError: Error, CustomStringConvertible {
    case invalidType
    case missingRequiredProperty(String)
    case invalidValue
    
    public var description: String {
        switch self {
        case .invalidType:
            "Invalid type"
        case .missingRequiredProperty(let string):
            "Missing required property: \(string)"
        case .invalidValue:
            "Invalid value"
        }
    }
}

public protocol JSONSchemaConvertible : Codable {
    static var type: String { get }
    static var jsonSchema: [String : Any] { get }
    static var properties: [String: JSONSchema.Property]? { get }
    init(from json: Any) throws
    static func decode<K: CodingKey>(from container: KeyedDecodingContainer<K>,
                                     forKey key: K) throws -> Self
}

extension RawRepresentable where Self : CaseIterable, RawValue : JSONSchemaConvertible, Self: Codable {
    public static var type: String {
        RawValue.type
    }
    public static var jsonSchema: [String: Any] {
        [
            "type": RawValue.type,
            "enum": Self.allCases.map(\.rawValue)
        ]
    }
}

extension JSONSchemaConvertible {
    public static var items: JSONSchema.Items? {
        nil
    }
    public static var `enum`: [String]? {
        nil
    }
    public static func decode<K: CodingKey>(from container: KeyedDecodingContainer<K>, forKey key: K) throws -> Self {
        return try container.decode(Self.self, forKey: key)
    }
}
extension String : JSONSchemaConvertible {
    public init(from json: Any) throws {
        guard let json = json as? String else {
            throw JSONDecodingError.invalidType
        }
        self = json
    }
    
    public static var properties: [String : JSONSchema.Property]? {
        nil
    }
    
    public static var type: String { "string" }
    public static var jsonSchema: [String: Any] {
        [
            "type": "string"
        ]
    }
}
extension UUID : JSONSchemaConvertible {
    public init(from json: Any) throws {
        guard let json = json as? String else {
            throw JSONDecodingError.invalidType
        }
        guard let json = UUID(uuidString: json) else {
            throw JSONDecodingError.invalidValue
        }
        self = json
    }
    
    public static var properties: [String : JSONSchema.Property]? {
        nil
    }
    
    public static var type: String { "string" }
    public static var jsonSchema: [String: Any] {
        [
            "type": "string"
        ]
    }
}
extension Int : JSONSchemaConvertible {
    public init(from json: Any) throws {
        guard let json = json as? Int else {
            throw JSONDecodingError.invalidType
        }
        self = json
    }
    public static var properties: [String : JSONSchema.Property]? {
        nil
    }
    
    public static var type: String { "number" }
    public static var jsonSchema: [String: Any] {
        [
            "type": "integer"
        ]
    }
}
extension Double : JSONSchemaConvertible {
    public init(from json: Any) throws {
        guard let json = json as? Double else {
            throw JSONDecodingError.invalidType
        }
        self = json
    }
    public static var properties: [String : JSONSchema.Property]? {
        nil
    }
    
    public static var type: String { "number" }
    public static var jsonSchema: [String: Any] {
        [
            "type": "number"
        ]
    }
}
extension Bool : JSONSchemaConvertible {
    public init(from json: Any) throws {
        guard let json = json as? Bool else {
            throw JSONDecodingError.invalidType
        }
        self = json
    }
    public static var properties: [String : JSONSchema.Property]? {
        nil
    }
    
    public static var type: String { "boolean" }
    public static var jsonSchema: [String: Any] {
        [
            "type": "boolean"
        ]
    }
}
extension Date : JSONSchemaConvertible {
    public init(from json: Any) throws {
        guard let json = json as? String else {
            throw JSONDecodingError.invalidType
        }
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        
        guard let json = formatter.date(from: json) else {
            throw JSONDecodingError.invalidValue
        }
        self = json
    }
    public static var properties: [String : JSONSchema.Property]? { nil }
    public static var type: String { "string" }

    public static var jsonSchema: [String: Any] {
        [
            "type": "string"
        ]
    }

    public static func decode<K: CodingKey>(from container: KeyedDecodingContainer<K>, forKey key: K) throws -> Self {
        let value = try container.decode(String.self, forKey: key)
        let detector = try? NSDataDetector(types: NSTextCheckingResult.CheckingType.date.rawValue)
        let matches = detector?.matches(in: value, options: [], range: NSMakeRange(0, value.utf16.count))
        return matches!.first!.date!
        // return ISO8601DateFormatter().date(from: value)!
    }
}

extension Array : JSONSchemaConvertible where Element : JSONSchemaConvertible {
    public init(from json: Any) throws {
        guard let json = json as? Array<Any> else {
            throw JSONDecodingError.invalidType
        }
        self = try json.map(Element.init)
    }
    public static var type: String { "array" }
    public static var items: JSONSchema.Items? {
        JSONSchema.Items(type: Element.type, enum: Element.enum, properties: Element.properties)
    }
    public static var properties: [String : JSONSchema.Property]? { nil }
    public static var jsonSchema: [String : Any] {
        [
            "type": "array",
            "items": Element.jsonSchema
        ]
    }
}

@attached(member, names: arbitrary)
@attached(extension, conformances: JSONSchemaConvertible, CaseIterable, names: arbitrary)
public macro JSONSchema() = #externalMacro(module: "JSONSchemaMacros",
                                           type: "JSONSchemaMacro")

//@attached(member, names: arbitrary)

