import Foundation

public protocol DynamicCallable: Sendable {
    @discardableResult
    func dynamicallyCall(withKeywordArguments args: [String: Any]) async throws -> String
}

public protocol _CodableArgument {
    static func _extractValue(_ value: AnyCodable, for key: String) throws -> Self
}

public protocol _Argument {
    associatedtype ArgType: JSONSchemaConvertible
    static var argKey: String { get }
}

extension String: _CodableArgument {
    public static func _extractValue(_ value: AnyCodable, for key: String) throws -> Self {
        guard case let .string(string) = value else {
            throw DynamicCallableErrorKind.invalidType(key: key)
        }
        return string
    }
}

extension Int: _CodableArgument {
    public static func _extractValue(_ value: AnyCodable, for key: String) throws -> Self {
        guard case let .int(int) = value else {
            throw DynamicCallableErrorKind.invalidType(key: key)
        }
        return int
    }
}

extension Double: _CodableArgument {
    public static func _extractValue(_ value: AnyCodable, for key: String) throws -> Self {
        guard case let .double(double) = value else {
            throw DynamicCallableErrorKind.invalidType(key: key)
        }
        return double
    }
}

extension Bool: _CodableArgument {
    public static func _extractValue(_ value: AnyCodable, for key: String) throws -> Self {
        guard case let .bool(bool) = value else {
            throw DynamicCallableErrorKind.invalidType(key: key)
        }
        return bool
    }
}

extension Date: _CodableArgument {
    public static func _extractValue(_ value: AnyCodable, for key: String) throws -> Self {
        guard case let .date(date) = value else {
            throw DynamicCallableErrorKind.invalidType(key: key)
        }
        return date
    }
}

extension UUID: _CodableArgument {
    public static func _extractValue(_ value: AnyCodable, for key: String) throws -> Self {
        guard case let .uuid(uuid) = value else {
            throw DynamicCallableErrorKind.invalidType(key: key)
        }
        return uuid
    }
}

extension Array: _CodableArgument where Element: _CodableArgument {
    public static func _extractValue(_ value: AnyCodable,
                                     for key: String) throws -> Array<Element> {
        guard case let .array(array) = value else {
            throw DynamicCallableErrorKind.invalidType(key: key)
        }
        return try array.map {
            try Element._extractValue($0, for: key)
        }
    }
}

extension _CodableArgument where Self: JSONSchemaConvertible {
    public static func _extractValue(_ value: AnyCodable,
                                     for key: String) throws -> Self {
        guard case let .object(object) = value else {
            throw DynamicCallableErrorKind.invalidType(key: key)
        }
        fatalError()
//        let data = try JSONSerialization
//            .data(withJSONObject: object.mapValues {
//                type(of: )
//            })
//        return try JSONDecoder().decode(Self.self, from: data)
    }
}

private enum DynamicCallableErrorKind: LocalizedError {
    case missingKey(String)
    case invalidEncoding(key: String)
    case invalidType(key: String)
    
    var errorDescription: String? {
        switch self {
        case .missingKey(let string):
            "Missing key: \(string)"
        case .invalidEncoding(let key):
            "Invalid encoding for key: \(key)"
        case .invalidType(let key):
            "Invalid type for key: \(key)"
        }
    }
}

extension DynamicCallable {
    private func extractArgument<T>(_ arg: T.Type,
                                    from args: [String: Any],
                                    to out: inout [String: Any]) throws where T: _Argument {
        guard let anyDecodable = args[T.argKey] else {
            throw DynamicCallableErrorKind.missingKey(T.argKey)
        }
        out[T.argKey] = try T.ArgType(from: anyDecodable)
    }
    
    public func extractArguments<each Ts>(_ arguments: repeat (each Ts).Type,
                                          from args: [String: Any]) throws -> [String: Any]
    where repeat each Ts: _Argument {
        var extractedArgs: [String: Any] = [:]
        try (repeat (extractArgument((each Ts).self, from: args, to: &extractedArgs)))
        return extractedArgs
    }
}

public enum AnyCodable: Codable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case uuid(UUID)
    case date(Date)
    indirect case array([AnyCodable])
    indirect case object([String: AnyCodable])
    
    case null
    // Add other cases as needed

    // Initializers for each type
    init(_ value: String) {
        self = .string(value)
    }
    init(_ value: Int) {
        self = .int(value)
    }
    init(_ value: Double) {
        self = .double(value)
    }
    init(_ value: Bool) {
        self = .bool(value)
    }
    init(_ value: UUID) {
        self = .uuid(value)
    }
    init(_ value: Date) {
        self = .date(value)
    }
    init(_ value: [AnyCodable]) {
        self = .array(value)
    }
    init() {
        self = .null
    }

    // Decodable conformance
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        
        if container.decodeNil() {
            self = .null
        } else if let intValue = try? container.decode(Int.self) {
            self = .int(intValue)
        } else if let doubleValue = try? container.decode(Double.self) {
            self = .double(doubleValue)
        } else if let boolValue = try? container.decode(Bool.self) {
            self = .bool(boolValue)
        } else if let uuidValue = try? container.decode(UUID.self) {
            self = .uuid(uuidValue)
        }  else if let dateValue = try? container.decode(Date.self) {
            self = .date(dateValue)
        } else if let stringValue = try? container.decode(String.self) {
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyy-MM-dd"
            if let date = formatter.date(from: stringValue) {
                self = .date(date)
            } else {
                self = .string(stringValue)
            }
        } else if let arrayValue = try? container.decode([AnyCodable].self) {
            self = .array(arrayValue)
        } else if let objectValue = try? container.decode([String: AnyCodable].self) {
            self = .object(objectValue)
        } else {
            let context = DecodingError.Context(
                codingPath: decoder.codingPath,
                debugDescription: "Cannot decode AnyDecodable"
            )
            throw DecodingError.typeMismatch(AnyCodable.self, context)
        }
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()

        switch self {
        case .string(let value):
            try container.encode(value)
        case .int(let value):
            try container.encode(value)
        case .double(let value):
            try container.encode(value)
        case .bool(let value):
            try container.encode(value)
        case .uuid(let value):
            try container.encode(value)
        case .date(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }
    
    var asAny: Any? {
        switch self {
        case .string(let value): return value
        case .int(let value): return value
        case .double(let value): return value
        case .bool(let value): return value
        case .uuid(let value): return value
        case .date(let value): return value
        case .array(let value): return value
        case .object(let value): return value
        case .null: return nil
        }
    }
}
