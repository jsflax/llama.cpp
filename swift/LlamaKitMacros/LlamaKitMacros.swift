import Foundation
import SwiftSyntaxMacros
import SwiftCompilerPlugin
import SwiftSyntax


enum LlamaKitMacroError: Error {
    case message(String)
}

struct ToolMacro: BodyMacro {
    static func expansion(of node: SwiftSyntax.AttributeSyntax, providingBodyFor declaration: some SwiftSyntax.DeclSyntaxProtocol & SwiftSyntax.WithOptionalCodeBlockSyntax, in context: some SwiftSyntaxMacros.MacroExpansionContext) throws -> [SwiftSyntax.CodeBlockItemSyntax] {
        []
    }
}

struct LlamaActorMacro: ExtensionMacro, MemberMacro {
    static func expansion(of node: AttributeSyntax, providingMembersOf declaration: some DeclGroupSyntax, conformingTo protocols: [TypeSyntax], in context: some MacroExpansionContext) throws -> [DeclSyntax] {
        return [
            """
            var session: LlamaToolSession!
            
            public init(params: GPTParams) async throws {
                \(raw: declaration.inheritanceClause != nil ? "self.init()" : "")
                let tools = Self.tools(self)
                self.session = try await LlamaToolSession(params: params, tools: tools)
            }
            """
        ]
    }
    
    static func expansion(of node: AttributeSyntax,
                          attachedTo declaration: some DeclGroupSyntax,
                          providingExtensionsOf type: some TypeSyntaxProtocol,
                          conformingTo protocols: [TypeSyntax],
                          in context: some MacroExpansionContext) throws -> [ExtensionDeclSyntax] {
        var tools: [
            (name: String,
             description: String,
             parameters: [(name: String,
                           type: String,
                           description: String)],
             callableString: String,
             callableName: String)
        ] = []
        let typeName = type.as(IdentifierTypeSyntax.self)!.name.text
        for member in declaration.memberBlock.members {
            var comments = member.leadingTrivia.filter { $0.isComment }
            guard let member = member.decl.as(FunctionDeclSyntax.self) else {
                continue
            }
            guard member.attributes.contains(where: { element in
                element.as(AttributeSyntax.self)?.attributeName.as(IdentifierTypeSyntax.self)?.name.text == "Tool"
            }) else {
                continue
            }

            let name = member.name
            // TODO: This should be better. It's basically dropping any non docstring comments
            // TODO: before the docstring comments.
            comments = Array(comments.drop(while: {
                if case .docLineComment(_) = $0 {
                    return false
                } else {
                    return true
                }
            }))
            guard case var .docLineComment(description) = comments.first else {
                throw LlamaKitMacroError.message("Missing comment for tool \(name.text).")
            }
            description = String(description.dropFirst(3))
            var parameters: [(name: String, type: String, description: String)] = []
            var index = 0
            for parameter in member.signature.parameterClause.parameters {
                let firstName = parameter.firstName.text
                var typeName: String
                if let type = parameter.type.as(IdentifierTypeSyntax.self) {
                    typeName = type.name.text
                } else if let type = parameter.type.as(OptionalTypeSyntax.self),
                    let type = type.wrappedType.as(IdentifierTypeSyntax.self) {
                    typeName = type.name.text
                } else if let type = parameter.type.as(ArrayTypeSyntax.self),
                    let type = type.element.as(IdentifierTypeSyntax.self) {
                    typeName = "[\(type.name.text)]"
                }
                else {
                    throw LlamaKitMacroError.message("Incorrect type for parameter \(parameter.debugDescription)")
                }
                guard case var .docLineComment(description) = comments[index + 1] else {
                    throw LlamaKitMacroError.message("Missing comment for \(firstName)")
                }
                description = String(description.dropFirst("/// - parameter ".count + firstName.count + ":".count)).trimmingCharacters(in: .whitespacesAndNewlines)
                parameters.append((name: firstName, type: typeName, description: description))
                index += 1
            }
            let callableName = context.makeUniqueName(name.text)
            let callableString = """
            @dynamicCallable struct \(callableName.text): DynamicCallable {
                \(parameters.map {
                """
                struct \($0.name)Arg: _Argument {
                    typealias ArgType = \($0.type)
                    static var argKey: String { "\($0.name)" }
                }
                """
                }.joined(separator: "\n"))

                private weak var llamaActor: \(typeName)?
                init(_ llamaActor: \(typeName)) {
                    self.llamaActor = llamaActor
                }
            
                @discardableResult
                func dynamicallyCall(withKeywordArguments args: [String: Any]) async throws -> String {
                    \(parameters.isEmpty ? "" : "let args = try extractArguments(\(parameters.map {$0.name + "Arg.self"}.joined(separator: ",")), from: args)")
                    \(parameters.map {
                        "let \($0.name) = args[\"\($0.name)\"] as! \($0.type)"
                    }.joined(separator: "\n"))
                    let returnValue = try await self.llamaActor!.\(name.text)(\(parameters.map { "\($0.name): \($0.name)" }.joined(separator: ",")))
                    let jsonValue = try JSONEncoder().encode(returnValue)
                    return String(data: jsonValue, encoding: .utf8)!
                }
            }
            """
            tools.append((name: name.text, description: description,
                          parameters: parameters,
                          callableString: callableString,
                          callableName: callableName.text))
        }
        
        
        return [
            .init(extendedType: type,
                inheritanceClause: .init(inheritedTypes: InheritedTypeListSyntax.init(arrayLiteral: .init(type: IdentifierTypeSyntax(name: "LlamaActor")))),
                  memberBlock: """
            {
                \(raw: tools.map {
                    $0.callableString
                }.joined(separator: "\n"))
            
                static func tools(_ self: \(raw: typeName)) -> [String: (DynamicCallable, _JSONFunctionSchema)] {
                    [\(raw: tools.map { tool in
                        """
                        "\(tool.name)": (\(tool.callableName)(self), _JSONFunctionSchema(name: "\(tool.name)", description: "\(tool.description)", parameters: _JSONFunctionSchema.Parameters(properties: \(tool.parameters.count == 0 ? "[:]" : "[" + tool.parameters.map { parameter in
                            """
                            "\(parameter.name)": _JSONFunctionSchema.Property(type: \(parameter.type).self, description: "\(parameter.description)"),
                            """
                            }.joined() + "]"), required: [\(tool.parameters.map { "\"\($0.name)\""}.joined(separator: ","))])))
                        """
                    }.joined(separator: ","))]
                }
            }
            """)
        ]
    }
}

@main
struct LlamaKitMacrosPlugin: CompilerPlugin {
    let providingMacros: [Macro.Type] = [
        LlamaActorMacro.self, ToolMacro.self
    ]
}
