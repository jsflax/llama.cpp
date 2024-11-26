import Foundation
import Testing
@testable import LlamaKit
import JSONSchema
import OSLog

@llamaActor actor MyLlama {
    /// Get the user's favorite season
    @Tool public func getFavoriteSeason() async throws -> String {
        return "autumn"
    }
    
    /// Get the user's favorite animal.
    @Tool public func getFavoriteAnimal() async throws -> String {
        return "cat"
    }
    
    /// Get the color of the sky.
    @Tool public func getSkyColor() async throws -> String {
        return "blue"
    }
    
    /// Reverse a given string.
    /// - parameter string: String to reverse.
    @Tool public func reverse(string: String) async throws -> String {
        String(string.reversed())
    }
    
    /// Reduce an array of integers into a sum
    /// - parameter numbers: the numbers to get the sum of
    @Tool public func sum(numbers: [Int]) async throws -> Int {
        numbers.reduce(0, +)
    }
    
    @JSONSchema struct Product: _CodableArgument {
        let name: String
        let price: Double
    }
    
    /// Calculate the total price of products
    /// - parameter products: the array of products to calculate the total price for
    @Tool public func totalPrice(products: [Product]) async throws -> Double {
        products.reduce(0) { $0 + $1.price }
    }
}

@llamaActor actor LlamaChainTools {
    @JSONSchema struct Product {
        let name: String
        let price: Double
        
        init(name: String, price: Double) {
            self.name = name
            self.price = price
        }
    }
    
    @JSONSchema struct ProductOptions {
        let name: String
        init(name: String) {
            self.name = name
        }
    }
    
    /// Get a list of available products.
    @Tool public func getAvailableProducts() async throws -> [Product] {
        [
            Product(name: "banana", price: 5),
            Product(name: "apple", price: 3)
        ]
    }
    
    /// Get options for a provided product.
    /// - parameter productName: the name of the product to get options for
    @Tool public func getOptionsForProduct(productName: String) async throws -> [ProductOptions] {
        if productName == "banana" {
            return [.init(name: "plantain"), .init(name: "miniature")]
        } else if productName == "apple" {
            return [.init(name: "granny smith"), .init(name: "red delicious")]
        } else {
            abort()
        }
    }
    
    /// Add a product to the cart. This *must* only be called after options have been presented for a product.
    /// - parameter product: the product to add to the cart
    /// - parameter options: the chosen options for the product
    @Tool public func addProductToCart(product: Product,
                                       options: [ProductOptions]) async throws -> Bool {
        return true
    }
}

// MARK: LlamaGrammarSession Suite
@Suite("LlamaSession Suite")
struct LlamaSessionSuite {
    @JSONSchema struct Trip {
        let location: String
        let startDate: TimeInterval
        let durationInDays: Int
    }

    func downloadFile(url: String, to path: String) async throws -> String {
        let fm = FileManager.default
        let tmpDir = fm.temporaryDirectory
        let destinationURL = tmpDir.appending(path: path)
        
        guard !fm.fileExists(atPath: destinationURL.path()) else {
            return destinationURL.path()
        }
        print("Downloading \(path), this may take a while...")
        // Define the URL
        guard let url = URL(string: url) else {
            print("Invalid URL.")
            throw URLError(.badURL)
        }
        
        // Start the async download
        let (tempURL, _) = try await URLSession.shared.download(from: url)
        
        // Define the destination path in the documents directory
        
        // Move the downloaded file to the destination
        try fm.moveItem(at: tempURL, to: destinationURL)
        print("File downloaded to: \(destinationURL.path())")
        return destinationURL.path()
    }
    
    
    func baseParams(url: String, to path: String) async throws -> GPTParams {
        let params = GPTParams()
        params.modelPath = try await downloadFile(url: url, to: path)
        params.nPredict = 256
        params.nCtx = 4096
        params.cpuParams.nThreads = 8
        params.cpuParamsBatch.nThreads = 8
        params.nBatch = 1024
        params.nGpuLayers = 99
        params.logging = true
        params.chatTemplate = """
        <|system|>
        {system_message}</s>
        <|user|>
        {prompt}</s>
        <|assistant|>
        """
        params.interactive = true
        return params
    }
    
    @Test func llamaInferenceSession() async throws {
        let params = try await baseParams(url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf?download=true", to: "tinyllama.gguf")
        params.prompt = """
        <|system|>
        You are an AI assistant. Answer queries simply and concisely.</s>
        """
        params.antiPrompts = ["</s>"]
        params.inputPrefix = "<|user|>"
        params.inputSuffix = "</s><|assistant|>"
        params.interactive = true
        let session = try await LlamaChatSession(params: params, flush: false)
        for await msg in await session.inferenceStream(message: "How are you today?") {
            print(msg, terminator: "")
        }
    }
    
    @Test func llamaGrammarSession() async throws {
        let params = try await baseParams(url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf?download=true", to: "tinyllama.gguf")
        params.prompt = """
        You are a travel agent. The current date epoch \(Date.now.timeIntervalSince1970).
        Responses should have the following fields:
        
            location: the location of the trip
            startDate: the start of the trip as the unix epoch since 1970
            durationInDays: the duration of the trip in days
        
        """
        let session = try await LlamaSession<Trip>(params: params)
        await #expect(throws: Never.self) {
            let trip = try await session.infer(message: "Please create a trip for me to New York City that starts two weeks from now. The duration of the trip MUST be 3 days long.")
            #expect(trip.location.contains("New York"))
            // TODO: Testing the other fields is difficult considering model size
            // TODO: so for now, we are just asserting the grammar works
        }
    }
    
    @JSONSchema struct IsCorrect {
        let isSpellingCorrect: Bool
    }
    
    // MARK: Grammar Test
    @Test func llamaSimpleGrammarSession() async throws {
        let params = try await baseParams(url: "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_S.gguf?download=true",
                                          to: "spellcheck_mistral.gguf")
        params.prompt = """
        You are a spell checker. I will provide you with versions of the word 'strawberry'. Tell me if the spelling is correct or not.
        """
        params.inputPrefix = "<s>[INST]"
        params.inputSuffix = "[/INST]"
        params.antiPrompts.append("</s>")
        let session = try await LlamaSession<IsCorrect>(params: params)
        var output = try await session.infer(message: "strawberry\n")
        #expect(output.isSpellingCorrect)
        output = try await session.infer(message: "st44rawberrry\n")
        #expect(!output.isSpellingCorrect)
    }
    
    // MARK: Tool Test
    @Test func llamaToolSession() async throws {
        
        for i in 0..<10 {
            print("Iteration: \(i)")
            let params = try await baseParams(url: "https://huggingface.co/bartowski/Llama-3-Groq-8B-Tool-Use-GGUF/resolve/main/Llama-3-Groq-8B-Tool-Use-Q8_0.gguf?download=true", to: "llama_tools.gguf")
            let llama = try await MyLlama(params: params)
            var output = try await llama.infer("What's my favorite animal?")
            try #require(output.contains("cat"))
            output = try await llama.infer("What's my favorite season?")
            try #require(output.contains("autumn"))
            output = try await llama.infer("Reverse this string: \"hello world\"")
            try #require(output.contains("dlrow olleh"))
            output = try await llama.infer("Add these numbers together: 1 + 2")
            try #require(output.contains("3"))
            output = try await llama.infer("Get the total price for bananas that cost $2 and apples that cost $3")
            try #require(output.contains("5"))
        }
    }

    // MARK: Tool Test
    @Test func llamaToolSessionPersistence() async throws {
        let params = try await baseParams(url: "https://huggingface.co/bartowski/Llama-3-Groq-8B-Tool-Use-GGUF/resolve/main/Llama-3-Groq-8B-Tool-Use-Q8_0.gguf?download=true", to: "llama_tools.gguf")
        params.logging = true
        let fm = FileManager.default
        let tmpDir = fm.temporaryDirectory
        params.promptFile = tmpDir.appending(path: "prompt.txt").path()
        defer {
            try? fm.removeItem(atPath: params.promptFile!)
        }
        try await Task {
            let llama = try await MyLlama(params: params)
            let output = try await llama.infer("What color is the sky?")
            #expect(output.contains("blue"))
        }.value
        
        try await Task {
            let llama = try await MyLlama(params: params)
            let output = try await llama.infer("What was the last question I asked you?")
            print(output)
            #expect(output.contains("sky"))
        }.value
    }

    // MARK: Session dealloc Test
    // Note this test will fail if run in parallel
    @Test func llamaToolSessionDealloc() async throws {
        let params = try await baseParams(url: "https://huggingface.co/bartowski/Llama-3-Groq-8B-Tool-Use-GGUF/resolve/main/Llama-3-Groq-8B-Tool-Use-Q8_0.gguf?download=true", to: "llama_tools.gguf")
        func reportMemoryUsage() -> UInt64? {
            var info = mach_task_basic_info()
            var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: info)) / 4

            let kerr = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                    task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
                }
            }

            guard kerr == KERN_SUCCESS else {
                print("Error with task_info(): \(kerr)")
                return nil
            }

            return info.resident_size // Memory in bytes
        }
        var memPostAlloc: UInt64!
        try await Task {
            let llama = try await MyLlama(params: params)
            memPostAlloc = reportMemoryUsage()! / 1024 / 1024
            #expect(memPostAlloc > 500) // we are estimating here
            var output = try await llama.infer("What's my favorite animal?")
            print(output)
            output = try await llama.infer("What question did i just ask you?")
            print(output)
        }.value
        var memDealloc = reportMemoryUsage()! / 1024 / 1024
        #expect(memDealloc < 200)
        try await Task {
            let llama = try await MyLlama(params: params)
            memPostAlloc = reportMemoryUsage()! / 1024 / 1024
            #expect(memPostAlloc > 500)
            _ = try await llama.infer("What was the first question I asked you?")
        }.value
        memDealloc = reportMemoryUsage()! / 1024 / 1024
        #expect(memDealloc < 200)
    }

    // MARK: Stream Test
    @Test func llamaToolSessionStream() async throws {
        let params = try await baseParams(url: "https://huggingface.co/bartowski/Llama-3-Groq-8B-Tool-Use-GGUF/resolve/main/Llama-3-Groq-8B-Tool-Use-Q8_0.gguf?download=true", to: "llama_tools.gguf")
        let llama = try await MyLlama(params: params)
        var buffer = ""
        for await output in await llama.inferenceStream(message: "What's my favorite animal and season?") {
            buffer += output
            print(output, terminator: "")
        }
        #expect(buffer.contains("cat") && buffer.contains("autumn"))
    }
    
    private func cosineSimilarity(vectorA: [Float], vectorB: [Float]) -> Float {
        // Ensure vectors are the same length
        guard vectorA.count == vectorB.count else {
            return 0.0
        }

        // Compute dot product and magnitudes
        let dotProduct = zip(vectorA, vectorB).map(*).reduce(0, +)
        let magnitudeA = sqrt(vectorA.map { $0 * $0 }.reduce(0, +))
        let magnitudeB = sqrt(vectorB.map { $0 * $0 }.reduce(0, +))

        // Avoid division by zero
        guard magnitudeA != 0 && magnitudeB != 0 else {
            return 0.0
        }

        return dotProduct / (magnitudeA * magnitudeB)
    }
    
    @Test func llamaEmbeddingsSession() async throws {
        let params = try await baseParams(url: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f32.gguf?download=true", to: "embeddings_fp32.gguf")
        params.nCtx = 2048
        params.nBatch = 2048
        params.embdNormalize = 1
        let session = LlamaEmbeddingSession(parameters: params)
        let embeddings = await session.embeddings(for: "sushi", isQuery: true)
        let embeddings2 = await session.embeddings(for: "Advancements in quantum computing technologies", isQuery: false)
        print(cosineSimilarity(vectorA: embeddings, vectorB: embeddings2))
    }
    
    @Test func llamaChainTest() async throws {
        let params = try await baseParams(url: "https://huggingface.co/bartowski/Llama-3-Groq-8B-Tool-Use-GGUF/resolve/main/Llama-3-Groq-8B-Tool-Use-Q8_0.gguf?download=true", to: "llama_tools.gguf")
        let session = try await LlamaChainTools(params: params)
        var output = try await session.infer("I would like to get a product")
        print(output)
        output = try await session.infer("I would like an apple")
        print(output)
        if output.contains("cart") {
            output = try await session.infer("Yes")
        } else {
            output = try await session.infer("I would like red delicious")
        }
        print(output)
    }
}
