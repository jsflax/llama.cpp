import Foundation
import LlamaObjC

public actor LlamaEmbeddingSession {
    let parameters: GPTParams
    let context: LlamaContext
    
    public init(parameters: GPTParams) {
        parameters.nBatch = Int32(parameters.nCtx)
        parameters.nUBatch = parameters.nBatch
        parameters.embedding = true
        parameters.poolingType = .cls
        parameters.logging = false
        self.parameters = parameters
        context = LlamaContext(params: self.parameters)
        llama_backend_init()
        llama_numa_init(ggml_numa_strategy(UInt32(parameters.numaStrategy)));
    }
    
    public func embeddings(for text: String, isQuery: Bool) -> [Float] {
        let nCtx = context.nCtx()
        let nCtxTrain = context.model.nCtxTrain()
        let nBatch = parameters.nBatch
        precondition(parameters.nBatch >= parameters.nCtx)
        let text = isQuery ? "search_query: \(text)" : "search_document: \(text)"
        let inputs = context.tokenize(text, addSpecial: true, parseSpecial: true)
        let batch = LlamaBatch(nBatch, embd: 0, nSeqMax: 1)
        let nEmbdCount: Int
        let poolingType = context.poolingType
        if context.model.hasEncoder() && context.model.hasDecoder() {
            assertionFailure("computing embeddings in encoder-decoder models is not supported\n")
        }
        if nCtx > nCtxTrain {
            print("warning: model was trained on only %d context tokens (%d specified)", nCtxTrain, nCtx);
        }
        if poolingType == .none {
            nEmbdCount = inputs.count
        } else {
            nEmbdCount = 1
        }
        let nEmbd = context.model.nEmbd()
        var embeddings: [Float] = [Float].init(repeating: 0, count: nEmbdCount * Int(nEmbd))
        
        embeddings.withUnsafeMutableBufferPointer { ptr in
            let embeddings = ptr.baseAddress!
            var e = 0 // number of embeddings already stored
            var s = 0 // number of prompts in current batch
            let nToks = inputs.count
            if Int(batch.nTokens) + nToks > nBatch {
                let out = embeddings + e * Int(nEmbd)
                context.decode(batch, output: out, nSeq: Int32(s), nEmbd: nEmbd, embdNorm: parameters.embdNormalize)
                e += poolingType == .none ? batch.nTokens : s
                batch.clear()
            }
            
            batch.addSequence(inputs, sequenceId: s)
            s += 1
            
            let out = embeddings + e * Int(nEmbd)
            context.decode(batch, output: out, nSeq: Int32(s), nEmbd: nEmbd, embdNorm: parameters.embdNormalize)
        }
        return embeddings
    }
}
