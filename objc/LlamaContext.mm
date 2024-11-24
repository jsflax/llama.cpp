#import <Foundation/Foundation.h>
#import "LlamaContext_Private.hpp"
#import "GGMLThreadpool_Private.hpp"
#import "GPTParams_Private.hpp"
#import "LlamaModel_Private.hpp"
#import "LlamaBatch_Private.hpp"
#import "../../common/common.h"


static void batch_decode(llama_context * ctx, llama_batch & batch, float * output, int n_seq, int n_embd, int embd_norm) {
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    const struct llama_model * model = llama_get_model(ctx);

    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_cache_clear(ctx);

    // run model
    if (llama_model_has_encoder(model) && !llama_model_has_decoder(model)) {
        // encoder-only model
        if (llama_encode(ctx, batch) < 0) {
            [NSException raise:@"EncodingFailure" format:@"failed to encode"];
        }
    } else if (!llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
        // decoder-only model
        if (llama_decode(ctx, batch) < 0) {
            [NSException raise:@"DecodingFailure" format:@"failed to decode"];
        }
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        const float * embd = nullptr;
        int embd_pos = 0;

        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            // try to get token embeddings
            embd = llama_get_embeddings_ith(ctx, i);
            embd_pos = i;
            GGML_ASSERT(embd != NULL && "failed to get token embeddings");
        } else {
            // try to get sequence embeddings - supported only when pooling_type is not NONE
            embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            embd_pos = batch.seq_id[i][0];
            GGML_ASSERT(embd != NULL && "failed to get sequence embeddings");
        }

        float * out = output + embd_pos * n_embd;
        common_embd_normalize(embd, out, n_embd, embd_norm);
    }
}

@implementation LlamaContext {
    llama_context *ctx;
    GPTParams *params;
}

- (instancetype)initWithParams:(GPTParams *)params {
    self = [super init];
    if (self) {
        self->params = [params copy];
        auto llama_init = common_init_from_params([params params]);
        self->ctx = llama_init.context;
        self->_model = [[LlamaModel alloc] init: llama_init.model];
    }
    return self;
    
}

- (instancetype)initWithContext:(llama_context *)context
                          model:(llama_model *)model
                   commonParams:(GPTParams *)params {
    self = [super init];
    if (self) {
        ctx = context;
        self->_model = [[LlamaModel alloc] init:model];
        self->params = params;
    }
    return self;
}

- (void)dealloc
{
    llama_free(ctx);
    [self.model dealloc];
    [super dealloc];
}

- (void)attachThreadpool:(GGMLThreadpool *)threadpool
         threadpoolBatch:(GGMLThreadpool *)threadpoolBatch {
    llama_attach_threadpool(ctx, [threadpool threadpool], [threadpoolBatch threadpool]);
}


- (NSUInteger)nCtx {
    return llama_n_ctx(ctx);
}

- (LlamaPoolingType)poolingType {
    return LlamaPoolingType(llama_pooling_type(self->ctx));
}

- (BOOL)loadStateFile:(NSString *)pathSession
            tokensOut:(llama_token *)tokensOut
        nTokenCpacity:(size_t)nTokenCapacity
       nTokenCountOut:(size_t *)nTokenCountOut {
    return llama_state_load_file(ctx, [pathSession cStringUsingEncoding:NSUTF8StringEncoding], tokensOut, nTokenCapacity, nTokenCountOut);
}

- (NSArray<NSNumber *> *)tokenize:(NSString *)text
                       addSpecial:(BOOL)addSpecial
                     parseSpecial:(BOOL)parseSpecial {
    NSMutableArray<NSNumber *> *tokens = [[NSMutableArray alloc] init];
    for (auto& token : [self cppTokenize:text addSpecial:addSpecial parseSpecial:parseSpecial]) {
        [tokens addObject:[[NSNumber alloc] initWithInt:token]];
    }
    return tokens;
}

- (std::vector<llama_token>)cppTokenize:(NSString *)text
                             addSpecial:(BOOL)addSpecial
                           parseSpecial:(BOOL)parseSpecial {
    return common_tokenize(ctx, [text cStringUsingEncoding:NSUTF8StringEncoding], addSpecial, parseSpecial);
}

- (std::string)convertTokensToString:(const std::vector<llama_token>&)tokens {
    return string_from(ctx, tokens);
}

- (llama_context *)cContext {
    return ctx;
}

- (int32_t)encode:(llama_batch)batch {
    return llama_encode(ctx, batch);
}

- (void)kvCacheSeqAdd:(LlamaSequenceId)sequenceId
                   p0:(LlamaPosition)p0
                   p1:(LlamaPosition)p1
                delta:(LlamaPosition)delta {
    llama_kv_cache_seq_add(ctx, sequenceId, p0, p1, delta);
}

- (void)kvCacheSeqDiv:(LlamaSequenceId)sequenceId
                   p0:(LlamaPosition)p0
                   p1:(LlamaPosition)p1
                delta:(LlamaPosition)delta {
    llama_kv_cache_seq_div(ctx, sequenceId, p0, p1, delta);
}

- (BOOL)kvCacheSeqRm:(LlamaSequenceId)sequenceId
                  p0:(LlamaPosition)p0
                  p1:(LlamaPosition)p1 {
    return llama_kv_cache_seq_rm(ctx, sequenceId, p0, p1);
}

- (NSString *)tokenToPiece:(LlamaToken)token {
    return [self tokenToPiece:token special:YES];
}

- (NSString *)tokenToPiece:(LlamaToken)token special:(BOOL)special {
    return [[NSString alloc] initWithCString:common_token_to_piece(ctx, token, special).c_str() encoding:NSUTF8StringEncoding];
}

- (NSInteger)decode:(LlamaBatch *)batch {
    return llama_decode(ctx, [batch cBatch]);
}

- (BOOL)saveStateFile:(NSString *)pathSession
               tokens:(const LlamaToken *)tokens
          nTokenCount:(size_t)nTokenCount {
    return llama_state_save_file(ctx,
                                 [pathSession cStringUsingEncoding:NSUTF8StringEncoding],
                                 tokens, nTokenCount);
}

- (void)decode:(LlamaBatch *)batch output:(float *)output nSeq:(int)nSeq nEmbd:(int)nEmbd embdNorm:(int)embdNorm {
    batch_decode(ctx, [batch cBatch], output, nSeq, nEmbd, embdNorm);
}

- (void)reset {
    if (ctx) {
        llama_free(ctx);
    }
    
    ctx = llama_new_context_with_model([self.model cModel], common_context_params_to_llama([params params]));
    if (!ctx) {
        [NSException raise:@"ContextReinitFailure" format:@"Failed to re-initialize the context"];
    }
}
@end
