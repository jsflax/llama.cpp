#import <Foundation/Foundation.h>
#import "LlamaBatch_Private.hpp"
#import "llama.h"
#import "../../common/common.h"

@implementation LlamaBatch {
    llama_batch batch;
}

- (instancetype)initWithBatch:(llama_batch)batch {
    self->batch = batch;
}

- (instancetype)init:(int32_t)nTokens embd:(int32_t)embd nSeqMax:(int32_t)nSeqMax {
    self = [super init];
    if (self) {
        self->batch = llama_batch_init(nTokens, embd, nSeqMax);
    }
    return self;

}
- (NSData *)output {
    return [[NSData alloc] initWithBytes:batch.logits length:batch.n_tokens];
}

- (llama_batch&)cBatch {
    return batch;
}

- (void)addSequence:(NSArray<NSNumber *> *)tokens sequenceId:(LlamaSequenceId)sequenceId {
    size_t n_tokens = [tokens count];
    for (size_t i = 0; i < n_tokens; i++) {
        common_batch_add(batch, [tokens[i] intValue], i, { static_cast<int32_t>(sequenceId) }, true);
    }
}
- (void)clear {
    common_batch_clear(batch);
}

- (void)dealloc
{
    llama_batch_free(batch);
    [super dealloc];
}

@end
